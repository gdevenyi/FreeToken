"""OpenAI logprobs formatting: neutral engine entries -> wire shapes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


def chat_logprobs_error(req: Any) -> str | None:
    if req.top_logprobs is not None and not req.logprobs:
        return "top_logprobs requires logprobs=true"
    if req.top_logprobs is not None and not 0 <= req.top_logprobs <= 20:
        return "top_logprobs must be between 0 and 20"
    return None


def completion_logprobs_error(req: Any) -> str | None:
    if req.logprobs is not None and not 0 <= req.logprobs <= 5:
        return "logprobs must be between 0 and 5"
    if req.echo and req.logprobs is not None:
        return "echo with logprobs is not supported"
    return None


def untrimmed(entries: list[dict] | None) -> list[dict]:
    """A reply's entries minus those of tokens a matched stop string cut from the text."""
    return [entry for entry in entries or () if not entry.get("stop_trimmed")]


def chat_content_entry(entry: dict) -> dict:
    return {
        "token": entry["token"],
        "logprob": entry["logprob"],
        "bytes": entry["bytes"],
        "top_logprobs": [
            {
                "token": candidate["token"],
                "logprob": candidate["logprob"],
                "bytes": candidate["bytes"],
            }
            for candidate in entry["top"]
        ],
    }


def completions_logprobs(entries: list[dict], start_offset: int = 0) -> dict:
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []
    text_offset: list[int] = []
    offset = start_offset

    for entry in entries:
        token = entry["token"]
        tokens.append(token)
        token_logprobs.append(entry["logprob"])
        top_logprobs.append({candidate["token"]: candidate["logprob"] for candidate in entry["top"]})
        text_offset.append(offset)
        offset += len(token)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


def chat_logprobs_object(entries: list[dict]) -> dict:
    return {"content": [chat_content_entry(entry) for entry in entries], "refusal": None}


@dataclass(frozen=True)
class LogprobMarkers:
    """Token ids the chat segmenter keys on, resolved once per server. A think marker is
    None unless the tokenizer encodes it as exactly one id that decodes back to it."""

    think_start: int | None
    think_end: int | None
    # Special and protocol-hidden tokens: never message content.
    hidden: frozenset[int]


def resolve_logprob_markers(
    tokenizer: Any, think_start: str, think_end: str, hidden_tokens: tuple[str, ...]
) -> LogprobMarkers:
    def single_id(marker: str) -> int | None:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) == 1 and tokenizer.decode(ids) == marker:
            return int(ids[0])
        return None

    hidden = {int(tid) for tid in getattr(tokenizer, "all_special_ids", None) or ()}
    hidden.update(tid for tid in map(single_id, hidden_tokens) if tid is not None)
    return LogprobMarkers(single_id(think_start), single_id(think_end), frozenset(hidden))


def _visible_text(entry: dict) -> str:
    """The part of an entry's token that reached the output text."""
    return entry["token"][: entry.get("stop_visible")]


class LogprobSegmenter:
    """Picks a chat generation's message-content logprob entries out of the sampled ones.

    A pure function of the entry sequence (token ids and single-token strings), so the
    streaming and non-streaming paths select the same entries for any ack chunking.
    The reasoning split is keyed on the think marker ids and mirrors the qwen3 parser:
    a leading ``<think>`` (after whitespace) opens reasoning once, the first ``</think>``
    ends it for good, and content after it is whitespace-trimmed at both ends. Tool
    markup is decided by a shadow tool-call parser fed one token string at a time: its
    held buffer is always a suffix of what it was fed, so every resolved character is
    either emitted as text or consumed as markup, and a token is content when any of
    its characters is emitted.
    """

    def __init__(
        self,
        markers: LogprobMarkers,
        *,
        reasoning: bool,
        force_reasoning: bool,
        tool_parser: Any | None = None,
    ) -> None:
        self._markers = markers
        self._state = "head" if reasoning else "content"
        self._force = force_reasoning
        self._trim = False  # content follows a real </think>: trim it at both ends
        self._content_started = False
        self._held: list[dict] = []  # whitespace-only tokens awaiting the next text
        self._tools = tool_parser
        self._queue: deque[list] = deque()  # [entry, start, end, is_content] being resolved
        self._fed = 0
        self._out: list[dict] = []

    def feed(self, entries: list[dict]) -> list[dict]:
        """Content entries that became final, in generation order."""
        for entry in entries:
            # Hidden ids skip the parsers whole: the final EOS adds no text to the output.
            if entry["token_id"] not in self._markers.hidden:
                self._split_reasoning(entry)
        out, self._out = self._out, []
        return out

    def finish(self) -> list[dict]:
        held, self._held = self._held, []
        if self._state == "head" and not self._force:
            for entry in held:
                self._content(entry)
        if self._tools is not None:
            det = self._tools.detector
            base = self._fed - len(det._buffer)
            region = det._buffer
            if self._tools.recover_truncated_call():
                self._resolve(base, region, "")
                base, region = self._fed, ""
            self._resolve(base, region, self._tools.finish_stream())
        out, self._out = self._out, []
        return out

    def _split_reasoning(self, entry: dict) -> None:
        # A token a stop string cut inside is only its visible text, never a marker.
        tid = -1 if "stop_visible" in entry else entry["token_id"]
        text = _visible_text(entry)
        blank = not text.strip()
        if self._state == "head":
            if tid == self._markers.think_start:
                self._held = []
                self._state = "reasoning"
            elif blank:
                self._held.append(entry)
            elif self._force:
                self._held = []
                self._state = "reasoning"
                self._split_reasoning(entry)
            else:
                self._state = "content"
                held, self._held = self._held, []
                for e in held + [entry]:
                    self._content(e)
        elif self._state == "reasoning":
            if tid == self._markers.think_end:
                self._state = "content"
                self._trim = True
        elif not self._trim:
            self._content(entry)
        elif blank:
            if self._content_started:
                self._held.append(entry)
        elif not self._content_started:
            # The reasoning parser strips this leading whitespace before the tool
            # parser sees it: " <" opening a bare call is markup, not a space.
            self._content_started = True
            self._content(entry, text.lstrip())
        else:
            held, self._held = self._held, []
            for e in held + [entry]:
                self._content(e)

    def _content(self, entry: dict, text: str | None = None) -> None:
        if self._tools is None:
            self._out.append(entry)
            return
        text = _visible_text(entry) if text is None else text
        held = self._tools.detector._buffer
        base = self._fed - len(held)
        self._queue.append([entry, self._fed, self._fed + len(text), False])
        self._fed += len(text)
        normal = "".join(
            payload for kind, payload in self._tools.parse_stream_events(text) if kind == "text"
        )
        resolved = self._fed - len(self._tools.detector._buffer) - base
        region = (held + text)[:resolved] if normal else ""
        self._resolve(base, region, normal, resolved)

    def _resolve(self, base: int, region: str, normal: str, length: int | None = None) -> None:
        """``region`` (``length`` chars from offset ``base``) left the parser's buffer,
        emitting ``normal``: mark the tokens it emitted from, then release resolved ones.
        ``region`` itself is only needed when something was emitted."""
        if normal:
            if region.startswith(normal):
                emitted = [(0, len(normal))]
            elif region.endswith(normal):
                # Text after markup in one token, as when a bare call's ">" merges with
                # the next character (">n"): a leftmost match could land inside the markup.
                emitted = [(len(region) - len(normal), len(region))]
            else:  # not seen in practice: fall back to the leftmost alignment
                emitted, i = [], 0
                for ch in normal:
                    j = region.find(ch, i)
                    if j == -1:
                        break
                    emitted.append((j, j + 1))
                    i = j + 1
            for item in self._queue:
                start, end = item[1] - base, item[2] - base
                if any(lo < end and start < hi for lo, hi in emitted):
                    item[3] = True
        resolved = base + (len(region) if length is None else length)
        while self._queue and self._queue[0][2] <= resolved:
            entry, _, _, is_content = self._queue.popleft()
            if is_content:
                self._out.append(entry)
