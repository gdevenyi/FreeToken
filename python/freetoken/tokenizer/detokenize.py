import json
import weakref
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List

from freetoken.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


def _stop_prefix_holdback(text: str, stop_strs: list[str]) -> int:
    """Length of the longest trailing suffix of ``text`` that is a proper prefix of some
    stop string. Those chars are withheld until a later token resolves whether the stop
    completes, so a partial stop is never streamed and then needs retracting."""
    hold = 0
    for stop in stop_strs:
        for i in range(min(len(stop) - 1, len(text)), 0, -1):
            if text.endswith(stop[:i]):
                hold = max(hold, i)
                break
    return hold


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # length of read ids
    surr_offset: int  # length of surr ids
    sent_offset: int  # length of sent out string
    skip_special: bool = False  # decode this request with skip_special_tokens
    think_end_at: int | None = None  # generated tokens up to and incl. the reasoning end tag
    in_reasoning: bool = False  # the prompt or the output opened a reasoning block
    generated: int = 0  # sampled tokens, EOS included
    output_len: int = 0  # decoded text so far, before holdback: where the next token starts

    def reasoning_tokens(self) -> int:
        if self.think_end_at is not None:
            return self.think_end_at
        # stopped inside the block (max_tokens, or EOS before the end tag): all of it
        return self.generated if self.in_reasoning else 0

    def update(self, read_str: str, surr_str: str) -> str:
        new_text = read_str[len(surr_str) :]
        if len(new_text) > 0 and not new_text.endswith("�"):
            self.decoded_str += new_text
            self.surr_offset = self.read_offset
            self.read_offset = len(self.decoded_ids)
            return self.decoded_str
        return self.decoded_str + find_printable_text(new_text)

    def decode(self, tokenizer: PreTrainedTokenizerBase) -> str:
        skip = self.skip_special
        return self.update(
            tokenizer.decode(self.decoded_ids[self.surr_offset :], skip_special_tokens=skip),
            tokenizer.decode(
                self.decoded_ids[self.surr_offset : self.read_offset], skip_special_tokens=skip
            ),
        )


class DetokenizeManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        eos_token_ids: FrozenSet[int] | None = None,
        think_end_id: int | None = None,
        think_start_id: int | None = None,
    ) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_ids = (
            eos_token_ids if eos_token_ids is not None else frozenset({tokenizer.eos_token_id})
        )
        # The reasoning end tag's id (e.g. ``</think>``): tokens generated up to and
        # including it are reported as usage.completion_tokens_details.reasoning_tokens.
        self.think_end_id = think_end_id if think_end_id is None else int(think_end_id)
        self.think_start_id = think_start_id if think_start_id is None else int(think_start_id)

    def discard(self, uid: int) -> None:
        """Drop a uid's decode state without a finished DetokenizeMsg. An aborted or errored
        request never sends one (its terminal reply is an ErrorReplyMsg), so without this its
        accumulated ids and text stay in ``decode_map`` for the life of the worker."""
        self.decode_map.pop(uid, None)

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        return self.detokenize_with_meta(msgs)[0]

    def _batch_decode(self, ids: List[List[int]], skip: List[bool]) -> List[str]:
        """batch_decode honouring a per-request skip_special_tokens flag (one call per
        distinct flag value, results back in order)."""
        out: List[str | None] = [None] * len(ids)
        for flag in (False, True):
            idx = [i for i, f in enumerate(skip) if f == flag]
            if idx:
                texts = self.tokenizer.batch_decode([ids[i] for i in idx], skip_special_tokens=flag)
                for i, t in zip(idx, texts, strict=True):
                    out[i] = t
        return [t if t is not None else "" for t in out]

    def detokenize_with_meta(
        self, msgs: List[DetokenizeMsg]
    ) -> tuple[List[str], List[int], List[int]]:
        """Incremental text per message; for a finished message, its reasoning token count
        (0 otherwise or when the generation never was in a reasoning block); and where each
        message's token starts in the generated text. That offset ignores stop-string
        holdback, which delays when text is sent but not where it lands."""
        # Each message must advance its request's decode state before the next one.
        if len({msg.uid for msg in msgs}) != len(msgs):
            parts = [self.detokenize_with_meta([msg]) for msg in msgs]
            return (
                [p[0][0] for p in parts],
                [p[1][0] for p in parts],
                [p[2][0] for p in parts],
            )
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        skip: List[bool] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                    skip_special=bool(getattr(msg, "skip_special_tokens", False)),
                    in_reasoning=bool(getattr(msg, "starts_in_reasoning", False)),
                )
            s = self.decode_map[msg.uid]
            s.generated += 1
            if not (
                msg.finished and not msg.matched_stop and msg.next_token in self.eos_token_ids
            ):
                s.decoded_ids.append(msg.next_token)
            if (
                self.think_end_id is not None
                and s.think_end_at is None
                and msg.next_token == self.think_end_id
            ):
                s.think_end_at = len(s.decoded_ids)
            if s.think_end_at is None and msg.next_token == self.think_start_id:
                s.in_reasoning = True
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])
            skip.append(s.skip_special)

        read_texts = self._batch_decode(read_ids, skip)
        surr_texts = self._batch_decode(surr_ids, skip)

        incremental_strs: List[str] = []
        reasoning_tokens: List[int] = []
        text_offsets: List[int] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            text_offsets.append(s.output_len)
            output_str = s.update(read_str, surr_str)
            s.output_len = max(s.output_len, len(output_str))

            prev_sent = s.sent_offset
            if msg.finished:
                # Generation is over: flush everything, trimming at the matched stop string
                # (or keeping it, for include_stop_str_in_output).
                if msg.matched_stop:
                    cut = output_str.find(msg.matched_stop)
                    if cut < 0:
                        emit_end = len(output_str)
                    elif getattr(msg, "keep_stop_str", False):
                        emit_end = cut + len(msg.matched_stop)
                    else:
                        emit_end = cut
                else:
                    emit_end = len(output_str)
            elif msg.stop_strs:
                # Hold back a trailing suffix that could still grow into a stop string.
                emit_end = len(output_str) - _stop_prefix_holdback(output_str, msg.stop_strs)
            else:
                emit_end = len(output_str)
            incremental_output = output_str[prev_sent:emit_end] if emit_end > prev_sent else ""
            s.sent_offset = max(prev_sent, emit_end)
            incremental_strs.append(incremental_output)
            reasoning_tokens.append(s.reasoning_tokens() if msg.finished else 0)
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs, reasoning_tokens, text_offsets


def _gpt2_byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's ``bytes_to_unicode``: the printable stand-in char -> its byte."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class _TokenBytes:
    """A token id's own bytes. A single-id decode cannot give them for a piece of a
    multi-byte character (it decodes to U+FFFD), so read the vocab piece instead."""

    def __init__(self, tokenizer: Any) -> None:
        decoder: dict = {}
        backend = getattr(tokenizer, "backend_tokenizer", None)
        try:
            decoder = json.loads(backend.decoder.__getstate__())
        except Exception:  # noqa: BLE001 -- no fast backend: fall back to the decoded text
            pass
        kinds = {d.get("type") for d in decoder.get("decoders", [decoder])}
        self.byte_level = _gpt2_byte_decoder() if "ByteLevel" in kinds else None
        self.byte_fallback = "ByteFallback" in kinds
        added = getattr(tokenizer, "added_tokens_decoder", None) or {}
        self.added = {int(tid): str(getattr(tok, "content", tok)) for tid, tok in added.items()}
        self.convert = getattr(tokenizer, "convert_ids_to_tokens", None)

    def __call__(self, tid: int, text: str) -> bytes:
        if tid in self.added:
            # the GPT-2 table cannot invert an added token's raw (e.g. full-width) chars
            return self.added[tid].encode("utf-8")
        try:
            piece = self.convert(tid) if self.convert is not None else None
        except Exception:  # noqa: BLE001 -- out-of-vocab id
            piece = None
        if isinstance(piece, str):
            if self.byte_level is not None:
                try:
                    return bytes(self.byte_level[c] for c in piece)
                except KeyError:
                    pass
            elif self.byte_fallback and len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    return bytes([int(piece[3:5], 16)])
                except ValueError:
                    pass
        return text.encode("utf-8")


_TOKEN_BYTES: "weakref.WeakKeyDictionary[Any, _TokenBytes]" = weakref.WeakKeyDictionary()


def _token_bytes(tokenizer: Any) -> _TokenBytes:
    try:
        cached = _TOKEN_BYTES.get(tokenizer)
        if cached is None:
            cached = _TOKEN_BYTES[tokenizer] = _TokenBytes(tokenizer)
    except TypeError:  # not weak-referenceable or unhashable
        return _TokenBytes(tokenizer)
    return cached


def build_logprobs_entry(
    tokenizer: PreTrainedTokenizerBase,
    token_id: int,
    chosen_logprob: float,
    top_ids: list[int] | None,
    top_logprobs: list[float] | None,
) -> dict:
    """Neutral sampled-token logprobs entry (UserReply.logprobs). Values are the raw
    pre-temperature logprobs the sampler computed; token text comes from a single-id
    decode, and ``bytes`` are the token's own bytes, so clients can reassemble a
    character split across tokens. Ids past the real vocab (the padded tail of the
    logits) report an empty token."""
    token_bytes = _token_bytes(tokenizer)

    def _fields(tid: int) -> tuple[str, list[int]]:
        try:
            text = tokenizer.decode([tid])
        except Exception:  # noqa: BLE001 -- a padded id some tokenizers refuse to decode
            text = ""
        return text, list(token_bytes(tid, text))

    token, data = _fields(token_id)
    top = []
    for tid, logprob in zip(top_ids or [], top_logprobs or []):
        text, cand = _fields(int(tid))
        top.append({"token_id": int(tid), "token": text, "bytes": cand, "logprob": float(logprob)})
    return {
        "token_id": token_id,
        "token": token,
        "bytes": data,
        "logprob": float(chosen_logprob),
        "top": top,
    }
