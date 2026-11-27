"""Grammar compilation and per-step bitmask matching backends (`xgrammar` primary, `llguidance` fallback)."""

from __future__ import annotations

import importlib.util
import json
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch

from freetoken.core import StructuredOutputSpec
from freetoken.utils import init_logger

logger = init_logger(__name__)


class GrammarUnavailableError(RuntimeError):
    """Raised when grammar-constrained decoding is requested without an installed backend."""


def is_xgrammar_installed() -> bool:
    return importlib.util.find_spec("xgrammar") is not None


def is_llguidance_installed() -> bool:
    return importlib.util.find_spec("llguidance") is not None


def is_grammar_available() -> bool:
    """Return True if either ``xgrammar`` or ``llguidance`` is importable."""
    return is_xgrammar_installed() or is_llguidance_installed()


class GrammarMatcher(ABC):
    """Narrow per-request stateful matcher interface consumed by Sampler and Scheduler."""

    @abstractmethod
    def fill_next_token_bitmask(self, bitmask: torch.Tensor, row_idx: int) -> None:
        """Write allowed token bits for the current grammar state into ``bitmask[row_idx]``."""

    @abstractmethod
    def accept_token(self, token_id: int) -> bool:
        """Advance the matcher by one token. Returns False if the token is illegal."""

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """Roll back the last ``num_tokens`` accepted tokens (speculative verify rollback)."""

    @abstractmethod
    def is_terminated(self) -> bool:
        """Return True if the grammar has reached a terminal state with no further tokens."""


class GrammarBackend(ABC):
    """Abstract grammar compiler backend."""

    name: str

    @abstractmethod
    def compile(self, spec: StructuredOutputSpec) -> Any:
        """Compile ``spec`` into a backend-specific immutable compiled grammar object."""

    @abstractmethod
    def create_matcher(
        self, compiled: Any, *, max_rollback_tokens: int = 32
    ) -> GrammarMatcher:
        """Instantiate a fresh stateful ``GrammarMatcher`` from a compiled grammar."""


# --------------------------------------------------------------------------- #
# xgrammar backend (primary: supports JSON Schema, regex, EBNF, StructuralTag)
# --------------------------------------------------------------------------- #


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    """Return the token id if ``text`` maps cleanly to a single token, else None."""
    if not text or tokenizer is None:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        tid = convert(text)
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            ids = encode(text, add_special_tokens=False)
            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], int):
                return ids[0]
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _tag_boundary(tokenizer: Any, text: str, st: Any) -> Any:
    """Use ``TokenFormat(token=id)`` when ``text`` is a single special/vocab token, else literal string."""
    tid = _single_token_id(tokenizer, text)
    if tid is not None:
        return st.TokenFormat(token=tid)
    return text


_PARSER_XML_STYLE: dict[str, str] = {
    "qwen3_coder": "qwen_xml",
    "deepseekv32": "deepseek_xml",
    "deepseekv41": "deepseek_v4_1_xml",
    "glm47": "glm_xml",
    "minimax": "minimax_xml",
    "minimax_m3": "minimax_m3_xml",
    "gemma4": "gemma",
}


def _tool_tag_spec(
    parser: str, tool: dict[str, Any], tokenizer: Any, st: Any
) -> tuple[str, Any]:
    """Build ``(trigger_str, TagFormat)`` for one tool under ``parser``."""
    name = str(tool["name"])
    params = tool.get("parameters") or {"type": "object"}

    if parser == "qwen3_coder":
        opener, closer = "<tool_call>", "</tool_call>"
        content = st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
                st.ConstStringFormat(value=f"<function={name}>\n"),
                st.JSONSchemaFormat(json_schema=params, style="qwen_xml"),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
                st.ConstStringFormat(value="</function>"),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
            ]
        )
        return opener, st.TagFormat(
            begin=_tag_boundary(tokenizer, opener, st),
            content=content,
            end=_tag_boundary(tokenizer, closer, st),
        )

    if parser in ("deepseekv32", "deepseekv41"):
        style = _PARSER_XML_STYLE[parser]
        if parser == "deepseekv41":
            opener, closer = "<｜DSML｜ calls>", "</｜DSML｜ calls>"
            inv_open, inv_close = (
                f'<｜DSML｜ invoke name="{name}">\n',
                "\n</｜DSML｜ invoke>",
            )
        else:
            opener, closer = "<｜DSML｜function_calls>", "</｜DSML｜function_calls>"
            inv_open, inv_close = (
                f'<｜DSML｜invoke name="{name}">\n',
                "\n</｜DSML｜invoke>",
            )
        content = st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
                st.ConstStringFormat(value=inv_open),
                st.JSONSchemaFormat(json_schema=params, style=style),
                st.ConstStringFormat(value=inv_close),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
            ]
        )
        return opener, st.TagFormat(
            begin=_tag_boundary(tokenizer, opener, st),
            content=content,
            end=_tag_boundary(tokenizer, closer, st),
        )

    if parser == "glm47":
        opener, closer = "<tool_call>", "</tool_call>"
        content = st.SequenceFormat(
            elements=[
                st.ConstStringFormat(value=f"{name}\n"),
                st.JSONSchemaFormat(json_schema=params, style="glm_xml"),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
            ]
        )
        return opener, st.TagFormat(
            begin=_tag_boundary(tokenizer, opener, st),
            content=content,
            end=_tag_boundary(tokenizer, closer, st),
        )

    if parser == "gemma4":
        opener, closer = "<|tool_call>", "<tool_call|>"
        content = st.SequenceFormat(
            elements=[
                st.ConstStringFormat(value=f"call:{name}"),
                st.JSONSchemaFormat(json_schema=params, style="gemma"),
            ]
        )
        return opener, st.TagFormat(
            begin=_tag_boundary(tokenizer, opener, st),
            content=content,
            end=_tag_boundary(tokenizer, closer, st),
        )

    if parser == "minimax":
        opener, closer = "<minimax:tool_call>", "</minimax:tool_call>"
        content = st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
                st.ConstStringFormat(value=f'<invoke name="{name}">\n'),
                st.JSONSchemaFormat(json_schema=params, style="minimax_xml"),
                st.ConstStringFormat(value="\n</invoke>"),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
            ]
        )
        return opener, st.TagFormat(
            begin=_tag_boundary(tokenizer, opener, st),
            content=content,
            end=_tag_boundary(tokenizer, closer, st),
        )

    if parser == "minimax_m3":
        opener, closer = "]<]minimax[>[<tool_call>", "]<]minimax[>[</tool_call>"
        content = st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
                st.ConstStringFormat(value=f']<]minimax[>[<invoke name="{name}">\n'),
                st.JSONSchemaFormat(json_schema=params, style="minimax_m3_xml"),
                st.ConstStringFormat(value="\n]<]minimax[>[</invoke>"),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
            ]
        )
        return opener, st.TagFormat(
            begin=opener,
            content=content,
            end=closer,
        )

    if parser in ("gpt-oss", "gpt_oss"):
        opener = (
            f"<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>"
        )
        closer = "<|call|>"
        return "<|channel|>commentary to=functions.", st.TagFormat(
            begin=opener,
            content=st.JSONSchemaFormat(json_schema=params),
            end=_tag_boundary(tokenizer, closer, st),
        )

    if parser == "llama3":
        opener = "<|python_tag|>"
        call_schema = {
            "type": "object",
            "properties": {"name": {"const": name}, "arguments": params},
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        return opener, st.TagFormat(
            begin=_tag_boundary(tokenizer, opener, st),
            content=st.JSONSchemaFormat(json_schema=call_schema),
            end="",
        )

    if parser == "mistral":
        opener = "[TOOL_CALLS]"
        call_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": params},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            },
        }
        return opener, st.TagFormat(
            begin=_tag_boundary(tokenizer, opener, st),
            content=st.SequenceFormat(
                elements=[
                    st.RegexFormat(pattern=r"[ \t\r\n]*"),
                    st.JSONSchemaFormat(json_schema=call_schema),
                ]
            ),
            end="",
        )

    # Default JSON tool call envelope: qwen / qwen25 / ricursive / generic
    opener, closer = "<tool_call>", "</tool_call>"
    call_schema = {
        "type": "object",
        "properties": {"name": {"const": name}, "arguments": params},
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }
    return opener, st.TagFormat(
        begin=_tag_boundary(tokenizer, opener, st),
        content=st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
                st.JSONSchemaFormat(json_schema=call_schema),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
            ]
        ),
        end=_tag_boundary(tokenizer, closer, st),
    )


def _build_xgrammar_format(
    payload_obj: dict[str, Any],
    tokenizer: Any,
    st: Any,
    stop_token_ids: list[int] | None = None,
) -> Any:
    mode = payload_obj.get("mode")
    if mode == "optional_think":
        think_open = str(payload_obj["think_open"])
        think_close = str(payload_obj["think_close"])
        inner = payload_obj["inner"]
        ikind = inner["kind"]
        ipay = inner.get("payload", "")
        if ikind == "json_object":
            inner_fmt = st.JSONSchemaFormat(json_schema={"type": "object"})
        elif ikind == "json_schema":
            inner_fmt = st.JSONSchemaFormat(json_schema=json.loads(ipay))
        elif ikind == "regex":
            inner_fmt = st.RegexFormat(pattern=ipay)
        elif ikind == "ebnf":
            inner_fmt = st.GrammarFormat(grammar=ipay)
        elif ikind == "structural_tag":
            inner_fmt = _build_xgrammar_format(
                json.loads(ipay), tokenizer, st, stop_token_ids
            )
        else:
            raise ValueError(f"Unsupported inner grammar kind: {ikind!r}")
        open_tid = _single_token_id(tokenizer, think_open)
        close_tid = _single_token_id(tokenizer, think_close)
        if open_tid is not None and close_tid is not None:
            excluded = list(dict.fromkeys([*(stop_token_ids or []), close_tid]))
            think_tag = st.TagFormat(
                begin=st.TokenFormat(token=open_tid),
                content=st.StarFormat(
                    content=st.ExcludeTokenFormat(exclude_tokens=excluded)
                ),
                end=st.TokenFormat(token=close_tid),
            )
        else:
            think_tag = st.TagFormat(
                begin=_tag_boundary(tokenizer, think_open, st),
                content=st.AnyTextFormat(excludes=[think_close]),
                end=think_close,
            )
        return st.SequenceFormat(
            elements=[
                st.OptionalFormat(content=think_tag),
                st.RegexFormat(pattern=r"[ \t\r\n]*"),
                inner_fmt,
            ]
        )

    if mode == "tool_call":
        parser = str(payload_obj.get("parser") or "qwen25")
        at_least_one = bool(payload_obj.get("at_least_one", False))
        stop_after_first = bool(payload_obj.get("stop_after_first", False))
        tools = payload_obj.get("tools") or []
        tags = []
        triggers: list[str] = []
        for tool in tools:
            trig, tag_fmt = _tool_tag_spec(parser, tool, tokenizer, st)
            tags.append(tag_fmt)
            if trig not in triggers:
                triggers.append(trig)

        if at_least_one:
            # Forced tool_choice ("required" / "any" / named): must emit valid tool call(s)
            # immediately (allowing leading whitespace/newlines after </think>).
            one_call = tags[0] if len(tags) == 1 else st.OrFormat(elements=tags)
            if stop_after_first:
                body = one_call
            else:
                body = st.PlusFormat(
                    content=st.SequenceFormat(
                        elements=[one_call, st.RegexFormat(pattern=r"[ \t\r\n]*")]
                    )
                )
            return st.SequenceFormat(
                elements=[st.RegexFormat(pattern=r"[ \t\r\n]*"), body]
            )

        # Auto mode with strict: true tools: leave normal text free, trigger constraint on opener.
        token_triggers = [_single_token_id(tokenizer, t) for t in triggers]
        if token_triggers and all(tid is not None for tid in token_triggers):
            return st.TokenTriggeredTagsFormat(
                trigger_tokens=[int(tid) for tid in token_triggers if tid is not None],
                tags=tags,
                at_least_one=False,
                stop_after_first=stop_after_first,
            )
        return st.TriggeredTagsFormat(
            triggers=triggers,
            tags=tags,
            at_least_one=False,
            stop_after_first=stop_after_first,
        )

    raise ValueError(f"Unsupported structural_tag mode: {mode!r}")


class _XGrammarMatcher(GrammarMatcher):
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def fill_next_token_bitmask(self, bitmask: torch.Tensor, row_idx: int) -> None:
        self._inner.fill_next_token_bitmask(bitmask, row_idx)

    def accept_token(self, token_id: int) -> bool:
        return bool(self._inner.accept_token(int(token_id)))

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self._inner.rollback(int(num_tokens))

    def is_terminated(self) -> bool:
        return bool(self._inner.is_terminated())


class XGrammarBackend(GrammarBackend):
    name = "xgrammar"

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        import xgrammar

        self._xg = xgrammar
        self._tokenizer = tokenizer
        self._vocab_size = int(vocab_size)
        stops = [
            int(t) for t in (stop_token_ids or []) if 0 <= int(t) < self._vocab_size
        ]
        self._stop_token_ids = stops
        if isinstance(tokenizer, xgrammar.TokenizerInfo):
            self._tokenizer_info = tokenizer
        else:
            try:
                self._tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(
                    tokenizer, vocab_size=self._vocab_size, stop_token_ids=stops or None
                )
            except Exception:  # noqa: BLE001 -- fallback for minimal/stub tokenizers in unit tests
                vocab = self._extract_encoded_vocab(tokenizer, self._vocab_size)
                self._tokenizer_info = xgrammar.TokenizerInfo(
                    vocab, vocab_size=self._vocab_size, stop_token_ids=stops or None
                )
        # Disable xgrammar's internal cache (known collision across structural tag variants);
        # GrammarCompiler maintains an explicit canonical LRU cache instead.
        self._compiler = xgrammar.GrammarCompiler(
            self._tokenizer_info, cache_enabled=False
        )

    @staticmethod
    def _extract_encoded_vocab(tokenizer: Any, vocab_size: int) -> list[str]:
        vocab = [""] * vocab_size
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            for tok, idx in get_vocab().items():
                if isinstance(idx, int) and 0 <= idx < vocab_size:
                    vocab[idx] = str(tok)
            return vocab
        decode = getattr(tokenizer, "decode", None)
        if callable(decode):
            for idx in range(vocab_size):
                try:
                    vocab[idx] = str(decode([idx]))
                except Exception:  # noqa: BLE001
                    vocab[idx] = ""
        return vocab

    def compile(self, spec: StructuredOutputSpec) -> Any:
        if spec.kind == "json_object":
            return self._compiler.compile_builtin_json_grammar()
        if spec.kind == "json_schema":
            return self._compiler.compile_json_schema(
                spec.payload, strict_mode=spec.strict, any_whitespace=True
            )
        if spec.kind == "regex":
            return self._compiler.compile_regex(spec.payload)
        if spec.kind == "ebnf":
            return self._compiler.compile_grammar(spec.payload)
        if spec.kind == "structural_tag":
            from xgrammar import structural_tag as st

            payload_obj = json.loads(spec.payload)
            fmt = _build_xgrammar_format(
                payload_obj, self._tokenizer, st, self._stop_token_ids
            )
            return self._compiler.compile_structural_tag(st.StructuralTag(format=fmt))
        raise ValueError(f"Unsupported StructuredOutputSpec.kind: {spec.kind!r}")

    def create_matcher(
        self, compiled: Any, *, max_rollback_tokens: int = 32
    ) -> GrammarMatcher:
        del max_rollback_tokens
        inner = self._xg.GrammarMatcher(compiled)
        return _XGrammarMatcher(inner)


# --------------------------------------------------------------------------- #
# llguidance fallback backend (when xgrammar is absent and llguidance is present)
# --------------------------------------------------------------------------- #


class _LLGuidanceMatcher(GrammarMatcher):
    def __init__(self, inner: Any) -> None:
        import llguidance.torch as llt

        self._inner = inner
        self._llt = llt

    def fill_next_token_bitmask(self, bitmask: torch.Tensor, row_idx: int) -> None:
        self._llt.fill_next_token_bitmask(self._inner, bitmask, index=row_idx)

    def accept_token(self, token_id: int) -> bool:
        return int(self._inner.try_consume_tokens([int(token_id)])) == 1

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self._inner.rollback(int(num_tokens))

    def is_terminated(self) -> bool:
        return bool(self._inner.is_stopped())


class LLGuidanceBackend(GrammarBackend):
    name = "llguidance"

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        import llguidance
        import llguidance.hf as llhf

        self._llg = llguidance
        self._vocab_size = int(vocab_size)
        stops = [
            int(t) for t in (stop_token_ids or []) if 0 <= int(t) < self._vocab_size
        ]
        self._stop_token_ids = stops
        self._ll_tok = llhf.from_tokenizer(
            tokenizer, n_vocab=self._vocab_size, eos_token=stops or None
        )

    def compile(self, spec: StructuredOutputSpec) -> str:
        if spec.kind == "json_object":
            return self._llg.LLMatcher.grammar_from_json_schema('{"type":"object"}')
        if spec.kind == "json_schema":
            return self._llg.LLMatcher.grammar_from_json_schema(spec.payload)
        if spec.kind == "regex":
            return self._llg.LLMatcher.grammar_from_regex(spec.payload)
        if spec.kind == "ebnf":
            return self._llg.LLMatcher.grammar_from_lark(
                self._llg.gbnf_to_lark(spec.payload)
            )
        raise ValueError(
            f"llguidance backend does not support kind={spec.kind!r}; install xgrammar via `pip install 'freetoken[grammar]'`"
        )

    def create_matcher(
        self, compiled: Any, *, max_rollback_tokens: int = 32
    ) -> GrammarMatcher:
        del max_rollback_tokens
        matcher = self._llg.LLMatcher(self._ll_tok, compiled)
        if matcher.is_error():
            raise ValueError(f"llguidance grammar error: {matcher.get_error()}")
        return _LLGuidanceMatcher(matcher)


# --------------------------------------------------------------------------- #
# Caching, async-compiling front-end (used by Scheduler on TP rank 0)
# --------------------------------------------------------------------------- #


class GrammarCompiler:
    """Thread-safe LRU-cached grammar compiler with background compilation executor."""

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        stop_token_ids: list[int] | None = None,
        *,
        backend: GrammarBackend | None = None,
        cache_size: int = 512,
        max_workers: int = 2,
    ) -> None:
        if backend is None:
            if is_xgrammar_installed():
                backend = XGrammarBackend(tokenizer, vocab_size, stop_token_ids)
            elif is_llguidance_installed():
                backend = LLGuidanceBackend(tokenizer, vocab_size, stop_token_ids)
            else:
                raise GrammarUnavailableError(
                    "Grammar-constrained decoding requires xgrammar or llguidance; "
                    "install with: pip install 'freetoken[grammar]'"
                )
        self.backend = backend
        self._cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[str, str, bool], Any] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="ft-grammar"
        )

    @staticmethod
    def _cache_key(spec: StructuredOutputSpec) -> tuple[str, str, bool]:
        return (spec.kind, spec.payload, bool(spec.strict))

    def compile(self, spec: StructuredOutputSpec) -> Any:
        key = self._cache_key(spec)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        compiled = self.backend.compile(spec)
        with self._lock:
            self._cache[key] = compiled
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return compiled

    def compile_async(self, spec: StructuredOutputSpec) -> Future[Any]:
        key = self._cache_key(spec)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                fut: Future[Any] = Future()
                fut.set_result(hit)
                return fut
        return self._executor.submit(self.compile, spec)

    def create_matcher(
        self, compiled: Any, *, max_rollback_tokens: int = 32
    ) -> GrammarMatcher:
        return self.backend.create_matcher(
            compiled, max_rollback_tokens=max_rollback_tokens
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
