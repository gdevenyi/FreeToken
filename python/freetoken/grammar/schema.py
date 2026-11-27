"""Strict JSON Schema validation, reasoning deferral resolution, and protocol-neutral
``StructuredOutputSpec`` builders for OpenAI Chat/Completions/Responses and Anthropic Messages."""

from __future__ import annotations

import json
import re
from typing import Any

from freetoken.core import StructuredOutputSpec
from freetoken.tokenizer.tokenize import resolve_thinking_mode


class SchemaValidationError(ValueError):
    """Raised when a client supplies an invalid JSON Schema (mapped to HTTP 400 invalid_json_schema)."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.code = "invalid_json_schema"
        self.param = param


_VALID_JSON_TYPES = frozenset(
    {"object", "array", "string", "number", "integer", "boolean", "null"}
)


def canonical_schema_json(schema: dict[str, Any]) -> str:
    """Canonical compact JSON string used as a stable grammar cache key."""
    return json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _resolve_local_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise SchemaValidationError(
            f"Unsupported JSON Schema $ref {ref!r}: only local '#/$defs/...' or '#/definitions/...' refs are supported"
        )
    parts = ref[2:].split("/")
    cur: Any = root
    for part in parts:
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(cur, dict) or key not in cur:
            raise SchemaValidationError(f"Unresolvable JSON Schema $ref: {ref!r}")
        cur = cur[key]
    if not isinstance(cur, dict):
        raise SchemaValidationError(
            f"JSON Schema $ref {ref!r} must resolve to an object"
        )
    return cur


def _validate_node(
    node: Any,
    root: dict[str, Any],
    *,
    strict: bool,
    path: str,
    seen_refs: set[str],
    is_root: bool = False,
) -> None:
    if not isinstance(node, dict):
        raise SchemaValidationError(f"JSON Schema at {path} must be an object")

    if "$ref" in node:
        ref = node["$ref"]
        target = _resolve_local_ref(root, ref)
        if (
            is_root
            and strict
            and target.get("type") != "object"
            and "anyOf" not in target
        ):
            raise SchemaValidationError(
                "Strict JSON Schema root must be an object (type: 'object')"
            )
        if ref not in seen_refs:
            seen_refs.add(ref)
            _validate_node(
                target,
                root,
                strict=strict,
                path=ref,
                seen_refs=seen_refs,
                is_root=False,
            )
        return

    node_type = node.get("type")
    if node_type is not None:
        if isinstance(node_type, str):
            if node_type not in _VALID_JSON_TYPES:
                raise SchemaValidationError(
                    f"Unsupported JSON Schema type {node_type!r} at {path}"
                )
        elif isinstance(node_type, list):
            if not node_type or any(
                not isinstance(t, str) or t not in _VALID_JSON_TYPES for t in node_type
            ):
                raise SchemaValidationError(
                    f"Invalid JSON Schema type union {node_type!r} at {path}"
                )
        else:
            raise SchemaValidationError(
                f"Invalid JSON Schema type {node_type!r} at {path}"
            )

    if is_root and strict:
        is_obj_root = node_type == "object" or (
            isinstance(node_type, list) and "object" in node_type
        )
        if not is_obj_root and "anyOf" not in node:
            raise SchemaValidationError(
                "Strict JSON Schema root must have type 'object'"
            )

    is_object = (
        node_type == "object"
        or (isinstance(node_type, list) and "object" in node_type)
        or ("properties" in node)
    )
    if is_object:
        props = node.get("properties")
        if props is not None and not isinstance(props, dict):
            raise SchemaValidationError(
                f"JSON Schema 'properties' at {path} must be an object"
            )
        req_keys = node.get("required")
        if req_keys is not None and (
            not isinstance(req_keys, list)
            or any(not isinstance(k, str) for k in req_keys)
        ):
            raise SchemaValidationError(
                f"JSON Schema 'required' at {path} must be a list of strings"
            )
        if strict:
            if node.get("additionalProperties") is not False:
                raise SchemaValidationError(
                    f"Strict JSON Schema object at {path} must set 'additionalProperties': false"
                )
            if isinstance(props, dict) and props:
                req_set = set(req_keys or [])
                missing = [k for k in props if k not in req_set]
                if missing:
                    raise SchemaValidationError(
                        f"Strict JSON Schema object at {path} must list every property in 'required' "
                        f"(missing: {', '.join(missing)})"
                    )
        if isinstance(props, dict):
            for key, sub in props.items():
                _validate_node(
                    sub,
                    root,
                    strict=strict,
                    path=f"{path}.properties.{key}",
                    seen_refs=seen_refs,
                )

    if "items" in node:
        items = node["items"]
        if isinstance(items, dict):
            _validate_node(
                items,
                root,
                strict=strict,
                path=f"{path}.items",
                seen_refs=seen_refs,
            )
        elif isinstance(items, list):
            for i, sub in enumerate(items):
                _validate_node(
                    sub,
                    root,
                    strict=strict,
                    path=f"{path}.items[{i}]",
                    seen_refs=seen_refs,
                )
        else:
            raise SchemaValidationError(
                f"JSON Schema 'items' at {path} must be an object or array"
            )

    for comb in ("anyOf", "oneOf", "allOf"):
        if comb in node:
            branches = node[comb]
            if not isinstance(branches, list) or not branches:
                raise SchemaValidationError(
                    f"JSON Schema {comb!r} at {path} must be a non-empty array"
                )
            for i, sub in enumerate(branches):
                _validate_node(
                    sub,
                    root,
                    strict=strict,
                    path=f"{path}.{comb}[{i}]",
                    seen_refs=seen_refs,
                )

    for defs_key in ("$defs", "definitions"):
        defs = node.get(defs_key)
        if defs is not None:
            if not isinstance(defs, dict):
                raise SchemaValidationError(
                    f"JSON Schema {defs_key!r} at {path} must be an object"
                )
            for name, sub in defs.items():
                _validate_node(
                    sub,
                    root,
                    strict=strict,
                    path=f"{path}.{defs_key}.{name}",
                    seen_refs=seen_refs,
                )


def validate_json_schema(
    schema: Any, *, strict: bool = False, param: str | None = None
) -> dict[str, Any]:
    """Validate a JSON Schema dict, enforcing OpenAI/Anthropic strict-mode rules when ``strict=True``."""
    if not isinstance(schema, dict):
        raise SchemaValidationError("JSON Schema must be an object", param=param)
    try:
        _validate_node(
            schema, schema, strict=strict, path="$", seen_refs=set(), is_root=True
        )
    except SchemaValidationError as exc:
        if exc.param is None and param is not None:
            exc.param = param
        raise
    return schema


# --------------------------------------------------------------------------- #
# Reasoning tag resolution (shared with server/generation.py)
# --------------------------------------------------------------------------- #

_REASONING_TAGS: dict[str, tuple[str, str]] = {
    "qwen3": ("<think>", "</think>"),
    "glm": ("<think>", "</think>"),
    "minimax": ("<think>", "</think>"),
    "minimax_m3": ("<mm:think>", "</mm:think>"),
    "deepseekv32": ("<think>", "</think>"),
    "gemma4": ("<|channel>thought\n", "<channel|>"),
    "flint": ("<think>", "</think>"),
    "ricursive": ("<think>", "</think>"),
    "gpt_oss": ("<|channel|>analysis<|message|>", "<|end|>"),
    "muse_glimmer": ("to=self<|message|>", "<|eom|>"),
}


def resolve_force_reasoning(
    parser_name: str | None,
    chat_template_kwargs: dict[str, Any] | None,
    template_tools: list[dict[str, Any]] | None,
) -> bool:
    """Single source of truth for whether the chat template pre-opens a reasoning block."""
    ctk = chat_template_kwargs or {}
    if parser_name == "qwen3":
        return ctk.get("enable_thinking") is not False and not ctk.get(
            "continue_final_message"
        )
    if parser_name == "glm":
        return ctk.get("enable_thinking") is not False
    if parser_name == "minimax":
        return True
    if parser_name == "gemma4":
        return (
            ctk.get("thinking_mode") == "thinking"
            or bool(ctk.get("enable_thinking"))
            or bool(ctk.get("thinking"))
        )
    if parser_name in ("flint", "ricursive"):
        return False
    if parser_name == "minimax_m3":
        return ctk.get("thinking_mode") == "enabled"
    return resolve_thinking_mode(ctk, template_tools) == "thinking"


def resolve_reasoning_grammar_mode(
    parser_name: str | None,
    chat_template_kwargs: dict[str, Any] | None,
    template_tools: list[dict[str, Any]] | None,
) -> tuple[str | None, tuple[str, str] | None]:
    """Return ``(defer_until_tag, optional_think_tags)``.

    - Forced-reasoning models (`force_reasoning=True`): return `(close_tag, None)` so the
      scheduler leaves logits unconstrained during `<think>...</think>` and activates the
      grammar right after `close_tag`.
    - Optional-think models (`flint`, `ricursive`, `minimax_m3` adaptive): return
      `(None, (open_tag, close_tag))` so the grammar wraps an optional token-bounded think
      block before the constrained payload.
    - Non-reasoning or explicitly disabled thinking: return `(None, None)`.
    """
    if not parser_name or parser_name == "off":
        return None, None
    tags = _REASONING_TAGS.get(parser_name)
    if tags is None:
        return None, None
    open_tag, close_tag = tags
    if resolve_force_reasoning(parser_name, chat_template_kwargs, template_tools):
        return close_tag, None
    ctk = chat_template_kwargs or {}
    if parser_name in ("flint", "ricursive"):
        if (
            ctk.get("enable_thinking") is False
            or ctk.get("thinking_mode") == "disabled"
        ):
            return None, None
        return None, (open_tag, close_tag)
    if parser_name == "minimax_m3" and ctk.get("thinking_mode") in (None, "adaptive"):
        return None, (open_tag, close_tag)
    return None, None


# --------------------------------------------------------------------------- #
# StructuredOutputSpec construction (response_format + guided_* + strict tools)
# --------------------------------------------------------------------------- #


def _choice_to_regex(choices: list[str]) -> str:
    if not choices or any(not isinstance(c, str) for c in choices):
        raise SchemaValidationError(
            "guided_choice must be a non-empty list of strings", param="guided_choice"
        )
    return "(" + "|".join(re.escape(c) for c in choices) + ")"


def _wrap_optional_think(
    spec: StructuredOutputSpec, optional_tags: tuple[str, str] | None
) -> StructuredOutputSpec:
    if optional_tags is None:
        return spec
    open_tag, close_tag = optional_tags
    wrapper = {
        "mode": "optional_think",
        "think_open": open_tag,
        "think_close": close_tag,
        "inner": {
            "kind": spec.kind,
            "payload": spec.payload,
            "strict": spec.strict,
        },
    }
    return StructuredOutputSpec(
        kind="structural_tag",
        payload=json.dumps(
            wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
        strict=spec.strict,
        defer_until_tag=None,
    )


def _extract_guided_spec(
    *,
    guided_json: Any = None,
    guided_regex: Any = None,
    guided_choice: Any = None,
    guided_grammar: Any = None,
    structured_outputs: Any = None,
) -> StructuredOutputSpec | None:
    if isinstance(structured_outputs, dict) and structured_outputs:
        if structured_outputs.get("json") is not None:
            guided_json = structured_outputs["json"]
        elif structured_outputs.get("json_object"):
            return StructuredOutputSpec(kind="json_object", payload="", strict=False)
        elif structured_outputs.get("regex") is not None:
            guided_regex = structured_outputs["regex"]
        elif structured_outputs.get("choice") is not None:
            guided_choice = structured_outputs["choice"]
        elif structured_outputs.get("grammar") is not None:
            guided_grammar = structured_outputs["grammar"]
        else:
            raise SchemaValidationError(
                "structured_outputs must specify one of: json, json_object, regex, choice, grammar",
                param="structured_outputs",
            )

    specified = [
        name
        for name, val in (
            ("guided_json", guided_json),
            ("guided_regex", guided_regex),
            ("guided_choice", guided_choice),
            ("guided_grammar", guided_grammar),
        )
        if val is not None
        and val is not False
        and val != ""
        and val != []
        and val != {}
    ]
    if not specified:
        return None
    if len(specified) > 1:
        raise SchemaValidationError(
            f"Specify at most one guided decoding constraint, got: {', '.join(specified)}",
            param=specified[0],
        )

    if guided_json is not None and guided_json is not False and guided_json != "":
        if isinstance(guided_json, str):
            try:
                parsed = json.loads(guided_json)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(
                    f"guided_json string must be valid JSON: {exc}", param="guided_json"
                ) from exc
        else:
            parsed = guided_json
        validate_json_schema(parsed, strict=False, param="guided_json")
        return StructuredOutputSpec(
            kind="json_schema", payload=canonical_schema_json(parsed), strict=False
        )
    if guided_regex is not None and guided_regex is not False and guided_regex != "":
        if not isinstance(guided_regex, str):
            raise SchemaValidationError(
                "guided_regex must be a string", param="guided_regex"
            )
        return StructuredOutputSpec(kind="regex", payload=guided_regex, strict=True)
    if guided_choice is not None and guided_choice is not False and guided_choice != []:
        if not isinstance(guided_choice, list):
            raise SchemaValidationError(
                "guided_choice must be a list of strings", param="guided_choice"
            )
        return StructuredOutputSpec(
            kind="regex", payload=_choice_to_regex(guided_choice), strict=True
        )
    if (
        guided_grammar is not None
        and guided_grammar is not False
        and guided_grammar != ""
    ):
        if not isinstance(guided_grammar, str):
            raise SchemaValidationError(
                "guided_grammar must be an EBNF string", param="guided_grammar"
            )
        return StructuredOutputSpec(kind="ebnf", payload=guided_grammar, strict=True)
    return None


def _build_tool_spec(
    tools: list[dict[str, Any]] | None,
    tool_choice_mode: str | None,
    tool_call_parser: str | None,
    disable_parallel_tool_use: bool,
) -> StructuredOutputSpec | None:
    if not tools:
        return None
    has_strict = any(
        bool((t.get("function") or {}).get("strict"))
        for t in tools
        if isinstance(t, dict)
    )
    forced = tool_choice_mode == "required" or (
        isinstance(tool_choice_mode, str) and tool_choice_mode.startswith("named:")
    )
    if not forced and not has_strict:
        return None

    # Validate strict tool parameter schemas up front.
    selected_name = (
        tool_choice_mode.removeprefix("named:")
        if isinstance(tool_choice_mode, str) and tool_choice_mode.startswith("named:")
        else None
    )
    entries: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        if selected_name is not None and name != selected_name:
            continue
        strict_flag = bool(fn.get("strict"))
        params = fn.get("parameters") or {"type": "object"}
        if strict_flag:
            validate_json_schema(params, strict=True, param="tools")
        entries.append(
            {
                "name": name,
                "parameters": params,
                "strict": strict_flag,
            }
        )
    if not entries:
        return None

    payload = {
        "mode": "tool_call",
        "parser": (tool_call_parser or "qwen25").lower(),
        "tool_choice_mode": tool_choice_mode or "auto",
        "at_least_one": forced,
        "stop_after_first": bool(disable_parallel_tool_use),
        "tools": entries,
    }
    return StructuredOutputSpec(
        kind="structural_tag",
        payload=json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
        strict=has_strict,
    )


def build_structured_output_spec(
    *,
    response_format: dict[str, Any] | None = None,
    guided_json: Any = None,
    guided_regex: Any = None,
    guided_choice: Any = None,
    guided_grammar: Any = None,
    structured_outputs: Any = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice_mode: str | None = None,
    tool_call_parser: str | None = None,
    disable_parallel_tool_use: bool = False,
    reasoning_parser: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> StructuredOutputSpec | None:
    """Resolve response_format / guided_* / forced or strict tool_choice into one
    protocol-neutral ``StructuredOutputSpec`` (or ``None`` if unconstrained)."""
    defer_tag, opt_think = resolve_reasoning_grammar_mode(
        reasoning_parser, chat_template_kwargs, tools
    )

    spec = _extract_guided_spec(
        guided_json=guided_json,
        guided_regex=guided_regex,
        guided_choice=guided_choice,
        guided_grammar=guided_grammar,
        structured_outputs=structured_outputs,
    )

    if spec is None and isinstance(response_format, dict):
        rtype = response_format.get("type")
        if rtype == "json_object":
            spec = StructuredOutputSpec(kind="json_object", payload="", strict=False)
        elif rtype == "json_schema":
            js = response_format.get("json_schema")
            container = js if isinstance(js, dict) else response_format
            schema = container.get("schema")
            strict = bool(container.get("strict", False))
            validate_json_schema(schema, strict=strict, param="response_format")
            spec = StructuredOutputSpec(
                kind="json_schema",
                payload=canonical_schema_json(schema),
                strict=strict,
            )

    if spec is None:
        spec = _build_tool_spec(
            tools, tool_choice_mode, tool_call_parser, disable_parallel_tool_use
        )

    if spec is None:
        return None

    if opt_think is not None:
        return _wrap_optional_think(spec, opt_think)
    if defer_tag is not None:
        spec.defer_until_tag = defer_tag
    return spec
