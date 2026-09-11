"""The cases #435 fixes, as a table: a $ref'd type must be honoured, not defaulted to string."""

from __future__ import annotations

import json

import pytest

from freetoken.server.function_call_parser import FunctionCallParser


def _tool(props: dict, defs: dict | None = None) -> list[dict]:
    params: dict = {"type": "object", "properties": props}
    if defs is not None:
        params["$defs"] = defs
    return [{"type": "function", "function": {"name": "search", "parameters": params}}]


def _call(raw: str) -> str:
    return f"<tool_call><function=search><parameter=limit>{raw}</parameter></function></tool_call>"


@pytest.mark.parametrize(
    ("props", "defs", "raw", "expected"),
    [
        # the baseline that already worked: a direct type
        ({"limit": {"type": "integer"}}, None, "5", 5),
        # a single $ref
        ({"limit": {"$ref": "#/$defs/Limit"}}, {"Limit": {"type": "integer"}}, "5", 5),
        # a $ref chain
        (
            {"limit": {"$ref": "#/$defs/A"}},
            {"A": {"$ref": "#/$defs/B"}, "B": {"type": "integer"}},
            "5",
            5,
        ),
        # oneOf / anyOf carrying the type
        ({"limit": {"oneOf": [{"type": "integer"}]}}, None, "5", 5),
        ({"limit": {"anyOf": [{"type": "integer"}]}}, None, "5", 5),
        # a nullable union: the non-null member decides
        ({"limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}, None, "5", 5),
        # a $ref to an array and to an object
        (
            {"limit": {"$ref": "#/$defs/L"}},
            {"L": {"type": "array", "items": {"type": "integer"}}},
            "[1, 2]",
            [1, 2],
        ),
        (
            {"limit": {"$ref": "#/$defs/L"}},
            {"L": {"type": "object"}},
            '{"a": 1}',
            {"a": 1},
        ),
    ],
)
def test_qwen3_coder_resolves_indirect_schemas(props, defs, raw, expected):
    parser = FunctionCallParser(_tool(props, defs), tool_call_parser="qwen3_coder")
    result = parser.parse_non_stream(_call(raw))
    assert len(result.calls) == 1, result
    assert json.loads(result.calls[0].parameters) == {"limit": expected}


def test_unresolvable_ref_falls_back_to_string_and_does_not_raise():
    # A dangling $ref must not take the server down; string is the safe default.
    parser = FunctionCallParser(_tool({"limit": {"$ref": "#/$defs/Missing"}}), tool_call_parser="qwen3_coder")
    result = parser.parse_non_stream(_call("5"))
    assert json.loads(result.calls[0].parameters) == {"limit": "5"}


def test_self_referential_ref_terminates():
    # A cycle must not hang the parser.
    parser = FunctionCallParser(
        _tool({"limit": {"$ref": "#/$defs/A"}}, {"A": {"$ref": "#/$defs/A"}}),
        tool_call_parser="qwen3_coder",
    )
    result = parser.parse_non_stream(_call("5"))
    assert len(result.calls) == 1
