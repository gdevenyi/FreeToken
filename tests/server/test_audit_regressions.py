from __future__ import annotations

import asyncio
import base64
import json
import os
from types import SimpleNamespace

import pytest
from freetoken.message.tokenizer import DetokenizeMsg
from freetoken.mm.media import _MAX_IMAGE_BYTES, _check_redirect_target, _fetch_one
from freetoken.scheduler.scheduler import logprob_candidates
from freetoken.server.anthropic_api import (
    _maintenance_detail,
    convert_anthropic_prompt,
    strip_anthropic_billing_cch,
)
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.api_models import CompletionRequest
from freetoken.server.bind import split_hosts
from freetoken.server.function_call_parser import FunctionCallParser
from freetoken.server.generation import _reasoning_finish
from freetoken.server.model_meta import effort_toggle_kwargs, thinking_override_kwargs
from freetoken.server.openai_api import _check_restricted_logprobs, _resolve_sampling
from freetoken.server.reasoning_parser import Qwen3ReasoningParser
from freetoken.server.responses_api import ResponsesRequest, convert_responses_prompt
from freetoken.tokenizer.detokenize import DetokenizeManager
from freetoken.tokenizer.tokenize import _map_developer_role


def test_stutter_tool_call_prose_not_eaten():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    text = "In XML format, avoid doubling <tool_call><tool_call> before prose or you lose everything after it."
    parser = FunctionCallParser(tools, "qwen3_coder")
    parsed = parser.parse_non_stream(text)
    assert parsed.calls == []
    assert parsed.normal_text == text

    # Streaming chunk-by-chunk must also preserve every character.
    stream_parser = FunctionCallParser(tools, "qwen3_coder")
    out_text = []
    out_calls = []
    for i in range(0, len(text), 5):
        for kind, payload in stream_parser.parse_stream_events(text[i : i + 5]):
            if kind == "text":
                out_text.append(payload)
            else:
                out_calls.extend(payload)
    rem = stream_parser.finish_stream()
    if rem:
        out_text.append(rem)
    assert out_calls == []
    assert "".join(out_text) == text


def test_stutter_tool_call_real_invoke_still_parses():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    raw = "Let me check.\n<tool_call><tool_call><function=get_weather><parameter=city>Montreal</parameter></function></tool_call>"
    parser = FunctionCallParser(tools, "qwen3_coder")
    parsed = parser.parse_non_stream(raw)
    assert len(parsed.calls) == 1
    assert parsed.calls[0].name == "get_weather"
    assert json.loads(parsed.calls[0].parameters) == {"city": "Montreal"}


def test_qwen3_reasoning_unclosed_think_json_and_nested_tags():
    # 1. Qwen2.5/Qwen3 JSON <tool_call>{...}</tool_call> without </think>
    rp_json = Qwen3ReasoningParser(force_reasoning=True, tool_names=["read_file"])
    rp_json.finish_reason = "stop"
    raw_json = 'Need to inspect config.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
    res_json = rp_json.detect_and_parse(raw_json)
    assert res_json.reasoning_text == "Need to inspect config."
    assert '"name": "read_file"' in res_json.normal_text

    # 2. Unclosed quoted XML parameter must NOT swallow a later real call
    rp_quote = Qwen3ReasoningParser(force_reasoning=True, tool_names=["edit_file"])
    rp_quote.finish_reason = "stop"
    raw_quote = (
        "Never run:\n<tool_call><function=edit_file><parameter=cmd>rm -rf /\n"
        "Instead:\n<tool_call><function=edit_file><parameter=cmd>ls</parameter></function></tool_call>"
    )
    res_quote = rp_quote.detect_and_parse(raw_quote)
    assert "rm -rf /" in res_quote.reasoning_text
    assert res_quote.normal_text == (
        "<tool_call><function=edit_file><parameter=cmd>ls</parameter></function></tool_call>"
    )


def test_reasoning_finish_checks_actual_stop_token_hit():
    spec = SimpleNamespace(sampling_params=SimpleNamespace(stop_token_ids=[151645]))
    assert _reasoning_finish(spec, "stop", None, hit_stop_token=False) == "stop"
    assert _reasoning_finish(spec, "stop", None, hit_stop_token=True) is None
    assert _reasoning_finish(spec, "stop", None, hit_stop_token=None) is None
    assert _reasoning_finish(spec, "stop", "STOP", hit_stop_token=False) is None


def test_anthropic_cch_stripping_and_adaptive_thinking():
    header = "x-anthropic-billing-header: cc_version=2.1; cch=deadbeef"
    assert (
        strip_anthropic_billing_cch(header)
        == "x-anthropic-billing-header: cc_version=2.1;"
    )
    # Non-header blocks containing cch= must NOT be mutated.
    prose = "Check query param cch=1234abcd; in x-anthropic-billing-header: docs"
    assert strip_anthropic_billing_cch(prose) == prose

    req = AnthropicMessagesRequest(
        model="qwen",
        max_tokens=64,
        system=[
            {
                "type": "text",
                "text": "x-anthropic-billing-header: v=1; cch=abcd1234;\n",
            },
            {"type": "text", "text": "Keep cch=9999ffff; literal here."},
        ],
        thinking={"type": "Adaptive"},
        messages=[{"role": "user", "content": "hi"}],
    )
    msgs, _, _, ctk = convert_anthropic_prompt(req)
    assert msgs[0]["role"] == "system"
    assert "cch=abcd1234" not in msgs[0]["content"]
    assert msgs[0]["content"] == (
        "x-anthropic-billing-header: v=1;\nKeep cch=9999ffff; literal here."
    )
    assert ctk == {"enable_thinking": True, "thinking_mode": "enabled"}

    # reasoning_strength is stripped by operator --thinking off override
    overridden = thinking_override_kwargs(
        "off", {"reasoning_strength": "high", "extra": 1}
    )
    assert "reasoning_strength" not in overridden
    assert overridden["enable_thinking"] is False
    assert overridden["extra"] == 1

    # effort_toggle_kwargs honors case-insensitive disabled / adaptive
    assert (
        effort_toggle_kwargs("high", {}, thinking_type="OFF")["enable_thinking"]
        is False
    )
    assert (
        effort_toggle_kwargs("medium", {}, thinking_type="adaptive")["enable_thinking"]
        is True
    )
    assert _maintenance_detail("stopping") == "server is shutting down"


def test_restricted_logprobs_defaults_and_candidates():
    req = CompletionRequest(
        model="m",
        prompt="hello",
        max_tokens=1,
        logprobs=5,
        allowed_token_ids=[10, 20],
    )
    # When model generation_config has temperature > 0 and client omits temperature=0,
    # _resolve_sampling + _check_restricted_logprobs refuse rather than silently dropping it.
    with pytest.raises(ValueError, match="temperature=0"):
        _resolve_sampling(req, {"temperature": 0.7})

    sp = _resolve_sampling(req, {})
    assert sp.temperature == 0.0
    sp.min_p = 0.1
    with pytest.raises(ValueError, match="min_p"):
        _check_restricted_logprobs(sp)

    # Restricted candidates keep allowed -inf entries while dropping non-allowed tail entries.
    ids, lps = logprob_candidates(
        [10, 20, 99, -1],
        [-0.1, -float("inf"), -float("inf"), -float("inf")],
        allowed_ids=frozenset({10, 20}),
    )
    assert ids == [10, 20]
    assert lps[0] == -0.1 and lps[1] == -float("inf")


def test_detokenize_trim_last_token_and_multi_think_ids():
    class DummyTok:
        eos_token_id = 2

        def batch_decode(self, seqs, skip_special_tokens=False):
            vocab = {1: "a", 2: "<eos>", 3: "<stop>", 4: "<channel|>", 5: "b"}
            return ["".join(vocab.get(t, "") for t in seq) for seq in seqs]

    mgr = DetokenizeManager(
        DummyTok(), eos_token_ids=frozenset({2}), think_end_id=frozenset({4, 99})
    )
    # ignore_eos=True -> trim_last_token=False keeps EOS token 2 in output
    texts, r_toks, _ = mgr.detokenize_with_meta(
        [
            DetokenizeMsg(
                uid=1, next_tokens=[1, 4, 2], finished=True, trim_last_token=False
            )
        ]
    )
    assert texts == ["a<channel|><eos>"]
    assert r_toks == [2]

    # client stop_token_id=3 with include_stop_str_in_output=False -> trim_last_token=True strips token 3
    texts2, _, _ = mgr.detokenize_with_meta(
        [
            DetokenizeMsg(
                uid=2, next_tokens=[1, 5, 3], finished=True, trim_last_token=True
            )
        ]
    )
    assert texts2 == ["ab"]


def test_developer_role_and_parallel_tool_images():
    msgs = [
        {"role": "developer", "content": "dev1"},
        {"role": "user", "content": "u1"},
        {"role": "system", "content": "inline_reminder"},
    ]
    # gpt-oss style template mentioning developer keeps developer untouched unless DSV4 encoder is active
    assert (
        _map_developer_role(
            msgs, "{% if m.role == 'developer' %}", has_dsv4_encoder=False
        )
        == msgs
    )
    mapped_dsv4 = _map_developer_role(
        msgs, "{% if m.role == 'developer' %}", has_dsv4_encoder=True
    )
    assert mapped_dsv4[0] == {"role": "system", "content": "dev1\n\ninline_reminder"}

    # preserve_inline_system keeps mid-turn system reminder in place
    preserved = _map_developer_role(msgs, "qwen template", preserve_inline_system=True)
    assert [m["role"] for m in preserved] == ["system", "user", "system"]

    # Parallel function_call_output items with images defer the synthetic user image turn until after all tool turns
    req = ResponsesRequest(
        model="m",
        input=[
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "f", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "input_text", "text": "r1"},
                    {"type": "input_image", "image_url": "https://a.com/1.png"},
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "c2",
                "output": [
                    {"type": "input_text", "text": "r2"},
                    {"type": "input_image", "image_url": "https://a.com/2.png"},
                ],
            },
        ],
    )
    out_msgs, _, _, _ = convert_responses_prompt(req)
    assert [m["role"] for m in out_msgs] == ["assistant", "tool", "tool", "user"]


def test_media_ssrf_and_bounds():
    cfg = SimpleNamespace(allowed_media_domains="", allowed_local_media_path="")
    for bad_url in (
        "http://127.0.0.1/a.png",
        "http://[::1]/a.png",
        "http://[::ffff:127.0.0.1]/a.png",
        "http://[fe80::1%25eth0]/a.png",
        "http://169.254.169.254/latest/meta-data/",
    ):
        with pytest.raises(ValueError, match="private or loopback"):
            _check_redirect_target(bad_url, cfg)

    # data: URL without ;base64 is rejected
    with pytest.raises(ValueError, match="base64"):
        asyncio.run(
            _fetch_one({"kind": "url", "data": "data:image/png,notbase64"}, cfg)
        )

    # Valid small data: URL decodes cleanly
    raw = b"\x89PNG\r\n\x1a\n"
    b64 = base64.b64encode(raw).decode("ascii")
    got = asyncio.run(
        _fetch_one({"kind": "url", "data": f"data:image/png;base64,{b64}"}, cfg)
    )
    assert got == raw

    # Oversized base64 string rejected before full decode
    huge = "A" * (((_MAX_IMAGE_BYTES + 2) // 3) * 4 + 70_000)
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(_fetch_one({"kind": "b64", "data": huge}, cfg))


def test_split_hosts_unwraps_ipv6_brackets():
    assert split_hosts(" [::1] , 127.0.0.1 , ::1 ") == ["::1", "127.0.0.1"]


def test_cudagraph_pool_recapture_preserves_mempool_use_count():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from freetoken.kernel.l2_persist import (
        clear_stream_persisting_l2_window,
        find_graph_pool_window,
        set_stream_persisting_l2_window,
    )

    dev_idx = torch.cuda.current_device()
    stream = torch.cuda.Stream()
    x = torch.ones(256, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=None, stream=stream):
        out.copy_(x * 2)
    pool = graph.pool()
    win1 = find_graph_pool_window(pool, dev_idx)
    assert win1 is not None
    set_stream_persisting_l2_window(
        stream, win1[0], win1[1], max_window_mb=16, device_index=dev_idx
    )
    recaptured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(recaptured, pool=pool, stream=stream):
        out.copy_(x * 2)
    graph.reset()
    graph = recaptured
    win2 = find_graph_pool_window(pool, dev_idx)
    assert win2 == win1
    clear_stream_persisting_l2_window(
        stream, reset_device_limit=False, device_index=dev_idx
    )
    graph.replay()
    stream.synchronize()
    clear_stream_persisting_l2_window(
        stream, reset_device_limit=True, device_index=dev_idx
    )
    assert float(out[0, 0].item()) == 2.0


def test_pread_exact_handles_short_reads(monkeypatch, tmp_path):
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    )
    import quantize_ple_lut4 as qpl

    p = tmp_path / "blob.bin"
    payload = b"0123456789abcdef"
    p.write_bytes(payload)
    fd = os.open(str(p), os.O_RDONLY)
    try:
        real_pread = os.pread
        monkeypatch.setattr(
            os, "pread", lambda f, n, off: real_pread(f, min(n, 3), off)
        )
        assert bytes(qpl._pread_exact(fd, len(payload), 0)) == payload
    finally:
        os.close(fd)
