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


def test_crf_small_prefill_ensure_unique_no_mid_layer_self_eviction():
    import torch
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.offload_cache import RETIRED_SLOT_USAGE, OffloadMoeCache
    from freetoken.moe.offload_kernels import _STEP_LOCK_USAGE

    # F6.1: RETIRED_SLOT_USAGE << 20 stays strictly positive in signed int64 (up to 1M slots).
    assert RETIRED_SLOT_USAGE == 1 << 42
    assert _STEP_LOCK_USAGE == (1 << 42) - 1
    assert 0 < (RETIRED_SLOT_USAGE << 20) < (1 << 63) - 1

    # F1: CRF + _ensure_unique across >32 unique experts must never self-evict earlier chunks'
    # slots when other slots hold forward CRF leases (e.g. seeded hot experts at lease 500).
    num_experts = 96
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=num_experts,
        cache_size=num_experts,
        device=torch.device("cpu"),
        cache_policy="crf",
    )
    gate_up = (
        torch.arange(1, num_experts + 1, dtype=torch.bfloat16)
        .view(num_experts, 1, 1)
        .expand(num_experts, 8, 16)
        .contiguous()
    )
    down = gate_up.clone()
    cache.set_bank_sources({"gate_up": [gate_up], "down": [down]})

    # Seed slots 48..95 with hot experts 48..95 carrying forward lease 500; slots 0..47 are cold.
    hot_ids = torch.arange(48, num_experts, dtype=torch.int32)
    cache.id_of_slot[48:num_experts] = hot_ids
    cache.slot_for_id[0, 48:num_experts] = hot_ids
    cache.usage[48:num_experts] = 500
    cache.expert_crf[0, 48:num_experts] = 16.0
    for _, bank_cache in cache.banks:
        bank_cache[48:num_experts].copy_(gate_up[48:num_experts])

    dummy_moe = object.__new__(OffloadMoELayer)
    dummy_moe.layer_id = 0
    dummy_moe.ep_size = 1
    dummy_moe.quant_method = SimpleNamespace(
        kernel=SimpleNamespace(supports_inactive_slots=True)
    )

    # Route 64 unique experts (0..63): 48 misses (0..47) + 16 hot hits (48..63).
    topk_ids = torch.arange(64, dtype=torch.int32).view(16, 4)
    mapped_slots = dummy_moe._ensure_unique(cache, topk_ids)
    flat_slots = mapped_slots.reshape(-1)
    assert len(set(flat_slots.tolist())) == 64

    # Every routed expert's mapped GPU bank slot MUST hold its own weight (zero mid-layer overwrite).
    gu_cache = cache.banks[0][1]
    for exp_id in range(64):
        slot = int(flat_slots[exp_id].item())
        assert float(gu_cache[slot, 0, 0].item()) == float(exp_id + 1)
    # Hot hits 48..63 kept their >= 500 forward CRF lease instead of being clobbered to bare step.
    for exp_id in range(48, 64):
        slot = int(flat_slots[exp_id].item())
        assert int(cache.usage[slot].item()) >= 500


def test_qwen3_coder_omitted_function_close_tag():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f1",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "f2",
                "parameters": {
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                },
            },
        },
    ]
    # 1. Trailing prose after omitted </function> must be preserved.
    raw1 = (
        "<tool_call><function=f1><parameter=x>hello</parameter></tool_call>After tool."
    )
    p1 = FunctionCallParser(tools, "qwen3_coder").parse_non_stream(raw1)
    assert len(p1.calls) == 1
    assert p1.calls[0].name == "f1"
    assert json.loads(p1.calls[0].parameters) == {"x": "hello"}
    assert p1.normal_text.strip() == "After tool."

    # 2. Back-to-back calls with omitted </function> must NOT merge f2's parameter into f1,
    # even when streamed 2 characters at a time.
    raw2 = (
        "<tool_call><function=f1><parameter=x>one</parameter></tool_call>\n"
        "<tool_call><function=f2><parameter=y>two</parameter></tool_call>"
    )
    sp = FunctionCallParser(tools, "qwen3_coder")
    names: dict[int, str] = {}
    args: dict[int, list[str]] = {}
    for i in range(0, len(raw2), 2):
        for kind, payload in sp.parse_stream_events(raw2[i : i + 2]):
            if kind == "calls":
                for item in payload:
                    if item.name:
                        names[item.tool_index] = item.name
                    if item.parameters:
                        args.setdefault(item.tool_index, []).append(item.parameters)
    sp.finish_stream()
    assert names == {0: "f1", 1: "f2"}
    assert json.loads("".join(args[0])) == {"x": "one"}
    assert json.loads("".join(args[1])) == {"y": "two"}


def test_qwen3_reasoning_json_tool_call_with_embedded_tags():
    rp = Qwen3ReasoningParser(force_reasoning=True, tool_names=["write_file"])
    rp.finish_reason = "stop"
    call_str = (
        '<tool_call>{"name": "write_file", "arguments": '
        '{"content": "wrap in <tool_call>x</tool_call> tags"}}</tool_call>'
    )
    raw = f"Plan first.\n{call_str}"
    res = rp.detect_and_parse(raw)
    assert res.reasoning_text == "Plan first."
    assert res.normal_text == call_str

    # Streaming chunk-by-chunk must hold across the inner </tool_call> inside the JSON string,
    # then emit the complete tool call on flush() once finish_reason="stop".
    rp_s = Qwen3ReasoningParser(force_reasoning=True, tool_names=["write_file"])
    r_parts, n_parts = [], []
    for i in range(0, len(raw), 7):
        step = rp_s.parse_streaming_increment(raw[i : i + 7])
        if step.reasoning_text:
            r_parts.append(step.reasoning_text)
        if step.normal_text:
            n_parts.append(step.normal_text)
    assert n_parts == []
    rp_s.finish_reason = "stop"
    tail = rp_s.flush()
    if tail.reasoning_text:
        r_parts.append(tail.reasoning_text)
    if tail.normal_text:
        n_parts.append(tail.normal_text)
    assert "".join(r_parts) == "Plan first."
    assert "".join(n_parts) == call_str


def test_json_parser_ignores_non_dict_literals():
    tools = [
        {
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object"}},
        }
    ]
    for bad in ("<tool_call>null</tool_call>", "<tool_call>123</tool_call>"):
        res = FunctionCallParser(tools, "qwen25").parse_non_stream(bad)
        assert res.calls == []
        sp = FunctionCallParser(tools, "qwen25")
        for ch in (bad[:12], bad[12:]):
            sp.parse_streaming_increment(ch)


def test_responses_api_suppresses_unstable_args_deltas():
    from freetoken.server import responses_api as rp
    from freetoken.server.function_call_parser import ToolCallItem
    from freetoken.server.generation import (
        GenDone,
        ToolCallArgsDelta,
        ToolCallsDelta,
        ToolCallStart,
    )

    async def _aiter(items):
        for it in items:
            yield it

    req = ResponsesRequest.model_validate({"model": "m", "input": "hi"})
    events = [
        ToolCallStart(tool_index=0, name="f", args_prefix_stable=False),
        ToolCallArgsDelta(tool_index=0, fragment='{"x": 1}'),
        ToolCallArgsDelta(tool_index=0, fragment=', "y": 2}'),
        ToolCallsDelta(
            calls=[ToolCallItem(tool_index=0, name="f", parameters='{"x":1,"y":2}')]
        ),
        GenDone(
            finish_reason="tool_calls",
            matched_stop=None,
            prompt_tokens=5,
            completion_tokens=4,
            cached_tokens=0,
            reasoning_tokens=0,
            timings=None,
        ),
    ]

    async def _run():
        deltas = []
        done_args = []
        async for frame in rp.responses_stream_generator(_aiter(events), req, "r1", 0):
            etype, data = None, None
            for line in frame.split("\n"):
                if line.startswith("event:"):
                    etype = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[len("data:") :].strip())
            if etype == "response.function_call_arguments.delta":
                deltas.append(data["delta"])
            elif etype == "response.function_call_arguments.done":
                done_args.append(data["arguments"])
        return deltas, done_args

    deltas, done_args = asyncio.run(_run())
    assert deltas == ['{"x":1,"y":2}']
    assert done_args == ['{"x":1,"y":2}']


def test_anthropic_thinking_auto_and_fail_pending_waiters():
    from freetoken.server.api_server import FrontendManager

    req_auto = AnthropicMessagesRequest(
        model="qwen",
        max_tokens=64,
        thinking={"type": "auto"},
        messages=[{"role": "user", "content": "hi"}],
    )
    _, _, _, ctk_auto = convert_anthropic_prompt(req_auto)
    assert ctk_auto == {}

    # Active HTTP request blocked in wait_for_ack wakes immediately when fail_pending_rebuilds fires.
    async def _check_wakeup():
        mgr = object.__new__(FrontendManager)
        mgr._loop = asyncio.get_running_loop()
        mgr.rebuild_futures = {}
        mgr.flush_futures = {}
        mgr.ack_map = {7: []}
        mgr.event_map = {7: asyncio.Event()}
        mgr.active_waiters = set()

        async def _consume():
            got = []
            async for ack in mgr.wait_for_ack(7):
                got.append(ack)
            return got

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0)
        mgr.fail_pending_rebuilds("worker rank 1 exited (code 1)")
        replies = await asyncio.wait_for(task, timeout=1.0)
        assert len(replies) == 1
        assert replies[0].finished is True
        assert "worker rank 1 exited" in (replies[0].error or "")

    asyncio.run(_check_wakeup())


def test_moe_ep_rejections_marlin_compat_and_kv_only_host_restore(monkeypatch):
    import torch
    from freetoken.layers.int8_g32 import _call_gptq_marlin_linear
    from freetoken.layers.quantization import GGUFConfig
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.mxfp4 import TritonGptossMxfp4MoEKernel
    from freetoken.layers.quantization.moe.nvfp4 import (
        B12xNvfp4MoEKernel,
        MarlinNvfp4MoEKernel,
    )
    from freetoken.models.qwen4_exp.weight import dense_emitter
    from freetoken.scheduler.scheduler import Scheduler

    cfg_ep = MoEConfig(
        num_experts=64, hidden=256, intermediate=128, top_k=4, ep_size=2, has_bias=True
    )
    assert "--moe-ep" in (
        MarlinNvfp4MoEKernel().unusable_reason(cfg_ep) or "CUDA-only --moe-ep"
    )
    assert "--moe-ep" in (
        B12xNvfp4MoEKernel().unusable_reason(cfg_ep) or "CUDA-only --moe-ep"
    )
    assert "--moe-ep" in (TritonGptossMxfp4MoEKernel().unusable_reason(cfg_ep) or "")

    # vLLM 0.14 (w_type, g_idx, is_k_full) vs vLLM 0.16+ (wtype, reduced signature) compatibility
    def fake_vllm_014(
        *,
        input,
        weight,
        weight_scale,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        w_type,
        output_size_per_partition,
        input_size_per_partition,
        is_k_full,
    ):
        return (w_type, is_k_full, output_size_per_partition)

    def fake_vllm_016(
        *,
        input,
        weight,
        weight_scale,
        weight_zp,
        g_idx_sort_indices,
        workspace,
        wtype,
        output_size_per_partition,
        input_size_per_partition,
    ):
        return (wtype, output_size_per_partition)

    z = torch.zeros(1)
    assert _call_gptq_marlin_linear(
        fake_vllm_014,
        input_t=z,
        weight=z,
        weight_scale=z,
        empty_i32=z,
        workspace=z,
        w_type="u8b128",
        output_size_per_partition=64,
        input_size_per_partition=64,
    ) == ("u8b128", True, 64)
    assert _call_gptq_marlin_linear(
        fake_vllm_016,
        input_t=z,
        weight=z,
        weight_scale=z,
        empty_i32=z,
        workspace=z,
        w_type="u8b128",
        output_size_per_partition=64,
        input_size_per_partition=64,
    ) == ("u8b128", 64)

    # GGUF sidecar/pack without FREETOKEN_GGUF_FLOAT_DENSE=1 must not emit hc8 weight_scale
    monkeypatch.setenv("FREETOKEN_HC_INT8", "1")
    monkeypatch.delenv("FREETOKEN_GGUF_FLOAT_DENSE", raising=False)
    gguf_cfg = SimpleNamespace(
        attn_quant="none",
        tie_word_embeddings=False,
        quant=GGUFConfig({}, lambda n, _: (n,)),
    )
    emit, _, _ = dense_emitter(gguf_cfg, 0, 1, include_lmhead=False)
    emitted = dict(
        emit(
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            torch.ones(64, 64, dtype=torch.bfloat16),
        )
    )
    assert list(emitted.keys()) == [
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
    ]

    # Scheduler._restore_linear_states must fence & restore host KV pages even when linear_state_pool is None
    events: list[str] = []

    class DummyKV:
        num_host_pages = 16

        def fence_host_copies(self):
            events.append("fenced")

        def pages_from_host(self, hp, dp):
            events.append(f"restored:{hp}->{dp}")

    sched = object.__new__(Scheduler)
    sched.engine = SimpleNamespace(linear_state_pool=None, kv_cache=DummyKV())
    req = SimpleNamespace(
        kv_restore=SimpleNamespace(host_pages=[1, 2], dev_pages=[5, 6])
    )
    sched._restore_linear_states(SimpleNamespace(is_prefill=True, reqs=[req]))
    assert events == ["fenced", "restored:[1, 2]->[5, 6]"]
    assert req.kv_restore is None
