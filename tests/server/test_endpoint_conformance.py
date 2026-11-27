"""Comprehensive conformance tests across Sampler, Speculative Logprobs, Hybrid Cache,
Tokenizer, OpenAI Chat/Completions/Models/Responses, and Anthropic Messages APIs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler
from freetoken.engine.spec_protocol import SpecForwardOutput
from freetoken.message import DetokenizeMsg, TokenizeMsg, UserReply
from freetoken.scheduler.scheduler import _attach_logprobs, _attach_verify_logprobs
from freetoken.server import anthropic_api as A
from freetoken.server import openai_api as OA
from freetoken.server import responses_api as RP
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.api_models import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
)
from freetoken.server.api_server import install_api_key
from freetoken.server.control_api import register_control_routes
from freetoken.server.function_call_parser import ToolCallItem
from freetoken.server.generation import GenResult
from freetoken.tokenizer.detokenize import DetokenizeManager
from freetoken.tokenizer.tokenize import TokenizeManager


def _run(coro):
    return asyncio.run(coro)


class _DummyReq:
    def __init__(self, sp: SamplingParams, prompt_len: int = 4, gen_len: int = 0):
        self.sampling_params = sp
        self.output_len = sp.max_tokens
        self.max_device_len = prompt_len + sp.max_tokens
        self.device_len = prompt_len + gen_len
        self.input_ids = torch.zeros(self.device_len, dtype=torch.int32)
        self.output_token_counts = None
        self.prompt_token_mask = None
        self.can_decode = True


class _DummyBatch:
    def __init__(self, reqs):
        self.reqs = reqs


# --------------------------------------------------------------------------- #
# 1. Engine Sampler: Seed Reproducibility, Restricted Tie-Break, & Logprob Targets
# --------------------------------------------------------------------------- #
def test_sampler_seed_is_reproducible_and_batch_invariant():
    sampler = Sampler(device=torch.device("cpu"), vocab_size=64)
    torch.manual_seed(123)
    row_logits = torch.randn(1, 64)
    other_logits = torch.randn(3, 64)

    sp_same_a = SamplingParams(temperature=0.8, top_k=16, top_p=0.9, seed=42)
    sp_same_b = SamplingParams(temperature=0.8, top_k=16, top_p=0.9, seed=42)
    sp_other = SamplingParams(temperature=0.9, seed=None)

    solo_seq = []
    mixed_seq = []
    for step in range(8):
        solo_batch = _DummyBatch([_DummyReq(sp_same_a, gen_len=step)])
        solo_tok = sampler.sample(row_logits.clone(), sampler.prepare(solo_batch))
        solo_seq.append(int(solo_tok[0].item()))

        mixed_batch = _DummyBatch(
            [
                _DummyReq(sp_other, gen_len=step),
                _DummyReq(sp_same_b, gen_len=step),
                _DummyReq(sp_other, gen_len=step + 3),
                _DummyReq(sp_other, gen_len=0),
            ]
        )
        mixed_logits = torch.cat(
            [other_logits[:1], row_logits, other_logits[1:]], dim=0
        )
        mixed_toks = sampler.sample(mixed_logits, sampler.prepare(mixed_batch))
        mixed_seq.append(int(mixed_toks[1].item()))

    assert solo_seq == mixed_seq

    # Speculative sample_verify at width=4 matches step-by-step sample() at positions 0..3!
    verify_batch = _DummyBatch([_DummyReq(sp_same_a, gen_len=0)])
    drafts = torch.zeros((1, 3), dtype=torch.int64)
    verify_logits = row_logits.repeat(4, 1)
    verify_toks = sampler.sample_verify(
        verify_logits, sampler.prepare(verify_batch), drafts
    )
    assert verify_toks.tolist() == solo_seq[:4]

    # DFlash reject_with_draft_probs with different unseeded drafts commits identical prefix tokens!
    from freetoken.spec.rejection import reject_with_draft_probs

    support = torch.tensor([[[0, 1], [0, 1], [0, 1]]], dtype=torch.int64)
    q_a = torch.tensor([[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]], dtype=torch.float32)
    q_b = torch.tensor([[[0.1, 0.9], [0.5, 0.5], [0.2, 0.8]]], dtype=torch.float32)
    drafts_exact = torch.tensor([solo_seq[:3]], dtype=torch.int64)
    drafts_wrong = torch.tensor(
        [[solo_seq[0], (solo_seq[1] + 1) % 64, 0]], dtype=torch.int64
    )
    out_a, em_a = reject_with_draft_probs(
        verify_logits,
        drafts_exact,
        support,
        q_a,
        sampler.prepare(verify_batch),
        sampler,
    )
    out_b, em_b = reject_with_draft_probs(
        verify_logits,
        drafts_wrong,
        support,
        q_b,
        sampler.prepare(verify_batch),
        sampler,
    )
    assert int(em_a[0].item()) == 3
    assert out_a[0, : int(em_a[0].item()) + 1].tolist() == solo_seq[:4]
    assert int(em_b[0].item()) == 1
    assert out_b[0, : int(em_b[0].item()) + 1].tolist() == solo_seq[:2]


def test_sampler_compute_logprobs_allowed_neg_inf_never_leaks_token_zero():
    sampler = Sampler(device=torch.device("cpu"), vocab_size=100)
    logits = torch.zeros(1, 100)
    logits[0, 50] = 5.0
    logits[0, 60] = float("-inf")  # allowed candidate with -inf logit
    logits[0, 0] = 99.0  # disallowed token 0 with huge logit

    sp = SamplingParams(
        temperature=0.0, logprobs=True, top_logprobs=5, allowed_token_ids=[50, 60]
    )
    args = sampler.prepare(_DummyBatch([_DummyReq(sp)]))
    sampled = torch.tensor([50], dtype=torch.int64)
    res = sampler.compute_logprobs(logits, sampled, args)
    assert res is not None
    chosen_lp, top_ids, top_lps = res
    assert abs(float(chosen_lp[0].item())) < 1e-6
    msg = DetokenizeMsg(uid=1, next_tokens=[50], finished=True)
    out = SimpleNamespace(
        chosen_logprobs_cpu=chosen_lp,
        top_ids_cpu=top_ids,
        top_logprobs_cpu=top_lps,
    )
    _attach_logprobs(msg, out, 0, sp.top_logprobs, sp.allowed_token_ids, None)
    assert msg.top_ids == [50, 60]
    assert 0 not in msg.top_ids


def test_sampler_logprob_token_ids_and_verify_logprobs():
    sampler = Sampler(device=torch.device("cpu"), vocab_size=50)
    logits = torch.arange(50, dtype=torch.float32).unsqueeze(0)  # argmax is 49
    sp = SamplingParams(
        temperature=0.0, logprobs=True, top_logprobs=1, logprob_token_ids=[3, 7]
    )
    args = sampler.prepare(_DummyBatch([_DummyReq(sp)]))
    sampled = torch.tensor([49], dtype=torch.int64)
    res = sampler.compute_logprobs(logits, sampled, args)
    assert res is not None
    expected_full_lps = torch.log_softmax(logits, dim=-1)[0]
    msg = DetokenizeMsg(uid=1, next_tokens=[49], finished=True)
    out = SimpleNamespace(
        chosen_logprobs_cpu=res[0], top_ids_cpu=res[1], top_logprobs_cpu=res[2]
    )
    _attach_logprobs(msg, out, 0, sp.top_logprobs, None, sp.logprob_token_ids)
    returned = dict(zip(msg.top_ids, msg.top_logprobs, strict=True))
    assert set(returned) == {49, 7, 3}
    for tid in (49, 7, 3):
        assert abs(returned[tid] - float(expected_full_lps[tid].item())) < 1e-5

    # Multi-token speculative verify logprobs across width=3
    v_logits = torch.stack(
        [logits[0], logits[0] * 0.5, logits[0] * 0.2], dim=0
    )  # [3, 50]
    v_sampled = torch.tensor([[49, 48, 47]], dtype=torch.int64)
    v_res = sampler.compute_verify_logprobs(v_logits, v_sampled, args, width=3)
    assert v_res is not None
    spec_out = SpecForwardOutput(
        v_sampled,
        v_sampled,
        None,  # type: ignore[arg-type]
        torch.tensor([1]),
        v_res[0],
        v_res[1],
        v_res[2],
    )
    v_msg = DetokenizeMsg(uid=2, next_tokens=[49, 48], finished=True)
    _attach_verify_logprobs(
        v_msg, spec_out, 0, 2, sp.top_logprobs, None, sp.logprob_token_ids
    )
    assert v_msg.step_logprobs is not None and len(v_msg.step_logprobs) == 2
    assert [s[0] for s in v_msg.step_logprobs] == [49, 48]


class _ByteStubTokenizer:
    eos_token_id = 0
    bos_token = "<bos>"

    def __len__(self) -> int:
        return 300

    def batch_decode(self, seqs, skip_special_tokens=False):
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in seqs]

    def decode(self, ids, skip_special_tokens=False):
        raw = bytes(
            t for t in ids if 1 <= t <= 255 and not (skip_special_tokens and t == 0)
        )
        return raw.decode("utf-8", errors="replace")

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        prefix = [254] if add_special_tokens else []
        arr = prefix + list(text.encode("utf-8"))
        if return_tensors == "pt":
            return torch.tensor([arr], dtype=torch.long)
        return arr

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        parts = [f"<{m['role']}>{m.get('content', '')}" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


def test_detokenize_step_meta_tracks_multi_token_spec_offsets():
    dm = DetokenizeManager(_ByteStubTokenizer(), eos_token_ids=frozenset({0}))
    crab_bytes = list("🦀!".encode())  # 4 bytes for 🦀 + 1 byte for '!'
    step_lps = [(b, -0.1, [b], [-0.1]) for b in crab_bytes]
    msg = DetokenizeMsg(
        uid=7,
        next_tokens=crab_bytes,
        finished=True,
        finish_reason="stop",
        step_logprobs=step_lps,
    )
    texts, _, first_offs, step_offs = dm.detokenize_with_step_meta([msg])
    assert texts == ["🦀!"]
    assert first_offs == [0]
    # All 4 UTF-8 byte tokens of '🦀' point to offset 0, and '!' starts at char offset 1!
    assert step_offs == [[0, 0, 0, 0, 1]]


def test_tokenize_manager_suppresses_double_bos_and_injects_directives_cleanly():
    tm = TokenizeManager(_ByteStubTokenizer())
    raw_with_bos = TokenizeMsg(
        uid=1, text="<bos>hello", sampling_params=SamplingParams()
    )
    raw_plain = TokenizeMsg(uid=2, text="hello", sampling_params=SamplingParams())
    forced_bos = TokenizeMsg(
        uid=3,
        text="<bos>hello",
        sampling_params=SamplingParams(),
        add_special_tokens=True,
    )
    u_bos, u_plain, u_forced = tm.tokenize([raw_with_bos, raw_plain, forced_bos])
    assert u_bos.input_ids[0].item() != 254  # suppressed duplicate auto-BOS
    assert u_plain.input_ids[0].item() == 254
    assert u_forced.input_ids[0].item() == 254

    caller_msgs = [{"role": "user", "content": "weather?"}]
    chat_msg = TokenizeMsg(
        uid=4,
        text=caller_msgs,
        sampling_params=SamplingParams(),
        tool_choice_mode="named:get_weather",
        response_format={"type": "json_object"},
    )
    rendered = tm.render_prompt(chat_msg)
    assert "get_weather" in rendered and "JSON object" in rendered
    assert caller_msgs == [{"role": "user", "content": "weather?"}]  # unmutated


# --------------------------------------------------------------------------- #
# 2. OpenAI Endpoints Conformance
# --------------------------------------------------------------------------- #
class _MultiState:
    def __init__(self, outputs=('```json\n{"ok": true}\n```',), tokenizer=None):
        self.config = SimpleNamespace(
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            model_path="/models/qwen3.8",
            served_model_name="qwen3.8",
            tool_call_parser="qwen3_coder",
            reasoning_parser=None,
            max_seq_len=32768,
            decode_rows_per_req=4,
            maintenance_state="serving",
            max_completion_prompts=256,
        )
        self.outputs = list(outputs)
        self._uid = 0
        self.sent: list[TokenizeMsg] = []
        if tokenizer is not None:
            mgr = TokenizeManager(tokenizer)
            self.frontend_tokenizer = lambda: mgr

    def new_user(self) -> int:
        self._uid += 1
        return self._uid

    async def send_one(self, msg: TokenizeMsg):
        self.sent.append(msg)

    async def abort_user(self, uid: int):
        pass

    async def wait_for_ack(self, uid: int):
        out = self.outputs[(uid - 1) % len(self.outputs)]
        yield UserReply(
            uid=uid, incremental_output="", finished=False, prompt_tokens_delta=6
        )
        yield UserReply(
            uid=uid,
            incremental_output=out,
            finished=True,
            completion_tokens_delta=3,
            finish_reason="stop",
            logprobs={
                "token_id": 65,
                "token": "A",
                "bytes": [65],
                "logprob": -0.25,
                "top": [
                    {"token_id": 65, "token": "A", "bytes": [65], "logprob": -0.25}
                ],
            },
            text_offset=0,
        )


def test_openai_models_detail_health_and_auth_envelopes():
    state = _MultiState(tokenizer=_ByteStubTokenizer())
    app = FastAPI(version="0.1.0")
    OA.register_openai_routes(app, lambda: state, dict)
    A.register_anthropic_routes(app, lambda: state, dict)
    RP.register_responses_routes(app, lambda: state, dict)
    register_control_routes(app, lambda: state, dict)
    install_api_key(app, "secret-key")
    client = TestClient(app)

    # /health, /v1/health, and /version are open; /v1/health reflects maintenance_state
    assert client.get("/health").status_code == 200
    assert client.get("/v1/health").status_code == 200
    assert client.get("/version").status_code == 200
    state.config = state.config
    state.maintenance_state = "loading"
    assert client.get("/v1/health").status_code == 503
    state.maintenance_state = "serving"

    # 401 on /v1/messages uses AnthropicErrorResponse; /v1/models keeps {"error": "Unauthorized"}
    an_401 = client.post("/v1/messages", json={})
    assert an_401.status_code == 401
    assert an_401.json()["error"]["type"] == "authentication_error"
    oa_401 = client.get("/v1/models")
    assert oa_401.status_code == 401
    assert oa_401.json() == {"error": "Unauthorized"}

    headers = {"Authorization": "Bearer secret-key"}
    assert client.get("/v1/models/qwen3.8", headers=headers).status_code == 200
    missing = client.get("/v1/models/unknown-model", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"
    assert missing.json()["error"]["param"] == "model"

    # DELETE /v1/responses/{id} returns 404 not_found_error
    del_r = client.delete("/v1/responses/resp_123", headers=headers)
    assert del_r.status_code == 404
    assert del_r.json()["error"]["type"] == "not_found_error"

    # GET /v1/messages/unknown returns Anthropic 404 envelope
    an_404 = client.get("/v1/messages/unknown", headers=headers)
    assert an_404.status_code == 404
    assert an_404.json()["type"] == "error"
    assert an_404.json()["error"]["type"] == "not_found_error"


def test_openai_nullable_fields_json_fence_stripping_and_tool_choice():
    state = _MultiState(tokenizer=_ByteStubTokenizer())
    req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.8",
            "messages": [{"role": "user", "content": "give json"}],
            "presence_penalty": None,
            "frequency_penalty": None,
            "n": None,
            "stream": None,
            "seed": 99,
            "response_format": {"type": "json_object"},
        }
    )
    resp = _run(OA.handle_chat_completion(req, None, state, {}))
    assert resp["choices"][0]["message"]["content"] == '{"ok": true}'
    assert state.sent[0].sampling_params.seed == 99

    # Non-streaming tool_calls must NOT carry "index"
    two_calls = (
        "<tool_call>\n<function=ping>\n<parameter=x>1</parameter>\n</function>\n</tool_call>"
        "<tool_call>\n<function=ping>\n<parameter=x>2</parameter>\n</function>\n</tool_call>"
    )
    tc_state = _MultiState(outputs=(two_calls,))
    tool_def = {
        "type": "function",
        "function": {"name": "ping", "parameters": {"type": "object"}},
    }
    tc_req = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.8",
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [tool_def],
            "tool_choice": "required",
            "parallel_tool_calls": False,
        }
    )
    tc_resp = _run(OA.handle_chat_completion(tc_req, None, tc_state, {}))
    calls = tc_resp["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1  # parallel_tool_calls=False capped at 1
    assert "index" not in calls[0]
    assert tc_state.sent[0].tool_choice_mode == "required"

    # tool_choice="required" without tools -> 400
    bad_tc = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.8",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        }
    )
    assert (
        _run(OA.handle_chat_completion(bad_tc, None, tc_state, {})).status_code == 400
    )

    # Invalid json_schema without schema -> 400
    bad_rf = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3.8",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        }
    )
    assert (
        _run(OA.handle_chat_completion(bad_rf, None, tc_state, {})).status_code == 400
    )


def test_openai_completion_stream_object_and_n_choices():
    state = _MultiState(outputs=("alpha", "beta"), tokenizer=_ByteStubTokenizer())
    req = CompletionRequest(
        model="qwen3.8",
        prompt="count",
        n=2,
        seed=10,
        stream=True,
        stream_options={"include_usage": True},
    )
    resp = _run(OA.handle_completion(req, None, state, {}))

    async def _drain():
        return [c async for c in resp.body_iterator]

    raw = _run(_drain())
    events = [
        json.loads(line.removeprefix("data: "))
        for chunk in raw
        for line in chunk.decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert all(ev["object"] == "text_completion" for ev in events)
    seen_indices = {c["index"] for ev in events for c in ev["choices"] if c.get("text")}
    assert seen_indices == {0, 1}
    from freetoken.engine.sample import mix_choice_seed

    assert [m.sampling_params.seed for m in state.sent] == [10, mix_choice_seed(10, 1)]
    assert state.sent[1].sampling_params.seed != 11
    assert events[-1]["usage"]["prompt_tokens"] == 6
    assert events[-1]["usage"]["completion_tokens"] == 6


def test_detokenize_bounds_check_and_responses_logprobs():
    state = _MultiState(outputs=("hello",), tokenizer=_ByteStubTokenizer())
    bad_detok = _run(OA.handle_detokenize(DetokenizeRequest(tokens=[999]), state))
    assert bad_detok.status_code == 400

    r_req = RP.ResponsesRequest(
        model="qwen3.8",
        input="hi",
        include=["message.output_text.logprobs"],
        top_logprobs=1,
        text={
            "format": {
                "type": "json_schema",
                "name": "out",
                "schema": {"type": "object"},
            }
        },
        instructions="be concise",
    )
    r_resp = _run(RP.handle_responses(r_req, None, state, {}))
    body = json.loads(r_resp.body)
    assert body["instructions"] == "be concise"
    assert body["completed_at"] is not None
    msg_item = body["output"][0]
    assert msg_item["content"][0]["logprobs"][0]["token"] == "A"


# --------------------------------------------------------------------------- #
# 3. Anthropic Messages API Conformance (P0/P1/P2)
# --------------------------------------------------------------------------- #
def test_anthropic_deferred_tools_signatures_empty_turns_and_validation():
    # P0-1: defer_loading + tool_reference + tool_addition + tool_removal
    req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-sonnet",
            "max_tokens": 128,
            "tools": [
                {
                    "name": "tool_search",
                    "description": None,
                    "input_schema": {"type": "object"},
                },
                {
                    "name": "deferred_grep",
                    "defer_loading": True,
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
                {
                    "name": "obsolete_tool",
                    "input_schema": {"type": "object"},
                },
            ],
            "messages": [
                {"role": "user", "content": "find files"},
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "enc"}],
                },
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "tool_search",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "Loaded tool:"},
                                {
                                    "type": "tool_reference",
                                    "tool_name": "deferred_grep",
                                },
                            ],
                        },
                        {
                            "type": "document",
                            "title": "README",
                            "source": {"type": "text", "data": "doc body"},
                        },
                    ],
                },
                {
                    "role": "system",
                    "content": [
                        {"type": "tool_removal", "tool_names": ["obsolete_tool"]},
                        {
                            "type": "tool_addition",
                            "tools": [
                                {
                                    "name": "dynamic_bash",
                                    "input_schema": {"type": "object"},
                                }
                            ],
                        },
                    ],
                },
            ],
            "thinking": {"type": "between_tools"},
        }
    )
    spec = A.convert_anthropic_to_genspec(req, {})
    # Empty/redacted_thinking assistant turn preserved so user turns did NOT merge into illegal role sequence!
    roles = [m["role"] for m in spec.messages]
    assert roles[:3] == ["user", "assistant", "user"]
    assert spec.messages[1]["content"] == ""
    tool_msg = next(m for m in spec.messages if m["role"] == "tool")
    assert "<additional_tools>" in tool_msg["content"]
    assert "deferred_grep" in tool_msg["content"]
    tmpl_names = {t["function"]["name"] for t in (spec.template_tools or [])}
    assert tmpl_names == {"tool_search", "deferred_grep", "dynamic_bash"}
    # between_tools disables thinking
    assert spec.chat_template_kwargs.get("enable_thinking") is False

    # P0-2 & P1-2: non-empty thinking signature, omitted empty text block on tool-only turn,
    # explicit stop_sequence: null, and max_tokens > tool_use > stop_sequence stop priority
    gen = GenResult(
        reasoning="step 1",
        content="",
        tool_calls=[ToolCallItem(tool_index=0, name="tool_search", parameters="")],
        finish_reason="tool_calls",
        matched_stop="STOP",
        prompt_tokens=10,
        completion_tokens=4,
    )
    full = A.anthropic_full_response(gen, "claude-sonnet", uid=5)
    dumped = A._dump_anthropic_message(full)
    assert [b["type"] for b in dumped["content"]] == ["thinking", "tool_use"]
    assert len(dumped["content"][0]["signature"]) >= 16
    assert dumped["stop_reason"] == "tool_use"
    assert "stop_sequence" in dumped and dumped["stop_sequence"] is None

    # P1-1: Streaming empty-parameter tool call emits valid "{}" partial_json, never ""
    async def _ev_gen():
        from freetoken.server.generation import GenDone, ReasoningDelta, ToolCallsDelta

        yield ReasoningDelta("hmm")
        yield ToolCallsDelta([ToolCallItem(tool_index=0, name="ping", parameters="")])
        yield GenDone("tool_calls", 5, 3)

    async def _collect_an():
        out = []
        async for frame in A.anthropic_event_stream(_ev_gen(), "claude-sonnet", uid=6):
            lines = [ln for ln in frame.strip().splitlines() if ln.startswith("data: ")]
            if lines:
                out.append(json.loads(lines[0].removeprefix("data: ")))
        return out

    an_frames = _run(_collect_an())
    msg_start = an_frames[0]["message"]
    assert "stop_reason" in msg_start and msg_start["stop_reason"] is None
    assert "stop_sequence" in msg_start and msg_start["stop_sequence"] is None
    sig_deltas = [
        f["delta"]["signature"]
        for f in an_frames
        if f.get("type") == "content_block_delta"
        and (f.get("delta") or {}).get("type") == "signature_delta"
    ]
    assert sig_deltas and len(sig_deltas[0]) >= 16
    json_deltas = [
        f["delta"]["partial_json"]
        for f in an_frames
        if f.get("type") == "content_block_delta"
        and (f.get("delta") or {}).get("type") == "input_json_delta"
    ]
    assert json_deltas == ["{}"]

    # Built-in server tool rejected with clear 400
    srv_tool_req = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude-sonnet",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
    )
    with pytest.raises(ValueError, match="Server tool type"):
        A.convert_anthropic_to_genspec(srv_tool_req, {})
