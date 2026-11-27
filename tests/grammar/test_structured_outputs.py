"""End-to-end and unit tests for Issue #100: grammar-constrained decoding for Structured
Outputs and strict tool calling across OpenAI Chat/Completions/Responses and Anthropic Messages."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import jsonschema
import pytest
import torch
from freetoken.core import Batch, Req, SamplingParams, StructuredOutputSpec
from freetoken.engine.sample import LogitsPlan, Sampler, apply_logits_processors
from freetoken.grammar import (
    GrammarCompiler,
    XGrammarBackend,
    allocate_host_bitmask,
    apply_token_bitmask_inplace,
    build_structured_output_spec,
    is_xgrammar_installed,
)
from freetoken.message import TokenizeMsg, UserMsg
from freetoken.scheduler.prefill import ChunkedReq
from freetoken.scheduler.scheduler import Scheduler
from freetoken.server import anthropic_api as A
from freetoken.server import openai_api as OA
from freetoken.server import responses_api as RP
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.spec.dflash.proposer import DFlashProposer
from freetoken.spec.rejection import reject, reject_with_draft_probs
from freetoken.tokenizer.tokenize import TokenizeManager

pytestmark = pytest.mark.skipif(
    not is_xgrammar_installed(), reason="xgrammar not installed"
)


class _CharVocabTokenizer:
    """Minimal deterministic character + special-tag tokenizer for real xgrammar tests."""

    def __init__(self) -> None:
        self._tokens = [
            "<eos>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "\n",
        ] + [chr(i) for i in range(32, 127)]
        self._tok2id = {t: i for i, t in enumerate(self._tokens)}
        self.eos_token_id = 0
        self.unk_token_id = -1
        self.bos_token = None
        self.chat_template = "stub"

    def __len__(self) -> int:
        return len(self._tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._tok2id)

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._tok2id.get(tok, -1)

    def encode(self, text: str, add_special_tokens: bool = False, return_tensors=None):
        ids: list[int] = []
        i = 0
        specials = ("<eos>", "<think>", "</think>", "<tool_call>", "</tool_call>")
        while i < len(text):
            matched = False
            for sp in specials:
                if text.startswith(sp, i):
                    ids.append(self._tok2id[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                ids.append(self._tok2id[text[i]])
                i += 1
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.int64)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out: list[str] = []
        for tid in ids:
            tok = self._tokens[int(tid)]
            if skip_special_tokens and tok == "<eos>":
                continue
            out.append(tok)
        return "".join(out)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        del tokenize, kwargs
        parts = [f"{m['role']}:{m.get('content', '')}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


def _make_req(sp: SamplingParams, *, prompt_ids=(6, 7), uid: int = 1) -> Req:
    return Req(
        input_ids=torch.tensor(list(prompt_ids), dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=128,
        uid=uid,
        sampling_params=sp,
        cache_handle=None,  # type: ignore[arg-type]
    )


def test_json_schema_end_to_end_sampling_always_valid_json():
    tok = _CharVocabTokenizer()
    vocab = len(tok)
    compiler = GrammarCompiler(tok, vocab_size=vocab, stop_token_ids=[0])
    schema = {
        "$defs": {
            "Tag": {"type": "string", "enum": ["work", "home"]},
        },
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 4},
            "priority": {
                "anyOf": [{"type": "integer", "enum": [1, 2]}, {"type": "null"}]
            },
            "meta": {
                "type": "object",
                "properties": {"tag": {"$ref": "#/$defs/Tag"}},
                "required": ["tag"],
                "additionalProperties": False,
            },
        },
        "required": ["title", "priority", "meta"],
        "additionalProperties": False,
    }
    spec = build_structured_output_spec(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "task", "strict": True, "schema": schema},
        }
    )
    assert spec is not None and spec.kind == "json_schema"
    compiled = compiler.compile(spec)
    sampler = Sampler(device=torch.device("cpu"), vocab_size=vocab)

    # Drive both greedy and seeded sampling with adversarial logits favoring markdown
    # fences (`) and illegal keys so ONLY grammar masking can guarantee valid JSON.
    for sp in (
        SamplingParams(temperature=0.0, structured_output=spec),
        SamplingParams(temperature=0.9, top_p=0.95, seed=123, structured_output=spec),
    ):
        req = _make_req(sp, uid=1)
        req.grammar_matcher = compiler.create_matcher(compiled)
        gen = torch.Generator().manual_seed(7)
        out_ids: list[int] = []
        for _ in range(120):
            logits = torch.randn(1, vocab, generator=gen)
            logits[0, tok.convert_tokens_to_ids("`")] = (
                50.0  # adversarial markdown fence
            )
            tok_id = int(
                sampler.sample(
                    logits, sampler.prepare(Batch([req], phase="decode"))
                ).item()
            )
            if tok_id == tok.eos_token_id:
                break
            out_ids.append(tok_id)
            assert req.grammar_matcher.accept_token(tok_id)
            req.complete_one()
            if req.grammar_matcher.is_terminated():
                break
        text = tok.decode(out_ids)
        parsed = json.loads(text)
        jsonschema.validate(parsed, schema)
    compiler.shutdown()


def test_strict_schema_validation_and_wire_parity_all_three_apis():
    valid_strict = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    invalid_missing_ap = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    invalid_missing_req = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    invalid_ref = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/DoesNotExist"}},
        "required": ["x"],
        "additionalProperties": False,
    }

    class _StubState:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                max_output_tokens=64,
                reasoning_parser=None,
                tool_call_parser="qwen3_coder",
                enable_cache_report=False,
                enable_timings_report=False,
                thinking=None,
                anthropic_inline_system="auto",
                max_completion_prompts=256,
            )
            self.sent: list[TokenizeMsg] = []
            self._uid = 0

        def new_user(self) -> int:
            self._uid += 1
            return self._uid

        async def send_one(self, msg: TokenizeMsg) -> None:
            self.sent.append(msg)

        async def wait_for_ack(self, uid: int):
            from freetoken.message import UserReply

            yield UserReply(
                uid=uid,
                incremental_output='{"answer":"ok"}',
                finished=True,
                prompt_tokens_delta=4,
                completion_tokens_delta=3,
                finish_reason="stop",
            )

    state = _StubState()

    # 1. OpenAI Chat: strict schema rejects invalid schemas with 400 + code="invalid_json_schema".
    for bad in (invalid_missing_ap, invalid_missing_req, invalid_ref):
        req = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "s", "strict": True, "schema": bad},
                },
            }
        )
        resp = asyncio.run(OA.handle_chat_completion(req, None, state, {}))
        assert resp.status_code == 400
        assert json.loads(resp.body)["error"]["code"] == "invalid_json_schema"

    # OpenAI Chat: guided_choice & structured_outputs accepted and wired to StructuredOutputSpec.
    guided_req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "pick"}],
            "guided_choice": ["yes", "no"],
        }
    )
    gspec = OA.chat_request_to_genspec(guided_req, {})
    assert gspec.sampling_params.structured_output is not None
    assert gspec.sampling_params.structured_output.kind == "regex"

    # 2. OpenAI Completions: preserves response_format & attaches structured_output onto TokenizeMsg.
    cmpl_req = CompletionRequest.model_validate(
        {
            "model": "m",
            "prompt": "json:",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "s", "strict": True, "schema": valid_strict},
            },
        }
    )
    cmpl_body = asyncio.run(OA.handle_completion(cmpl_req, None, state, {}))
    assert cmpl_body["choices"][0]["text"] == '{"answer":"ok"}'
    assert state.sent[-1].response_format is not None
    assert state.sent[-1].sampling_params.structured_output is not None
    assert state.sent[-1].sampling_params.structured_output.kind == "json_schema"

    # 3. OpenAI Responses: echoes text.format, preserves tools[].strict, rejects invalid strict schema.
    bad_resp_req = RP.ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "bad",
                    "strict": True,
                    "schema": invalid_missing_ap,
                }
            },
        }
    )
    bad_r = asyncio.run(RP.handle_responses(bad_resp_req, None, state, {}))
    assert bad_r.status_code == 400
    assert json.loads(bad_r.body)["error"]["code"] == "invalid_json_schema"

    good_resp_req = RP.ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ans",
                    "strict": True,
                    "schema": valid_strict,
                }
            },
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "strict": True,
                    "parameters": valid_strict,
                }
            ],
        }
    )
    good_r = asyncio.run(RP.handle_responses(good_resp_req, None, state, {}))
    assert good_r.status_code == 200
    rbody = json.loads(good_r.body)
    assert rbody["text"]["format"]["type"] == "json_schema"
    assert rbody["text"]["format"]["name"] == "ans"
    assert rbody["text"]["format"]["strict"] is True
    conv_tools = RP._convert_tools(good_resp_req.tools)
    assert conv_tools is not None and conv_tools[0]["function"]["strict"] is True

    # 4. Anthropic Messages: beta output_format accepted, missing schema rejected (400),
    # tools[].strict preserved, and forced tool_choice ("any" / "tool") attaches grammar.
    bad_an = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": {"type": "json_schema"}},
        }
    )
    bad_an_r = asyncio.run(A.handle_anthropic_messages(bad_an, None, state, {}))
    assert bad_an_r.status_code == 400

    beta_an = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
            "output_format": {
                "type": "json_schema",
                "schema": valid_strict,
                "strict": True,
            },
        }
    )
    beta_spec = A.convert_anthropic_to_genspec(
        beta_an, {}, tool_call_parser="qwen3_coder"
    )
    assert beta_spec.sampling_params.structured_output is not None
    assert beta_spec.sampling_params.structured_output.kind == "json_schema"

    strict_tool_an = AnthropicMessagesRequest.model_validate(
        {
            "model": "claude",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "call it"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": valid_strict,
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
    )
    tool_spec = A.convert_anthropic_to_genspec(
        strict_tool_an, {}, tool_call_parser="qwen3_coder"
    )
    assert tool_spec.template_tools is not None
    assert tool_spec.template_tools[0]["function"]["strict"] is True
    assert tool_spec.sampling_params.structured_output is not None
    assert tool_spec.sampling_params.structured_output.kind == "structural_tag"


def test_msgpack_wire_roundtrip_and_prefix_cache_preserved():
    schema_with_type_key = {
        "type": "object",
        "properties": {"__type__": {"type": "string"}},
        "required": ["__type__"],
        "additionalProperties": False,
    }
    spec_a = build_structured_output_spec(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "a",
                "strict": True,
                "schema": schema_with_type_key,
            },
        }
    )
    spec_b = build_structured_output_spec(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "b",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"other": {"type": "integer"}},
                    "required": ["other"],
                    "additionalProperties": False,
                },
            },
        }
    )
    assert spec_a is not None and spec_b is not None

    # Msgpack round-trip for both TokenizeMsg and UserMsg.
    tmsg = TokenizeMsg(
        uid=1, text="hi", sampling_params=SamplingParams(structured_output=spec_a)
    )
    tdec = TokenizeMsg.decoder(TokenizeMsg.encoder(tmsg))
    assert tdec.sampling_params.structured_output == spec_a

    umsg = UserMsg(
        uid=1,
        input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        sampling_params=SamplingParams(structured_output=spec_a),
    )
    udec = UserMsg.decoder(umsg.encoder())
    assert udec.sampling_params.structured_output == spec_a

    # Prefix-cache preservation: two requests sharing a system prompt with DIFFERENT
    # JSON schemas render byte-identical prompts (no _render_directives mutation!).
    tm = TokenizeManager(_CharVocabTokenizer())
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract fields."},
    ]
    u_a, u_b = tm.tokenize(
        [
            TokenizeMsg(
                uid=1,
                text=msgs,
                sampling_params=SamplingParams(structured_output=spec_a),
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": schema_with_type_key},
                },
            ),
            TokenizeMsg(
                uid=2,
                text=msgs,
                sampling_params=SamplingParams(structured_output=spec_b),
                response_format={"type": "json_object"},
            ),
        ]
    )
    assert torch.equal(u_a.input_ids, u_b.input_ids)


def test_reasoning_deferral_and_optional_think_grammar():
    tok = _CharVocabTokenizer()
    vocab = len(tok)
    compiler = GrammarCompiler(tok, vocab_size=vocab, stop_token_ids=[0])
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    # Forced-reasoning (qwen3 with thinking on): defers grammar until </think>.
    spec = build_structured_output_spec(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "r", "strict": True, "schema": schema},
        },
        reasoning_parser="qwen3",
        chat_template_kwargs={"enable_thinking": True},
    )
    assert spec is not None and spec.defer_until_tag == "</think>"

    req = _make_req(SamplingParams(temperature=0.0, structured_output=spec), uid=1)
    req.grammar_matcher = compiler.create_matcher(compiler.compile(spec))
    req.grammar_think_end_ids = (tok.convert_tokens_to_ids("</think>"),)
    req.grammar_waiting_reasoning_end = True

    sampler = Sampler(device=torch.device("cpu"), vocab_size=vocab)
    # While waiting for </think>, logits are NOT masked (arbitrary reasoning text allowed).
    plan_during_think = sampler.prepare(Batch([req], phase="decode")).plan
    assert plan_during_think is None

    # Commit reasoning tokens + </think> via real Scheduler._commit_tokens.
    sched = object.__new__(Scheduler)
    sched.eos_token_ids = {0}
    sched.toolcall_anchor_id = None
    sched.tokenizer = tok
    think_ids = torch.tensor(tok.encode("thinking...</think>"), dtype=torch.int64)
    msg, n = sched._commit_tokens(req, think_ids)
    assert n == len(think_ids) and not msg.finished
    assert req.grammar_waiting_reasoning_end is False

    # Immediately after </think>, grammar activates and masks disallowed tokens.
    logits = torch.full((1, vocab), 1.0)
    logits[0, tok.convert_tokens_to_ids("X")] = 100.0  # illegal after </think>
    next_tok = int(
        sampler.sample(logits, sampler.prepare(Batch([req], phase="decode"))).item()
    )
    assert tok.decode([next_tok]) in ("{", " ", "\n", "\t", "\r")

    # Optional-think (ricursive): accepts both direct JSON and <think>...</think>\n{...}.
    opt_spec = build_structured_output_spec(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "r", "strict": True, "schema": schema},
        },
        reasoning_parser="ricursive",
        chat_template_kwargs={},
    )
    assert opt_spec is not None and opt_spec.kind == "structural_tag"
    opt_compiled = compiler.compile(opt_spec)
    for sample_str in ('{"ok":true}', '<think>plan</think>\n{"ok":false}'):
        m = compiler.create_matcher(opt_compiled)
        for tid in tok.encode(sample_str):
            assert m.accept_token(tid), (
                f"rejected at {tok.decode([tid])!r} in {sample_str!r}"
            )
        assert m.accept_token(0)
    m_bad = compiler.create_matcher(opt_compiled)
    for tid in tok.encode("<think>plan</think>"):
        assert m_bad.accept_token(tid)
    assert not m_bad.accept_token(tok.convert_tokens_to_ids("X"))
    compiler.shutdown()


def test_strict_and_forced_tool_calling_qwen3_coder_and_json():
    tok = _CharVocabTokenizer()
    compiler = GrammarCompiler(tok, vocab_size=len(tok), stop_token_ids=[0])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    # Forced tool_choice on qwen3_coder must accept valid XML call and reject plain prose.
    forced_qwen3 = build_structured_output_spec(
        tools=tools,
        tool_choice_mode="named:get_weather",
        tool_call_parser="qwen3_coder",
        disable_parallel_tool_use=True,
    )
    assert forced_qwen3 is not None
    m = compiler.create_matcher(compiler.compile(forced_qwen3))
    assert not m.accept_token(tok.convert_tokens_to_ids("H"))  # prose rejected
    valid_xml = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    for tid in tok.encode(valid_xml):
        assert m.accept_token(tid), f"failed on {tok.decode([tid])!r}"
    assert m.accept_token(0)

    # Auto mode with strict: true tool allows plain text OR strictly schema-valid tool call.
    auto_json = build_structured_output_spec(
        tools=tools,
        tool_choice_mode=None,
        tool_call_parser="qwen25",
    )
    assert auto_json is not None
    cg_auto = compiler.compile(auto_json)
    m_text = compiler.create_matcher(cg_auto)
    for tid in tok.encode("Hello!"):
        assert m_text.accept_token(tid)
    assert m_text.accept_token(0)

    m_call = compiler.create_matcher(cg_auto)
    valid_call = (
        '<tool_call>\n{"name":"get_weather","arguments":{"city":"Lyon"}}\n</tool_call>'
    )
    for tid in tok.encode(valid_call):
        assert m_call.accept_token(tid)
    assert m_call.accept_token(0)
    compiler.shutdown()


def test_bitmask_ordering_preserves_sole_valid_eos_under_min_tokens_and_min_p():
    vocab = 32
    logits = torch.full((1, vocab), 2.0)
    logits[0, 5] = 10.0  # disallowed by grammar
    logits[0, 0] = 0.1  # ONLY grammar-valid token (EOS)
    bm = allocate_host_bitmask(1, vocab)
    bm.zero_()
    bm[0, 0] = 1 << 0  # only token 0 (EOS) allowed

    plan = LogitsPlan(
        rows=torch.tensor([0], dtype=torch.int64),
        temps=torch.tensor([[1.0]], dtype=torch.float32),
        grammar_rows=torch.tensor([0], dtype=torch.int64),
        grammar_bitmask=bm,
        min_rows=torch.tensor([0], dtype=torch.int64),
        min_ids=torch.tensor([[0]], dtype=torch.int64),  # min_tokens tries to mask EOS!
        min_p=torch.tensor([[0.5]], dtype=torch.float32),
    )
    out = apply_logits_processors(logits, plan, vocab)
    assert torch.isfinite(out[0, 0])
    assert torch.isinf(out[0, 5]) and out[0, 5] < 0


def test_speculative_verify_chain_bitmask_rollback_and_dflash_topk_filter():
    tok = _CharVocabTokenizer()
    vocab = len(tok)
    compiler = GrammarCompiler(tok, vocab_size=vocab, stop_token_ids=[0])
    spec = StructuredOutputSpec(kind="regex", payload="ab(c|d)")
    compiled = compiler.compile(spec)
    sampler = Sampler(device=torch.device("cpu"), vocab_size=vocab)

    a_id, b_id, c_id, d_id, x_id = (
        tok.convert_tokens_to_ids(ch) for ch in ("a", "b", "c", "d", "x")
    )

    # 1. Verify chain with drafts [a (valid), x (illegal), c] -> slot 0 accepts 'a',
    # slot 1 masks out 'x' and picks 'b', matcher rolls back to pre-verify state S_0!
    req = _make_req(SamplingParams(temperature=0.0, structured_output=spec), uid=1)
    req.grammar_matcher = compiler.create_matcher(compiled, max_rollback_tokens=16)
    batch = Batch([req], phase="verify", rows_per_req=4)
    args = sampler.prepare(batch)
    drafts = torch.tensor([[a_id, x_id, c_id]], dtype=torch.int64)
    sampler.attach_verify_grammar(batch, args, drafts)

    # Verify matcher rolled back cleanly to S_0 (only 'a' is allowed now!).
    bm_check = allocate_host_bitmask(1, vocab)
    req.grammar_matcher.fill_next_token_bitmask(bm_check, 0)
    assert bool((int(bm_check[0, a_id >> 5].item()) >> (a_id & 31)) & 1)
    assert not bool((int(bm_check[0, b_id >> 5].item()) >> (b_id & 31)) & 1)

    # Adversarial target logits favor illegal 'x' at every slot; grammar forces [a, b].
    logits = torch.zeros(4, vocab)
    logits[:, x_id] = 50.0
    out, emitted = reject(logits.clone(), drafts, args, sampler)
    assert int(emitted[0].item()) == 1
    assert out[0, :2].tolist() == [a_id, b_id]

    # Also test reject_with_draft_probs (non-greedy) commits only grammar-legal tokens.
    req_ng = _make_req(
        SamplingParams(temperature=0.7, seed=5, structured_output=spec), uid=2
    )
    req_ng.grammar_matcher = compiler.create_matcher(compiled, max_rollback_tokens=16)
    batch_ng = Batch([req_ng], phase="verify", rows_per_req=4)
    args_ng = sampler.prepare(batch_ng)
    sampler.attach_verify_grammar(batch_ng, args_ng, drafts)
    support = torch.tensor(
        [[[a_id, x_id], [x_id, b_id], [c_id, d_id]]], dtype=torch.int64
    )
    q = torch.tensor([[[0.8, 0.2], [0.9, 0.1], [0.5, 0.5]]], dtype=torch.float32)
    out_ng, em_ng = reject_with_draft_probs(
        logits.clone(), drafts, support, q, args_ng, sampler
    )
    assert int(em_ng[0].item()) == 1
    assert out_ng[0, :2].tolist() == [a_id, b_id]

    # 2. Mid-block </think> during verify: slot 0 unconstrained, slot 1 = </think>,
    # slot 2 activates grammar immediately!
    req_th = _make_req(SamplingParams(temperature=0.0, structured_output=spec), uid=3)
    req_th.grammar_matcher = compiler.create_matcher(compiled, max_rollback_tokens=16)
    req_th.grammar_think_end_ids = (tok.convert_tokens_to_ids("</think>"),)
    req_th.grammar_waiting_reasoning_end = True
    batch_th = Batch([req_th], phase="verify", rows_per_req=4)
    args_th = sampler.prepare(batch_th)
    end_id = tok.convert_tokens_to_ids("</think>")
    drafts_th = torch.tensor([[x_id, end_id, x_id]], dtype=torch.int64)
    sampler.attach_verify_grammar(batch_th, args_th, drafts_th)
    assert req_th.grammar_waiting_reasoning_end is True  # unchanged until commit!
    logits_th = torch.zeros(4, vocab)
    logits_th[0, x_id] = 10.0  # reasoning text
    logits_th[1, end_id] = 10.0  # </think>
    logits_th[2, x_id] = 50.0  # illegal after </think> -> masked to -inf, picks 'a'!
    out_th, em_th = reject(logits_th, drafts_th, args_th, sampler)
    assert int(em_th[0].item()) == 2
    assert out_th[0, :3].tolist() == [x_id, end_id, a_id]

    # 3. DFlash Phase 2 top-K candidate grammar filter masks illegal candidates & rolls back.
    recorded_unary: list[torch.Tensor] = []

    class _FakeSelector:
        def sample_path(self, cids, unary, pred, pending, temp, greedy, uniforms):
            del pred, pending, temp, greedy, uniforms
            recorded_unary.append(unary.clone())
            best = unary.argmax(dim=-1, keepdim=True)
            toks = cids.gather(-1, best).squeeze(-1)
            probs = torch.softmax(unary, dim=-1)
            return toks, probs

    class _FakeDraft:
        candidate_selector = _FakeSelector()

        def forward_block(self, pending, starts, rows, ring):
            del starts, rows, ring
            return torch.zeros(pending.shape[0] * 4, 8)

        def compute_candidates(self, pred):
            del pred
            cids = torch.tensor(
                [[[x_id, a_id], [x_id, b_id], [x_id, c_id]]], dtype=torch.int64
            )
            unary = torch.tensor(
                [[[50.0, 1.0], [50.0, 1.0], [50.0, 1.0]]], dtype=torch.float32
            )
            return cids, unary

    proposer = object.__new__(DFlashProposer)
    proposer.device = torch.device("cpu")
    proposer.k = 3
    proposer.block = 4
    proposer.draft_config = SimpleNamespace(vocab_size=vocab)
    proposer.dflash_ring = None
    proposer._graphs = {}
    proposer._comm = None
    proposer._tp_info = SimpleNamespace(is_primary=lambda: True)
    proposer.draft = _FakeDraft()

    req_df = _make_req(SamplingParams(temperature=0.0, structured_output=spec), uid=4)
    req_df.grammar_matcher = compiler.create_matcher(compiled, max_rollback_tokens=16)
    prop = proposer.propose(
        [req_df],
        torch.tensor([4], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
    )
    assert prop.draft_token_ids[0].tolist() == [a_id, b_id, c_id]
    assert torch.isinf(recorded_unary[0][0, 0, 0]) and recorded_unary[0][0, 0, 0] < 0
    # Matcher state rolled back cleanly to S_0 after drafting.
    assert req_df.grammar_matcher.accept_token(a_id)
    compiler.shutdown()


def test_mixed_batch_overlap_barrier_and_compiler_lru_cache():
    tok = _CharVocabTokenizer()
    vocab = len(tok)
    backend = XGrammarBackend(tok, vocab_size=vocab, stop_token_ids=[0])
    compile_calls = 0
    orig_compile = backend.compile

    def _counting_compile(spec):
        nonlocal compile_calls
        compile_calls += 1
        return orig_compile(spec)

    backend.compile = _counting_compile  # type: ignore[method-assign]
    compiler = GrammarCompiler(tok, vocab_size=vocab, backend=backend)

    spec = StructuredOutputSpec(kind="regex", payload="Z")
    f1 = compiler.compile_async(spec)
    c1 = f1.result(timeout=10)
    f2 = compiler.compile_async(spec)
    c2 = f2.result(timeout=10)
    assert c1 is c2 and compile_calls == 1

    # Mixed batch: constrained + unconstrained + greedy + seeded.
    z_id = tok.convert_tokens_to_ids("Z")
    x_id = tok.convert_tokens_to_ids("X")
    r_gram = _make_req(SamplingParams(temperature=0.8, structured_output=spec), uid=1)
    r_gram.grammar_matcher = compiler.create_matcher(c1)
    r_free = _make_req(SamplingParams(temperature=0.0), uid=2)
    r_seed = _make_req(SamplingParams(temperature=0.8, seed=42), uid=3)
    sampler = Sampler(device=torch.device("cpu"), vocab_size=vocab)
    logits = torch.full((3, vocab), -5.0)
    logits[:, x_id] = 20.0
    logits[0, z_id] = 1.0
    toks = sampler.sample(
        logits, sampler.prepare(Batch([r_gram, r_free, r_seed], phase="decode"))
    )
    assert int(toks[0].item()) == z_id
    assert int(toks[1].item()) == x_id

    # Overlap barrier predicate: False for unconstrained / ChunkedReq, True for active constrained.
    free_batch = Batch([r_free], phase="decode")
    gram_batch = Batch([r_gram], phase="decode")
    assert Scheduler._needs_grammar_sync(None) is False
    assert (
        Scheduler._needs_grammar_sync((SimpleNamespace(batch=free_batch), None))
        is False
    )
    assert (
        Scheduler._needs_grammar_sync((SimpleNamespace(batch=gram_batch), None)) is True
    )

    chunked = ChunkedReq(
        input_ids=torch.tensor([6, 7], dtype=torch.int32),
        table_idx=9,
        cached_len=0,
        output_len=16,
        uid=9,
        sampling_params=SamplingParams(structured_output=spec),
        cache_handle=None,  # type: ignore[arg-type]
    )
    chunk_batch = Batch([chunked], phase="prefill")
    assert (
        Scheduler._needs_grammar_sync((SimpleNamespace(batch=chunk_batch), None))
        is False
    )

    # Standalone bitmask applier sanity on multi-row subset.
    sub_logits = torch.ones(2, vocab)
    bm = allocate_host_bitmask(1, vocab)
    bm.zero_()
    bm[0, z_id >> 5] = 1 << (z_id & 31)
    apply_token_bitmask_inplace(
        sub_logits, bm, torch.tensor([1], dtype=torch.int64), vocab
    )
    assert torch.isfinite(sub_logits[0]).all()  # row 0 untouched
    assert torch.isfinite(sub_logits[1, z_id]) and torch.isinf(sub_logits[1, x_id])
    compiler.shutdown()


def test_llguidance_fallback_backend_json_schema_regex_and_rollback():
    from freetoken.grammar import LLGuidanceBackend, is_llguidance_installed

    if not is_llguidance_installed():
        pytest.skip("llguidance not installed")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {"<eos>": 0}
    for idx, ch in enumerate(range(33, 127), start=1):
        vocab[chr(ch)] = idx
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<eos>")

    backend = LLGuidanceBackend(hf_tok, vocab_size=len(hf_tok), stop_token_ids=[0])
    schema_spec = StructuredOutputSpec(
        kind="json_schema",
        payload=json.dumps(
            {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            }
        ),
        strict=True,
    )
    matcher = backend.create_matcher(backend.compile(schema_spec))
    bm = allocate_host_bitmask(1, len(hf_tok))
    matcher.fill_next_token_bitmask(bm, 0)
    open_id = vocab["{"]
    bad_id = vocab["X"]
    assert bool((int(bm[0, open_id >> 5].item()) >> (open_id & 31)) & 1)
    assert not bool((int(bm[0, bad_id >> 5].item()) >> (bad_id & 31)) & 1)

    for ch in '{"ok":true}':
        assert matcher.accept_token(vocab[ch])
    matcher.rollback(5)  # rewind "true}"
    for ch in "false}":
        assert matcher.accept_token(vocab[ch])
    assert matcher.is_terminated() or matcher.accept_token(0)
    assert matcher.is_terminated()

    regex_matcher = backend.create_matcher(
        backend.compile(StructuredOutputSpec(kind="regex", payload="ab"))
    )
    assert not regex_matcher.accept_token(vocab["b"])
    assert regex_matcher.accept_token(vocab["a"])
    assert regex_matcher.accept_token(vocab["b"])
    assert regex_matcher.is_terminated() or regex_matcher.accept_token(0)
