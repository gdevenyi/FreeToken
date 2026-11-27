"""Tests for Issue #101: vectorized inverse-CDF seeded sampling, DFlash seeded drafting,
seed validation / collision-free choice seeds, and deterministic system_fingerprint."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.fingerprint import compute_system_fingerprint
from freetoken.engine.sample import (
    _DRAFT_SEED_DOMAIN,
    Sampler,
    mix_choice_seed,
    sample_seeded_row,
    step_uniform,
)
from freetoken.kernel import torch_sampling
from freetoken.kvcache.cache_status import compute_cache_status_meta
from freetoken.message import TokenizeMsg, UserMsg
from freetoken.server import openai_api as OA
from freetoken.server import responses_api as RP
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.server.generation import INT64_MAX, INT64_MIN, resolve_sampling
from freetoken.spec.dflash.proposer import DFlashProposer
from freetoken.spec.rejection import reject, reject_with_draft_probs


def _make_req(
    sp: SamplingParams, *, prompt_ids=(1, 2, 3), gen_len: int = 0, uid: int = 1
) -> Req:
    req = Req(
        input_ids=torch.tensor(list(prompt_ids), dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=32,
        uid=uid,
        sampling_params=sp,
        cache_handle=None,  # type: ignore[arg-type]
    )
    if gen_len > 0:
        req.cached_len = len(prompt_ids) + gen_len - 1
        req.device_len = len(prompt_ids) + gen_len
    return req


def test_vectorized_seeded_sampling_matches_sample_verify_and_dflash():
    vocab = 64
    sampler = Sampler(device=torch.device("cpu"), vocab_size=vocab)
    gen = torch.Generator().manual_seed(2026)
    base_logits = torch.randn(4, vocab, generator=gen)
    noise_logits = torch.randn(4, vocab, generator=gen)

    sp_seeded = SamplingParams(
        temperature=0.85,
        top_k=12,
        top_p=0.85,
        min_p=0.04,
        presence_penalty=0.35,
        frequency_penalty=0.15,
        repetition_penalty=1.12,
        seed=9001,
    )
    sp_unseeded = SamplingParams(temperature=0.9, top_k=16, top_p=0.9, seed=None)
    sp_greedy = SamplingParams(temperature=0.0)

    # Step-by-step solo decode with real penalty/repetition accumulation.
    solo_req = _make_req(sp_seeded, prompt_ids=(2, 5, 7), gen_len=0, uid=1)
    solo_seq: list[int] = []
    for step in range(4):
        batch = Batch([solo_req], phase="decode")
        tok = int(
            sampler.sample(
                base_logits[step : step + 1].clone(), sampler.prepare(batch)
            ).item()
        )
        solo_seq.append(tok)
        solo_req.complete_one()

    # Mixed batch (unseeded + seeded + greedy) must match solo step-for-step.
    mix_seeded = _make_req(sp_seeded, prompt_ids=(2, 5, 7), gen_len=0, uid=2)
    mix_unseeded = _make_req(sp_unseeded, prompt_ids=(1, 2), gen_len=0, uid=3)
    mix_greedy = _make_req(sp_greedy, prompt_ids=(3, 4), gen_len=0, uid=4)
    mix_seq: list[int] = []
    for step in range(4):
        batch = Batch([mix_unseeded, mix_seeded, mix_greedy], phase="decode")
        logits = torch.stack(
            [noise_logits[step], base_logits[step], noise_logits[(step + 1) % 4]],
            dim=0,
        )
        toks = sampler.sample(logits, sampler.prepare(batch))
        mix_seq.append(int(toks[1].item()))
        mix_seeded.complete_one()

    assert mix_seq == solo_seq

    # Speculative verify (sample_verify, reject, and reject_with_draft_probs) with exact
    # prefix drafts must match solo_seq across all 4 slots (including penalties!).
    k = 3
    drafts_exact = torch.tensor([solo_seq[:k]], dtype=torch.int64)
    v_req = _make_req(sp_seeded, prompt_ids=(2, 5, 7), gen_len=0, uid=5)
    v_args = sampler.prepare(Batch([v_req], phase="decode"))
    v_toks = sampler.sample_verify(base_logits.clone(), v_args, drafts_exact)
    assert v_toks.tolist() == solo_seq

    r_out, r_em = reject(base_logits.clone(), drafts_exact, v_args, sampler)
    assert int(r_em[0].item()) == k
    assert r_out[0].tolist() == solo_seq

    # All-seeded DFlash reject_with_draft_probs delegates cleanly and matches solo_seq.
    support = torch.tensor([[[0, 1], [0, 1], [0, 1]]], dtype=torch.int64)
    q = torch.tensor([[[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]]], dtype=torch.float32)
    df_out, df_em = reject_with_draft_probs(
        base_logits.clone(), drafts_exact, support, q, v_args, sampler
    )
    assert int(df_em[0].item()) == k
    assert df_out[0].tolist() == solo_seq

    # Mixed batch in reject_with_draft_probs (unseeded + seeded + greedy) matches solo_seq
    # on the seeded row regardless of the unseeded row's draft probabilities.
    m_reqs = [
        _make_req(sp_unseeded, prompt_ids=(1, 2), gen_len=0, uid=6),
        _make_req(sp_seeded, prompt_ids=(2, 5, 7), gen_len=0, uid=7),
        _make_req(sp_greedy, prompt_ids=(3, 4), gen_len=0, uid=8),
    ]
    m_args = sampler.prepare(Batch(m_reqs, phase="decode"))
    assert not m_args.all_non_greedy_seeded and len(m_args.seeded_rows) == 1
    m_logits = torch.cat([noise_logits, base_logits, noise_logits], dim=0)
    m_drafts = torch.tensor(
        [
            [0, 1, 2],
            [solo_seq[0], (solo_seq[1] + 1) % vocab, 0],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )
    m_support = support.repeat(3, 1, 1)
    m_q = q.repeat(3, 1, 1)
    m_out, m_em = reject_with_draft_probs(
        m_logits, m_drafts, m_support, m_q, m_args, sampler
    )
    assert int(m_em[1].item()) == 1
    assert m_out[1, :2].tolist() == solo_seq[:2]

    # Reference helper sample_seeded_row matches Sampler.sample() on identical processed logits.
    sp_plain = SamplingParams(temperature=0.85, top_k=12, top_p=0.85, seed=777)
    ref_req = _make_req(sp_plain, gen_len=3, uid=9)
    samp_tok = int(
        sampler.sample(
            base_logits[0:1].clone(), sampler.prepare(Batch([ref_req], phase="decode"))
        ).item()
    )
    ref_tok = int(
        sample_seeded_row(
            base_logits[0], seed=777, step=3, temp=0.85, top_k=12, top_p=0.85
        ).item()
    )
    assert samp_tok == ref_tok


def test_mixed_batch_dispatch_invariance_and_zero_prob_tail_guard():
    vocab = 48
    sampler = Sampler(device=torch.device("cpu"), vocab_size=vocab)
    logits_a = torch.randn(1, vocab, generator=torch.Generator().manual_seed(11))
    logits_b = torch.randn(1, vocab, generator=torch.Generator().manual_seed(22))

    # Row A uses only top_k; Row B uses only top_p. Mixed batch dispatches top_k_top_p.
    sp_a = SamplingParams(temperature=0.9, top_k=8, top_p=1.0, seed=101)
    sp_b = SamplingParams(temperature=0.9, top_k=-1, top_p=0.75, seed=202)

    for step in range(6):
        solo_a = int(
            sampler.sample(
                logits_a.clone(),
                sampler.prepare(Batch([_make_req(sp_a, gen_len=step)], phase="decode")),
            ).item()
        )
        solo_b = int(
            sampler.sample(
                logits_b.clone(),
                sampler.prepare(Batch([_make_req(sp_b, gen_len=step)], phase="decode")),
            ).item()
        )
        mixed = sampler.sample(
            torch.cat([logits_a, logits_b], dim=0),
            sampler.prepare(
                Batch(
                    [
                        _make_req(sp_a, gen_len=step, uid=1),
                        _make_req(sp_b, gen_len=step, uid=2),
                    ],
                    phase="decode",
                )
            ),
        )
        assert mixed.tolist() == [solo_a, solo_b]

    # Zero-prob tail guard: probs concentrated on indices 1 and 3 with trailing zeros.
    probs = torch.zeros(1, 32, dtype=torch.float32)
    probs[0, 1] = 0.25
    probs[0, 3] = 0.75
    u_hi = torch.tensor([1.0 - 2.0**-24], dtype=torch.float32)
    u_lo = torch.tensor([0.0], dtype=torch.float32)
    assert int(torch_sampling.sampling_from_probs(probs, uniforms=u_hi).item()) == 3
    assert int(torch_sampling.sampling_from_probs(probs, uniforms=u_lo).item()) == 1


def test_dflash_seeded_propose_uses_deterministic_uniforms_without_forcing_greedy():
    recorded: dict[str, torch.Tensor] = {}

    class _FakeSelector:
        def sample_path(
            self, candidate_ids, unary_logits, pred, pending, temp, greedy, uniforms
        ):
            recorded["temp"] = temp.clone()
            recorded["greedy"] = greedy.clone()
            recorded["uniforms"] = uniforms.clone()
            bs, k, top_k = candidate_ids.shape
            toks = candidate_ids[:, :, 0]
            q = torch.full((bs, k, top_k), 1.0 / top_k, dtype=torch.float32)
            return toks, q

    class _FakeDraft:
        candidate_selector = _FakeSelector()

        def forward_block(self, pending, starts, rows, ring):
            return torch.zeros(pending.shape[0] * 4, 8, dtype=torch.float32)

        def compute_candidates(self, pred):
            bs, k = pred.shape[:2]
            cands = torch.arange(4, dtype=torch.int64).view(1, 1, 4).repeat(bs, k, 1)
            logits = torch.zeros(bs, k, 4, dtype=torch.float32)
            return cands, logits

    proposer = object.__new__(DFlashProposer)
    proposer.device = torch.device("cpu")
    proposer.k = 3
    proposer.block = 4
    proposer.dflash_ring = None
    proposer._graphs = {}
    proposer._comm = None
    proposer.draft = _FakeDraft()

    # Simulate Scheduler._prepare_batch: req has 4 prompt tokens + 2 generated tokens,
    # then widen_pending(3) advances device_len before propose() is called.
    r_seeded = _make_req(
        SamplingParams(temperature=0.8, seed=42),
        prompt_ids=(1, 2, 3, 4),
        gen_len=2,
        uid=1,
    )
    r_seeded.widen_pending(3)
    r_greedy = _make_req(
        SamplingParams(temperature=0.0, seed=42),
        prompt_ids=(1, 2, 3, 4),
        gen_len=2,
        uid=2,
    )
    r_greedy.widen_pending(3)

    proposal = proposer.propose(
        [r_seeded, r_greedy],
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([9, 8], dtype=torch.int64),
    )

    assert recorded["greedy"].tolist() == [False, True]
    assert recorded["temp"].tolist() == pytest.approx([0.8, 1.0])
    expected_u = [step_uniform(42, 2 + j, domain=_DRAFT_SEED_DOMAIN) for j in range(3)]
    assert recorded["uniforms"][0].tolist() == pytest.approx(expected_u)
    assert proposal.draft_support_probs is not None
    assert proposal.draft_support_ids is not None


def test_seed_validation_and_collision_free_choice_seeds():
    base_kw = {
        "temperature": 0.8,
        "top_k": None,
        "top_p": None,
        "max_tokens": 16,
        "ignore_eos": False,
        "model_sampling": {},
    }
    for bad in (True, False, 1.5, "42", INT64_MIN - 1, INT64_MAX + 1):
        with pytest.raises(ValueError) as ei:
            resolve_sampling(**base_kw, seed=bad)
        assert getattr(ei.value, "param", None) == "seed"

    # Boundary signed int64 values pass and choice seeds never overflow or collide.
    for edge in (INT64_MIN, 0, 42, INT64_MAX):
        sp = resolve_sampling(**base_kw, seed=edge)
        assert sp.seed == edge
        derived = [mix_choice_seed(edge, i) for i in range(4)]
        assert derived[0] == edge
        assert len(set(derived)) == 4
        assert all(INT64_MIN <= s <= INT64_MAX for s in derived)
        if edge < INT64_MAX:
            assert derived[1] != edge + 1
        # Verify msgpack wire round-trip at int64 boundaries.
        msg = TokenizeMsg(
            uid=1, text="hi", sampling_params=SamplingParams(seed=derived[1])
        )
        assert (
            TokenizeMsg.decoder(TokenizeMsg.encoder(msg)).sampling_params.seed
            == derived[1]
        )
        umsg = UserMsg(
            uid=1,
            input_ids=torch.tensor([1, 2], dtype=torch.int32),
            sampling_params=SamplingParams(seed=derived[1]),
        )
        assert UserMsg.decoder(umsg.encoder()).sampling_params.seed == derived[1]

    # HTTP endpoints return 400 with param="seed" for bool and out-of-range seeds.
    state = SimpleNamespace(config=SimpleNamespace(max_output_tokens=64))
    for bad in (True, INT64_MAX + 1):
        chat_req = ChatCompletionRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "seed": bad,
            }
        )
        chat_resp = asyncio.run(OA.handle_chat_completion(chat_req, None, state, {}))
        assert chat_resp.status_code == 400
        assert json.loads(chat_resp.body)["error"]["param"] == "seed"

        cmpl_req = CompletionRequest.model_validate(
            {"model": "m", "prompt": "hi", "seed": bad}
        )
        cmpl_resp = asyncio.run(OA.handle_completion(cmpl_req, None, state, {}))
        assert cmpl_resp.status_code == 400
        assert json.loads(cmpl_resp.body)["error"]["param"] == "seed"

        resp_req = RP.ResponsesRequest.model_validate(
            {"model": "m", "input": "hi", "seed": bad}
        )
        resp_out = asyncio.run(RP.handle_responses(resp_req, None, state, {}))
        assert resp_out.status_code == 400
        assert json.loads(resp_out.body)["error"]["param"] == "seed"


def test_system_fingerprint_stability_sensitivity_and_meta_propagation(monkeypatch):
    @dataclass
    class _Cfg:
        model_path: str = "/models/qwen38-nvfp4"
        dtype: torch.dtype = torch.bfloat16
        quant_backend: str = "auto"
        kv_cache_dtype: str = "fp8_e4m3"
        attention_backend: str = "triton"
        moe_strategy: str = "hybrid"
        moe_ep: bool = True
        ple_backend: str = "host"
        ple_table: str | None = "/models/ple-lut4"
        speculative_algorithm: str | None = "dflash"
        speculative_draft_model_path: str | None = "/models/dflash"
        speculative_num_steps: int = 3
        tp_info: Any = field(default_factory=lambda: SimpleNamespace(rank=0, size=2))

    # Clear env flags for deterministic baseline.
    for var in (
        "FREETOKEN_FP8_DENSE",
        "FREETOKEN_FP8_LMHEAD",
        "FREETOKEN_INT8_DENSE",
        "FREETOKEN_HC_INT8",
        "FREETOKEN_FP32_ROUTER",
        "FREETOKEN_VERIFY_INVARIANT",
    ):
        monkeypatch.delenv(var, raising=False)

    fp_r0 = compute_system_fingerprint(_Cfg(), git_commit="5b763ab503cb")
    fp_r1 = compute_system_fingerprint(
        _Cfg(tp_info=SimpleNamespace(rank=1, size=2)), git_commit="5b763ab503cb"
    )
    assert fp_r0.startswith("fp_") and len(fp_r0) == 15
    assert fp_r0 == fp_r1  # rank-independent across TP workers

    # Any material backend shift changes the fingerprint.
    assert fp_r0 != compute_system_fingerprint(
        _Cfg(model_path="/models/qwen38-strata"), git_commit="5b763ab503cb"
    )
    assert fp_r0 != compute_system_fingerprint(
        _Cfg(speculative_algorithm=None), git_commit="5b763ab503cb"
    )
    assert fp_r0 != compute_system_fingerprint(
        _Cfg(tp_info=SimpleNamespace(rank=0, size=1)), git_commit="5b763ab503cb"
    )
    monkeypatch.setenv("FREETOKEN_INT8_DENSE", "1")
    assert fp_r0 != compute_system_fingerprint(_Cfg(), git_commit="5b763ab503cb")
    monkeypatch.delenv("FREETOKEN_INT8_DENSE")
    monkeypatch.setenv("FREETOKEN_VERIFY_INVARIANT", "1")
    assert fp_r0 != compute_system_fingerprint(_Cfg(), git_commit="5b763ab503cb")

    # Readiness metadata surfaces system_fingerprint cleanly.
    fake_engine = SimpleNamespace(
        config=None,
        kv_cache=None,
        moe_offload_cache=None,
        linear_state_pool=None,
        _post_weights_free=0,
        _baseline_free=0,
        _weights_bytes=0,
        system_fingerprint=fp_r0,
    )
    meta = compute_cache_status_meta(fake_engine)
    assert meta["system_fingerprint"] == fp_r0
