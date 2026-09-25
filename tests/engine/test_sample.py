"""Presence/frequency penalties in engine.sample.Sampler (greedy and sampled rows)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine import sample as S


def _batch(*params: SamplingParams, can_decode: bool = True):
    reqs = [
        SimpleNamespace(sampling_params=p, can_decode=can_decode, output_token_counts=None)
        for p in params
    ]
    return SimpleNamespace(reqs=reqs)


def test_no_penalty_allocates_nothing():
    sampler = S.Sampler(torch.device("cpu"), 5)
    batch = _batch(SamplingParams(temperature=0.0))
    args = sampler.prepare(batch)
    assert args.penalties == [] and batch.reqs[0].output_token_counts is None


def test_frequency_penalty_counts_generated_tokens_on_the_greedy_path():
    sampler = S.Sampler(torch.device("cpu"), 5)
    batch = _batch(SamplingParams(temperature=0.0, frequency_penalty=10.0))
    logits = torch.tensor([[0.0, 5.0, 1.0, 0.0, 0.0]])

    assert sampler.sample(logits, sampler.prepare(batch)).tolist() == [1]
    # Token 1 now carries -10, so the runner-up wins; the caller's logits stay untouched.
    assert sampler.sample(logits, sampler.prepare(batch)).tolist() == [2]
    assert batch.reqs[0].output_token_counts.tolist() == [0, 1, 1, 0, 0]
    assert logits[0, 1] == 5.0


def test_intermediate_prefill_chunks_are_not_counted():
    sampler = S.Sampler(torch.device("cpu"), 5)
    batch = _batch(SamplingParams(temperature=0.0, presence_penalty=1.0), can_decode=False)
    assert sampler.prepare(batch).penalties == []


def test_penalized_logits_reach_the_probability_sampler(monkeypatch):
    """Sampled rows (the torch top-k/top-p fallback on Pascal) see the penalized logits."""
    seen = {}

    def fake_sample_impl(logits, temperatures, top_k, top_p):
        seen["logits"] = logits.clone()
        return torch.argmax(logits, dim=-1)

    monkeypatch.setattr(S, "sample_impl", fake_sample_impl)
    monkeypatch.setattr(S, "make_device_tensor", lambda data, dtype, device: torch.tensor(data, dtype=dtype))
    sampler = S.Sampler(torch.device("cpu"), 4)
    batch = _batch(
        SamplingParams(temperature=1.0, top_k=2, presence_penalty=3.0),
        SamplingParams(temperature=1.0, top_k=2),
    )
    batch.reqs[0].output_token_counts = torch.tensor([0, 2, 0, 0], dtype=torch.int32)
    logits = torch.tensor([[0.0, 4.0, 2.0, 0.0], [0.0, 4.0, 2.0, 0.0]])

    tokens = sampler.sample(logits, sampler.prepare(batch))

    assert seen["logits"][0].tolist() == [0.0, 1.0, 2.0, 0.0]  # presence: -3 once, not per count
    assert seen["logits"][1].tolist() == [0.0, 4.0, 2.0, 0.0]
    assert tokens.tolist() == [2, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_penalty_through_the_pre_volta_torch_sampling_fallback(monkeypatch):
    import freetoken.kernel.backend as backend
    import freetoken.utils.arch as arch

    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: False)
    monkeypatch.setattr(arch, "is_sm70_supported", lambda: False)
    sampler = S.Sampler(torch.device("cuda"), 4)
    batch = _batch(SamplingParams(temperature=1.0, top_k=1, frequency_penalty=5.0))
    logits = torch.tensor([[0.0, 4.0, 2.0, 0.0]], device="cuda")

    assert sampler.sample(logits, sampler.prepare(batch)).tolist() == [1]
    assert sampler.sample(logits, sampler.prepare(batch)).tolist() == [2]
