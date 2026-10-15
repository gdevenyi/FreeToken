from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import BatchSamplingArgs, Sampler


def test_compute_logprobs_matches_raw_log_softmax_and_sorted_top() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)

    logits = torch.tensor(
        [
            [2.0, 0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([2, 3, 0], dtype=torch.long)
    args = BatchSamplingArgs(
        temperatures=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        logprob_row_idx=torch.tensor([0, 2]),
        max_top_logprobs=3,
    )

    result = sampler.compute_logprobs(logits, sampled_tokens, args)

    assert result is not None
    chosen_logprobs, top_ids, top_logprobs = result
    expected = torch.log_softmax(logits.float(), dim=-1)

    assert torch.isclose(chosen_logprobs[0], expected[0, sampled_tokens[0]])
    assert torch.isnan(chosen_logprobs[1])
    assert torch.isclose(chosen_logprobs[2], expected[2, sampled_tokens[2]])

    expected_top0 = torch.topk(expected[0], k=3)
    expected_top2 = torch.topk(expected[2], k=3)
    assert torch.equal(top_ids[0], expected_top0.indices)
    assert torch.allclose(top_logprobs[0], expected_top0.values)
    assert torch.equal(top_ids[2], expected_top2.indices)
    assert torch.allclose(top_logprobs[2], expected_top2.values)

    assert torch.equal(top_ids[1], torch.full((3,), -1, dtype=torch.int32))
    assert torch.isneginf(top_logprobs[1]).all()
    assert top_ids.shape == (3, 3)
    assert top_logprobs.shape == (3, 3)


def test_compute_logprobs_ignores_temperatures() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)

    logits = torch.tensor(
        [
            [0.5, 1.0, 2.0, 3.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([1, 2], dtype=torch.long)

    cold = BatchSamplingArgs(
        temperatures=torch.full((2,), 0.3, dtype=torch.float32),
        logprob_all_rows=True,
        max_top_logprobs=2,
    )
    hot = BatchSamplingArgs(
        temperatures=torch.full((2,), 2.5, dtype=torch.float32),
        logprob_all_rows=True,
        max_top_logprobs=2,
    )

    cold_out = sampler.compute_logprobs(logits, sampled_tokens, cold)
    hot_out = sampler.compute_logprobs(logits, sampled_tokens, hot)

    assert cold_out is not None and hot_out is not None
    cold_chosen, cold_top_ids, cold_top_logprobs = cold_out
    hot_chosen, hot_top_ids, hot_top_logprobs = hot_out

    assert torch.allclose(cold_chosen, hot_chosen)
    assert torch.equal(cold_top_ids, hot_top_ids)
    assert torch.allclose(cold_top_logprobs, hot_top_logprobs)


def test_compute_logprobs_returns_none_without_requested_rows() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.zeros((2, 4), dtype=torch.float32)
    sampled_tokens = torch.tensor([0, 1], dtype=torch.long)
    args = BatchSamplingArgs(
        temperatures=torch.ones(2),
        logprob_row_idx=torch.empty(0, dtype=torch.int64),
        max_top_logprobs=2,
    )

    assert sampler.compute_logprobs(logits, sampled_tokens, args) is None


def test_compute_logprobs_handles_zero_top_logprobs() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([3, 0, 1], dtype=torch.long)
    args = BatchSamplingArgs(
        temperatures=torch.full((3,), 1.0),
        logprob_row_idx=torch.tensor([0, 2]),
        max_top_logprobs=0,
    )

    result = sampler.compute_logprobs(logits, sampled_tokens, args)

    assert result is not None
    chosen_logprobs, top_ids, top_logprobs = result
    expected = torch.log_softmax(logits, dim=-1)
    assert torch.isclose(chosen_logprobs[0], expected[0, 3])
    assert torch.isnan(chosen_logprobs[1])
    assert torch.isclose(chosen_logprobs[2], expected[2, 1])
    assert top_ids.shape == (3, 0)
    assert top_logprobs.shape == (3, 0)


def test_compute_logprobs_with_prepare_keeps_max_top_logprobs_clamped(monkeypatch) -> None:
    # prepare() pins host tensors for the H2D copy; pinning needs a CUDA context,
    # so stub the transfer helper to keep this test host-agnostic.
    import freetoken.engine.sample as sample_mod

    monkeypatch.setattr(
        sample_mod, "make_device_tensor",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype),
    )
    sampler = Sampler(torch.device("cpu"), vocab_size=5)

    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(sampling_params=SamplingParams(logprobs=True, top_logprobs=17)),
            SimpleNamespace(sampling_params=SamplingParams(logprobs=False, top_logprobs=0)),
        ]
    )
    args = sampler.prepare(batch)
    assert args.max_top_logprobs == sampler.vocab_size

    logits = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.5, 2.0],
            [2.0, 1.0, 0.0, -1.0, -2.0],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([4, 0], dtype=torch.long)

    result = sampler.compute_logprobs(logits, sampled_tokens, args)
    assert result is not None
    chosen_logprobs, top_ids, top_logprobs = result

    assert chosen_logprobs.shape == (2,)
    assert torch.isclose(chosen_logprobs[0], torch.log_softmax(logits[0], dim=-1)[4])
    assert torch.isnan(chosen_logprobs[1])

    assert top_ids.shape == (2, sampler.vocab_size)
    assert top_logprobs.shape == (2, sampler.vocab_size)
    assert torch.equal(top_ids[1], torch.full((sampler.vocab_size,), -1, dtype=torch.int32))
    assert torch.isneginf(top_logprobs[1]).all()


def _reference_logprobs(logits, sampled_tokens, want, k):
    """The pre-index implementation (bool mask + torch.nonzero), kept as the oracle."""
    mask = torch.tensor(want, dtype=torch.bool)
    rows = torch.nonzero(mask, as_tuple=False).flatten()
    if rows.numel() == 0:
        return None
    logprobs = torch.log_softmax(logits.index_select(0, rows).float(), dim=-1)
    chosen = torch.full((logits.shape[0],), float("nan"))
    chosen.index_copy_(
        0, rows, logprobs[torch.arange(rows.numel()), sampled_tokens.long().index_select(0, rows)]
    )
    top_ids = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    top_logprobs = torch.full((logits.shape[0], k), float("-inf"))
    if k > 0:
        values, ids = torch.topk(logprobs, k=k, dim=-1)
        top_ids[rows] = ids.to(torch.int32)
        top_logprobs[rows] = values
    return chosen, top_ids, top_logprobs


def test_compute_logprobs_builds_row_index_on_host(monkeypatch) -> None:
    # A nonzero over a CUDA mask is a host-device sync on every logprobs step; the row
    # index must come from prepare's host list instead, with values unchanged.
    import freetoken.engine.sample as sample_mod

    monkeypatch.setattr(
        sample_mod, "make_device_tensor",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype),
    )
    sampler = Sampler(torch.device("cpu"), vocab_size=6)
    logits = torch.randn(4, 6, generator=torch.Generator().manual_seed(0))
    sampled_tokens = torch.tensor([5, 0, 3, 1], dtype=torch.int32)

    for want in ([True, False, True, False], [True] * 4, [False] * 4):
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(sampling_params=SamplingParams(logprobs=w, top_logprobs=3 if w else 0))
                for w in want
            ]
        )
        args = sampler.prepare(batch)
        expected = _reference_logprobs(logits, sampled_tokens, want, 3)
        with monkeypatch.context() as m:
            m.setattr(torch, "nonzero", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("sync")))
            result = sampler.compute_logprobs(logits, sampled_tokens, args)
        if expected is None:
            assert result is None
            continue
        assert result is not None
        for got, ref in zip(result, expected, strict=True):
            assert got.shape == ref.shape and got.dtype == ref.dtype
            assert torch.equal(got.nan_to_num(-7.0), ref.nan_to_num(-7.0))
