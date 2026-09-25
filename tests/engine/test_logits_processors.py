"""The logits processors behind min_p, the penalties, logit_bias and min_tokens: pure
torch on the CPU, so the math is checked without a GPU or the sampling kernels."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.core import SamplingParams
from freetoken.engine.sample import (
    LogitsPlan,
    RowPenalty,
    Sampler,
    apply_logits_processors,
    apply_penalties,
)

V = 8
CPU = torch.device("cpu")


def _plan(rows, **fields) -> LogitsPlan:
    plan = LogitsPlan(
        rows=torch.tensor(rows, dtype=torch.int64),
        temps=torch.ones((len(rows), 1), dtype=torch.float32),
    )
    for k, v in fields.items():
        setattr(plan, k, v)
    return plan


def _ids(rows: list[list[int]]) -> torch.Tensor:
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), V, dtype=torch.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r)
    return out


def test_repetition_penalty_covers_prompt_and_output_with_hf_semantics():
    logits = torch.tensor([[2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0, 1.0]])
    counts = torch.zeros(V, dtype=torch.int32)
    counts[[0, 1]] = 1  # generated: 0 (positive), 1 (negative)
    prompt = torch.zeros(V, dtype=torch.bool)
    prompt[[2, 3]] = True  # prompt: 2 (positive), 3 (negative)
    out = apply_penalties(logits, [RowPenalty(0, counts, 0.0, 0.0, 2.0, prompt)])
    assert out[0, 0].item() == 1.0 and out[0, 2].item() == 1.0  # divided
    assert out[0, 1].item() == -4.0 and out[0, 3].item() == -4.0  # multiplied
    assert out[0, 4].item() == 1.0  # unseen
    assert logits[0, 0].item() == 2.0  # the caller's logits stay untouched


def test_logit_bias_adds_per_row():
    logits = torch.zeros((2, V))
    plan = _plan(
        [0, 1],
        bias_rows=torch.tensor([0, 1, 1]),
        bias_ids=torch.tensor([2, 2, 6]),
        bias_vals=torch.tensor([5.0, -100.0, 1.5]),
    )
    out = apply_logits_processors(logits, plan, V)
    assert (
        out[0, 2].item() == 5.0
        and out[1, 2].item() == -100.0
        and out[1, 6].item() == 1.5
    )


def test_min_tokens_masks_the_stop_ids_of_the_row_only():
    logits = torch.zeros((2, V))
    plan = _plan([0, 1], min_rows=torch.tensor([1]), min_ids=_ids([[0, 7]]))
    out = apply_logits_processors(logits, plan, V)
    assert out[1, 0].item() == float("-inf") and out[1, 7].item() == float("-inf")
    assert out[1, 3].item() == 0.0
    assert torch.isfinite(out[0]).all()


@pytest.mark.parametrize("overlap", [True, False])
def test_min_tokens_releases_the_stop_ids_at_output_index_min_tokens(overlap):
    """Overlap scheduling prepares batch N before batch N-1's token reaches req.input_ids on
    the host; the count must not lag by that token and hold EOS one step too long."""
    from freetoken.core import Req

    sp = SamplingParams(min_tokens=3, max_tokens=10)
    sp.min_tokens_stop_ids = [0]
    req = Req(input_ids=torch.tensor([5, 6], dtype=torch.int32), table_idx=0, cached_len=0,
              output_len=10, uid=1, sampling_params=sp, cache_handle=None)
    sampler = Sampler(CPU, V)
    masked, pending = [], None
    for _ in range(6):
        plan = sampler.prepare(SimpleNamespace(reqs=[req])).plan
        masked.append(plan is not None and plan.min_rows is not None)
        req.complete_one()  # the forward launch advances the device length
        if overlap:
            if pending is not None:  # batch N-1 drains after batch N launched
                req.append_host(pending)
            pending = torch.tensor([7], dtype=torch.int32)
        else:
            req.append_host(torch.tensor([7], dtype=torch.int32))
    assert [i for i, m in enumerate(masked) if m] == [0, 1, 2]


def test_min_p_drops_tokens_below_the_fraction_of_the_top_probability():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005]]))
    plan = _plan(
        [0], min_p=torch.tensor([[0.15]])
    )  # threshold 0.075, between 0.1 and 0.05
    out = apply_logits_processors(logits, plan, V)
    kept = torch.isfinite(out[0])
    assert kept.tolist() == [True, True, True, False, False, False, False, False]
    # a row with min_p 0 is left alone
    plan0 = _plan([0], min_p=torch.tensor([[0.0]]))
    assert torch.isfinite(apply_logits_processors(logits, plan0, V)[0]).all()


def _req(
    prompt: list[int], generated: list[int], max_tokens: int, **sp
) -> SimpleNamespace:
    ids = torch.tensor(prompt + generated, dtype=torch.int32)
    return SimpleNamespace(
        input_ids=ids,
        device_len=len(ids),
        output_len=max_tokens,
        max_device_len=len(prompt) + max_tokens,
        sampling_params=SamplingParams(**sp),
        can_decode=True,
        output_token_counts=None,
        prompt_token_mask=None,
    )


def test_prepare_builds_no_plan_without_processors_and_a_plan_with_them():
    sampler = Sampler(CPU, V)
    plain = SimpleNamespace(reqs=[_req([1, 2], [3], 4)])
    assert sampler.prepare(plain).plan is None

    # all greedy: the argmax path runs on the CPU, the sampling kernels need a GPU
    batch = SimpleNamespace(
        reqs=[
            _req([1, 2], [3], 4),
            _req(
                [1, 2],
                [3, 3],
                4,
                presence_penalty=1.0,
                repetition_penalty=1.5,
                logit_bias=[[5, 2.0]],
            ),
            _req([4], [], 4, min_tokens=2, min_tokens_stop_ids=[0, 7]),
            _req(
                [4], [6, 6], 4, min_tokens=2, min_tokens_stop_ids=[0]
            ),  # already past min_tokens
        ]
    )
    args = sampler.prepare(batch)
    plan = args.plan
    assert plan is not None
    assert plan.rows.tolist() == [1, 2, 3]
    assert [pen.row for pen in args.penalties] == [1]
    assert batch.reqs[1].prompt_token_mask.nonzero().flatten().tolist() == [1, 2]
    assert plan.bias_rows.tolist() == [0] and plan.bias_ids.tolist() == [5]
    assert plan.min_rows.tolist() == [1]  # local row of the third request only
    assert plan.min_ids[0].tolist() == [0, 7]

    logits = torch.zeros((4, V))
    out = apply_logits_processors(logits, plan, V)
    assert out[1, 3].item() == 0.0  # the penalties are not plan work
    assert out[1, 5].item() == 2.0
    assert out[2, 0].item() == float("-inf") and out[2, 7].item() == float("-inf")
    assert torch.isfinite(out[3]).all()
    # greedy path: argmax over the processed logits
    assert sampler.sample(logits, args)[1].item() == 5


def test_penalties_ride_the_device_state_not_the_plan():
    # the plan must not apply them a second time on top of the sampler's own counts
    sampler = Sampler(CPU, V)
    only = SimpleNamespace(reqs=[_req([1], [], 4, presence_penalty=1.0, frequency_penalty=0.5)])
    args = sampler.prepare(only)
    assert args.plan is None and [row for row, *_ in args.penalties] == [0]

    req = _req([1], [], 4, presence_penalty=1.0, logit_bias=[[5, 0.25]])
    args = sampler.prepare(SimpleNamespace(reqs=[req]))
    assert args.plan is not None and [row for row, *_ in args.penalties] == [0]
    req.output_token_counts[3] = 1
    logits = torch.zeros((1, V))
    logits[0, 3] = 1.5
    # once: 3 -> 0.5 beats 5 -> 0.25; twice would drop 3 to -0.5 and pick 5
    assert sampler.sample(logits, args).item() == 3
    assert req.output_token_counts[3].item() == 2


def test_repetition_penalty_ignores_multimodal_placeholder_ids_in_the_prompt():
    # image tokens carry pseudo-ids >= MM_PAD_SHIFT_VALUE; scattering them into the
    # [rows, V+1] prompt mask indexed out of bounds (a device-side assert on CUDA)
    sampler = Sampler(CPU, V)
    batch = SimpleNamespace(reqs=[_req([1, 1_000_000, 2], [3], 4, repetition_penalty=2.0)])
    args = sampler.prepare(batch)
    out = apply_penalties(torch.ones(1, V), args.penalties)
    assert out[0, 1] == 0.5 and out[0, 2] == 0.5 and out[0, 4] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_min_p_with_temperature_draws_only_the_kept_tokens():
    # min_p leaves -inf over nearly the whole row; the sampling softmax must not turn that
    # into NaN (which the Pascal fallback then samples as a uniform row)
    V = 32768
    sampler = Sampler(torch.device("cuda"), V)
    req = _req([1, 2], [], 4, temperature=0.8, min_p=0.1)
    args = sampler.prepare(SimpleNamespace(reqs=[req]))
    logits = torch.full((1, V), -10.0, device="cuda")
    kept = [100, 20000, 31000]
    logits[0, kept] = torch.tensor([5.0, 4.5, 4.0], device="cuda")
    for _ in range(32):
        assert sampler.sample(logits, args).item() in kept


def test_repetition_penalty_reads_the_prompt_once_and_counts_output_on_the_device():
    # the prompt (up to the whole context) is uploaded once per request, not every step,
    # and the generated side comes from the counts sample() keeps, not the host history
    sampler = Sampler(CPU, V)
    req = _req([1, 2], [], 4, repetition_penalty=2.0)
    batch = SimpleNamespace(reqs=[req])
    args = sampler.prepare(batch)
    mask = req.prompt_token_mask
    logits = torch.tensor([[0.0, 1.0, 1.0, 4.0, 0.5, 0.0, 0.0, 0.0]])
    assert sampler.sample(logits, args).item() == 3
    assert req.output_token_counts.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    req.input_ids = torch.tensor([9, 9, 3], dtype=torch.int32)  # a stale host view is not read
    args = sampler.prepare(batch)
    assert req.prompt_token_mask is mask
    out = apply_penalties(logits, args.penalties)
    assert out[0].tolist() == [0.0, 0.5, 0.5, 2.0, 0.5, 0.0, 0.0, 0.0]
