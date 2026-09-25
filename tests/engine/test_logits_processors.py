"""The logits processors behind min_p, the penalties, logit_bias and min_tokens: pure
torch on the CPU, so the math is checked without a GPU or the sampling kernels."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from freetoken.core import SamplingParams
from freetoken.engine.sample import LogitsPlan, Sampler, apply_logits_processors

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
    plan = _plan(
        [0],
        hist_ids=_ids([[0, 1]]),  # generated: 0 (positive), 1 (negative)
        prompt_ids=_ids([[2, 3]]),  # prompt: 2 (positive), 3 (negative)
        repetition=torch.tensor([[2.0]]),
    )
    out = apply_logits_processors(logits, plan, V)
    assert out[0, 0].item() == 1.0 and out[0, 2].item() == 1.0  # divided
    assert out[0, 1].item() == -4.0 and out[0, 3].item() == -4.0  # multiplied
    assert out[0, 4].item() == 1.0  # unseen


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
        output_len=max_tokens,
        max_device_len=len(prompt) + max_tokens,
        sampling_params=SamplingParams(**sp),
        can_decode=True,
        output_token_counts=None,
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
    assert plan.hist_ids[0].tolist()[:2] == [3, 3]  # row 1's generated tokens
    assert plan.prompt_ids[0].tolist()[:2] == [1, 2]
    assert plan.bias_rows.tolist() == [0] and plan.bias_ids.tolist() == [5]
    assert plan.min_rows.tolist() == [1]  # local row of the third request only
    assert plan.min_ids[0].tolist() == [0, 7]

    logits = torch.zeros((4, V))
    out = apply_logits_processors(logits, plan, V)
    assert out[1, 3].item() == 0.0  # repetition on 0 is a no-op; presence is not plan work
    assert out[1, 5].item() == 2.0
    assert out[2, 0].item() == float("-inf") and out[2, 7].item() == float("-inf")
    assert torch.isfinite(out[3]).all()
    # greedy path: argmax over the processed logits
    assert sampler.sample(logits, args)[1].item() == 5


def test_presence_and_frequency_ride_the_device_counts_not_the_plan():
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
