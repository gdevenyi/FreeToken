import torch
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
        logprob_rows=torch.tensor([True, False, True], dtype=torch.bool),
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
    assert torch.equal(top_ids[0], expected_top0.indices.to(torch.int32))
    assert torch.allclose(top_logprobs[0], expected_top0.values)
    assert torch.equal(top_ids[2], expected_top2.indices.to(torch.int32))
    assert torch.allclose(top_logprobs[2], expected_top2.values)

    assert torch.equal(top_ids[1], torch.full((3,), -1, dtype=torch.int32))
    assert torch.isneginf(top_logprobs[1]).all()


def test_compute_logprobs_ignores_temperatures() -> None:
    # OpenAI and vLLM report the model's raw logprobs, whatever temperature sampled the token.
    sampler = Sampler(torch.device("cpu"), vocab_size=4)

    logits = torch.tensor(
        [
            [0.5, 1.0, 2.0, 3.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([1, 2], dtype=torch.long)
    logprob_rows = torch.tensor([True, True], dtype=torch.bool)

    for temperature in (0.3, 2.5):
        args = BatchSamplingArgs(
            temperatures=torch.full((2,), temperature, dtype=torch.float32),
            logprob_rows=logprob_rows,
            max_top_logprobs=2,
        )
        chosen, top_ids, top_logprobs = sampler.compute_logprobs(
            logits, sampled_tokens, args
        )
        expected = torch.log_softmax(logits, dim=-1)
        assert torch.allclose(chosen, expected[[0, 1], sampled_tokens])
        top = torch.topk(expected, k=2)
        assert torch.equal(top_ids, top.indices.to(torch.int32))
        assert torch.allclose(top_logprobs, top.values)
