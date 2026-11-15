from freetoken.server.logprobs import completions_logprobs
from freetoken.tokenizer.detokenize import build_logprobs_entry


class _FakeTokenizer:
    def __init__(self, token_table: dict[int, str]) -> None:
        self.token_table = token_table

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.token_table[token_id] for token_id in token_ids)


def test_logprobs_entry_reaches_the_completions_wire() -> None:
    # The tokenizer worker builds the entry and the API process reads it: the keys and the
    # single-id decode (AnyJev's label key) must agree across that process boundary.
    tokenizer = _FakeTokenizer({1: "hel", 2: "lo", 3: "é"})
    entry = build_logprobs_entry(
        tokenizer,
        token_id=3,
        chosen_logprob=-0.123,
        top_ids=[3, 2, 1],
        top_logprobs=[-0.1, -0.9, -1.5],
    )

    assert completions_logprobs([entry]) == {
        "tokens": ["é"],
        "token_logprobs": [-0.123],
        "top_logprobs": [{"é": -0.1, "lo": -0.9, "hel": -1.5}],
        "text_offset": [0],
    }
    assert [c["token_id"] for c in entry["top"]] == [3, 2, 1]
