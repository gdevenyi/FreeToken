import os

import pytest

from freetoken.tokenizer.detokenize import build_logprobs_entry


class _FakeTokenizer:
    def __init__(self, token_table: dict[int, str]) -> None:
        self.token_table = token_table

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.token_table[token_id] for token_id in token_ids)


def test_build_logprobs_entry_matches_contract() -> None:
    tokenizer = _FakeTokenizer({1: "hel", 2: "lo", 3: "é"})
    entry = build_logprobs_entry(
        tokenizer,
        token_id=3,
        chosen_logprob=-0.123,
        top_ids=[3, 2, 1],
        top_logprobs=[-0.1, -0.9, -1.5],
    )

    assert entry["token_id"] == 3
    assert entry["token"] == "é"
    assert entry["bytes"] == list("é".encode("utf-8"))
    assert entry["logprob"] == -0.123

    top = entry["top"]
    assert len(top) == 3
    assert top[0] == {
        "token_id": 3,
        "token": "é",
        "bytes": list("é".encode("utf-8")),
        "logprob": -0.1,
    }
    assert top[1] == {
        "token_id": 2,
        "token": "lo",
        "bytes": list("lo".encode("utf-8")),
        "logprob": -0.9,
    }
    assert top[2] == {
        "token_id": 1,
        "token": "hel",
        "bytes": list("hel".encode("utf-8")),
        "logprob": -1.5,
    }


def test_build_logprobs_entry_empty_top() -> None:
    tokenizer = _FakeTokenizer({7: "x", 8: "y"})
    entry = build_logprobs_entry(
        tokenizer,
        token_id=7,
        chosen_logprob=-2.0,
        top_ids=[],
        top_logprobs=[],
    )

    assert entry["top"] == []


def _byte_level_tokenizer():
    """A tiny GPT-2-style ByteLevel BPE tokenizer built in memory (no download): the 256
    byte stand-ins, one merge ("ðŁ", the first two bytes of most emoji) and one added
    token whose content the byte table cannot invert."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["ðŁ"] = len(vocab)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[("ð", "Ł")]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_tokens([AddedToken("<｜end｜>", special=False)])
    return PreTrainedTokenizerFast(tokenizer_object=tk)


def test_logprobs_bytes_rebuild_a_split_character() -> None:
    # 🦀 is three byte-level tokens that each decode to U+FFFD; their bytes must still
    # concatenate back to the character.
    tokenizer = _byte_level_tokenizer()
    ids = tokenizer.encode("🦀", add_special_tokens=False)
    assert len(ids) == 3
    entries = [build_logprobs_entry(tokenizer, i, -0.5, [i], [-0.5]) for i in ids]
    assert all(e["token"] == "�" for e in entries)
    assert bytes(b for e in entries for b in e["bytes"]) == "🦀".encode("utf-8")
    assert [e["top"][0]["bytes"] for e in entries] == [e["bytes"] for e in entries]


def test_logprobs_bytes_added_and_padded_ids() -> None:
    tokenizer = _byte_level_tokenizer()
    added = tokenizer.convert_tokens_to_ids("<｜end｜>")
    padded = len(tokenizer) + 10  # the logits' padded tail: no vocab entry at all
    entry = build_logprobs_entry(tokenizer, added, -0.1, [added, padded], [-0.1, -9.0])
    assert entry["token"] == "<｜end｜>"
    assert entry["bytes"] == list("<｜end｜>".encode("utf-8"))
    assert entry["top"][1] == {"token_id": padded, "token": "", "bytes": [], "logprob": -9.0}


def test_logprobs_bytes_real_qwen_tokenizer() -> None:
    path = os.path.expanduser("~/models/Qwen3.8-Flash-Next-NVFP4-FTW-fp8")
    if not os.path.exists(os.path.join(path, "tokenizer.json")):
        pytest.skip("local Qwen3.8 tokenizer not present")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    entries = [build_logprobs_entry(tokenizer, i, -0.1, [248300], [-9.0]) for i in (9008, 99, 222)]
    assert bytes(b for e in entries for b in e["bytes"]) == "🦀".encode("utf-8")
    assert entries[0]["top"][0]["bytes"] == []
