"""CPU-only: with the small prefill path on, warmup still reaches every whole-layer prefill bucket."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from freetoken.engine.engine import Engine
from freetoken.kernel import nvfp4_moe
from freetoken.layers import moe as moe_layer


@pytest.fixture
def sm89(monkeypatch):
    tuning = nvfp4_moe._PREFILL_TUNING[("pro", (8, 9))]
    monkeypatch.setattr(nvfp4_moe, "_prefill_tuning", lambda device_index: tuning)
    monkeypatch.setattr(nvfp4_moe, "_prefill_block_m_steps", lambda device_index: tuning.steps)


def _lens(monkeypatch, threshold: int, cap: int = 8192) -> list[int]:
    monkeypatch.setattr(moe_layer, "_SMALL_PREFILL_TOKENS", threshold)
    engine = SimpleNamespace(
        max_seq_len=cap,
        device=SimpleNamespace(index=0),
        config=SimpleNamespace(
            max_forward_len=cap,
            model_config=SimpleNamespace(
                is_moe=True, num_experts_per_tok=10, num_experts=512, has_linear_attention=True
            ),
        ),
    )
    return Engine._warmup_prefill_lens(engine)


def test_buckets_are_those_of_prefill_config(sm89):
    lens = nvfp4_moe.prefill_bucket_lens(257, 8192, 10, 512, 0)
    assert lens == [257, 820, 1844, 2458, 8192]
    configs = [nvfp4_moe.prefill_config(M, 10, 512, 0) for M in range(257, 8193)]
    distinct = {tuple(sorted(cfg.items())) for cfg in configs}
    assert {tuple(sorted(nvfp4_moe.prefill_config(M, 10, 512, 0).items())) for M in lens} == distinct


def test_small_prefill_adds_one_length_per_whole_layer_bucket(monkeypatch, sm89):
    without = set(_lens(monkeypatch, 0))
    with_small = set(_lens(monkeypatch, 256))
    assert with_small == without | {257, 820, 1844, 2458, 8192}


def test_warmup_cap_bounds_the_added_lengths(monkeypatch, sm89):
    assert max(_lens(monkeypatch, 256, cap=1000)) == 1000
    assert set(_lens(monkeypatch, 256, cap=1000)) == set(_lens(monkeypatch, 0, cap=1000)) | {257, 1000}


def test_threshold_at_the_cap_adds_nothing(monkeypatch, sm89):
    assert _lens(monkeypatch, 8192) == _lens(monkeypatch, 0)
