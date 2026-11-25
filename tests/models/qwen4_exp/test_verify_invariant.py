"""Unit tests for opt-in Issue #948 verify/decode numerical invariance (FREETOKEN_VERIFY_INVARIANT=1)."""

from __future__ import annotations

import math

import torch
from freetoken.models.config import verify_invariant_enabled


def test_verify_invariant_env_toggle(monkeypatch):
    monkeypatch.delenv("FREETOKEN_VERIFY_INVARIANT", raising=False)
    assert not verify_invariant_enabled()
    monkeypatch.setenv("FREETOKEN_VERIFY_INVARIANT", "1")
    assert verify_invariant_enabled()


def test_causal_conv1d_decode_routes_to_triton_when_verify_invariant(monkeypatch):
    import freetoken.kernel.backend as kbackend
    import freetoken.kernel.causal_conv1d as cconv
    import freetoken.kernel.triton.causal_conv1d_triton as triton_cconv

    called = {"triton": 0}

    def fake_triton_decode(x, conv_state, weight, conv_state_indices):
        called["triton"] += 1
        return x + 1.0

    monkeypatch.setattr(kbackend, "is_sgl_kernel_installed", lambda: True)
    monkeypatch.setattr(triton_cconv, "causal_conv1d_decode", fake_triton_decode)
    monkeypatch.setenv("FREETOKEN_VERIFY_INVARIANT", "1")

    x = torch.zeros((2, 16), dtype=torch.bfloat16)
    st = torch.zeros((4, 16, 3), dtype=torch.bfloat16)
    w = torch.zeros((16, 4), dtype=torch.bfloat16)
    idx = torch.tensor([0, 1], dtype=torch.int32)
    out = cconv.causal_conv1d_decode(x, st, w, idx)
    assert called["triton"] == 1
    assert torch.all(out == 1.0)


def test_qsa_attend_uses_batch_invariant_splits_under_verify_invariant(monkeypatch):
    import freetoken.kernel.triton.qsa.attend as qsa_attend

    launched_grids: list[tuple[int, int, int]] = []

    class _DummyKernel:
        def __getitem__(self, grid):
            launched_grids.append(tuple(grid))

            def _launch(*_args, **_kwargs):
                return None

            return _launch

    monkeypatch.setattr(
        qsa_attend, "_qsa_sparse_paged_gqa_splitk_kernel", _DummyKernel()
    )
    monkeypatch.setattr(qsa_attend, "_qsa_merge_splitk_kernel", _DummyKernel())
    monkeypatch.setenv("FREETOKEN_VERIFY_INVARIANT", "1")

    k_cache = torch.zeros((8, 64, 2, 128), dtype=torch.bfloat16)
    v_cache = torch.zeros((8, 64, 2, 128), dtype=torch.bfloat16)
    block_table = torch.zeros((8, 4), dtype=torch.int32)

    for m in (1, 4, 16, 32):
        launched_grids.clear()
        q = torch.zeros((m, 16, 128), dtype=torch.bfloat16)
        logical_indices = torch.zeros((m, 128), dtype=torch.int32)
        token_to_req = torch.zeros((m,), dtype=torch.int32)
        qsa_attend.qsa_sparse_paged_attention(
            q, k_cache, v_cache, logical_indices, block_table, token_to_req
        )
        # First launch is split-K partial grid: (M, num_kv_heads=2, num_splits=8) for ALL M in 1..64!
        assert launched_grids[0] == (m, 2, 8), (m, launched_grids)


def test_ple_fp32_gate_matches_across_batch_views():
    torch.manual_seed(11)
    hc_count, hidden = 4, 2560
    key = torch.randn((8, hc_count * hidden), dtype=torch.bfloat16)
    query = torch.randn((8, hc_count * hidden), dtype=torch.bfloat16)

    def fp32_gate(k_t: torch.Tensor, q_t: torch.Tensor) -> torch.Tensor:
        shape = (-1, hc_count, hidden)
        gf = (k_t.float().view(shape) * q_t.float().view(shape)).sum(
            -1, keepdim=True
        ) / math.sqrt(hidden)
        return torch.sigmoid(gf.sign() * gf.abs().clamp_min(1e-6).sqrt()).to(
            torch.bfloat16
        )

    batch_out = fp32_gate(key, query)
    for row in range(8):
        single_out = fp32_gate(key[row : row + 1], query[row : row + 1])
        assert torch.equal(batch_out[row : row + 1], single_out)
