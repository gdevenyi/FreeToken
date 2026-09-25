"""The QSA sparse attend's shared-memory retry ladder (kernel/triton/qsa/attend.py).

Triton's operand staging can exceed the launcher's byte estimate, so a launch that raises
``OutOfResources`` is retried on a half-width column tile down to the ``tl.dot`` minimum
of 16. No GPU in CI reaches that path, so the kernels are replaced by fakes that refuse
tiles above a floor and record every launch. The invariants: each retry halves BLOCK_N and
never goes below 16, the split count and partial buffers are recomputed for the tile that
actually launched (the merge must see those, not the ones of a refused attempt), and a
launch that cannot fit at 16 re-raises instead of looping.
"""

from __future__ import annotations

import pytest
import torch
import triton
from triton.runtime.errors import OutOfResources

import freetoken.kernel.triton.qsa.attend as attend

HQ, KVH, D, PAGE = 16, 2, 128, 64


class _FakeKernel:
    def __init__(self, calls: list, floor: int | None = None):
        self.calls = calls
        self.floor = floor

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            # a ladder that never stops would hang pytest; fail fast instead
            assert len(self.calls) < 8, "retry ladder did not terminate"
            self.calls.append((grid, args, kwargs))
            if self.floor is not None and kwargs["BLOCK_N"] > self.floor:
                raise OutOfResources(99999, 49152, "shared memory")

        return launch


def _run(monkeypatch, capability, smem, floor, rows, topk):
    partials, merges = [], []
    monkeypatch.setattr(attend, "device_capability", lambda: capability)
    monkeypatch.setattr(attend, "_optin_smem_bytes", lambda index: smem)
    monkeypatch.setattr(attend, "_qsa_sparse_paged_gqa_splitk_kernel", _FakeKernel(partials, floor))
    monkeypatch.setattr(attend, "_qsa_merge_splitk_kernel", _FakeKernel(merges))
    # CPU tensors have no device index, so the launcher asks torch.cuda for one
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    q = torch.zeros(rows, HQ, D, dtype=torch.bfloat16)
    k_cache = torch.zeros(8, PAGE, KVH, D, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    args = (
        q,
        k_cache,
        k_cache.clone(),
        torch.zeros(rows, topk, dtype=torch.int32),
        torch.zeros(1, 8, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
    )
    return partials, merges, out, args


@pytest.mark.parametrize(
    "capability, smem, floor, rows, topk, block_ns, num_splits",
    [
        # mid profile: 64 -> 32 -> 16 at a constant 8 splits
        ((8, 9), 101376, 16, 32, 2048, [64, 32, 16], 8),
        # the split count grows per rung (1 -> 2 -> 4), so a refused attempt's buffers differ
        ((8, 9), 101376, 16, 32, 64, [64, 32, 16], 4),
        # prefill profile: one split writes straight into out and skips the merge
        ((8, 9), 101376, 32, 1024, 2048, [64, 32], 1),
        # pre-Volta pins the 16-wide tile, so there is nothing to retry
        ((6, 1), 49152, 16, 32, 2048, [16], 8),
    ],
)
def test_retry_ladder_launches_the_first_fitting_tile(
    monkeypatch, capability, smem, floor, rows, topk, block_ns, num_splits
):
    partials, merges, out, args = _run(monkeypatch, capability, smem, floor, rows, topk)

    assert attend.qsa_sparse_paged_attention(*args, out=out) is out

    assert [kwargs["BLOCK_N"] for _, _, kwargs in partials] == block_ns
    for grid, launch_args, kwargs in partials:
        assert kwargs["NUM_TILES"] == triton.cdiv(topk, kwargs["BLOCK_N"])
        assert grid[2] == kwargs["NUM_SPLITS"]
        if kwargs["NUM_SPLITS"] > 1:
            assert launch_args[10].shape == (kwargs["NUM_SPLITS"], rows, HQ, D)
        else:
            assert launch_args[10] is out and launch_args[11] is out

    _, launch_args, kwargs = partials[-1]
    assert kwargs["NUM_SPLITS"] == num_splits
    if num_splits == 1:
        assert merges == []
        return
    assert len(merges) == 1
    _, merge_args, merge_kwargs = merges[0]
    assert merge_args[0] is launch_args[10] and merge_args[1] is launch_args[11]
    assert merge_kwargs["NUM_SPLITS"] == num_splits == merge_args[0].shape[0]
    assert merge_kwargs["BLOCK_SPLITS"] == triton.next_power_of_2(num_splits)


@pytest.mark.parametrize(
    "capability, smem, block_ns",
    [((8, 9), 101376, [64, 32, 16]), ((6, 1), 49152, [16])],
)
def test_retry_ladder_reraises_when_no_tile_fits(monkeypatch, capability, smem, block_ns):
    partials, merges, out, args = _run(monkeypatch, capability, smem, 8, 32, 2048)

    with pytest.raises(OutOfResources):
        attend.qsa_sparse_paged_attention(*args, out=out)

    assert [kwargs["BLOCK_N"] for _, _, kwargs in partials] == block_ns
    assert merges == []
