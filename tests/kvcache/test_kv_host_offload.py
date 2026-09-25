"""--kv-host-pages prefill grouping (upstream #499): each group's page list must reach the GPU intact."""

import pytest
import torch

from freetoken.kvcache.kv_host_offload import KVHostOffloader

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _offloader(num_slots: int, num_logical: int) -> KVHostOffloader:
    off = KVHostOffloader.__new__(KVHostOffloader)
    off.device = torch.device("cuda")
    off.num_slots = num_slots
    off.num_logical = num_logical
    off.max_sel_pages = 4
    off._sel_host = off._counts_host = off._query_host = off._query_buf = off._eff_buf = None
    off.phys_of = torch.zeros(num_logical + 1, dtype=torch.int32, device="cuda")
    return off


def test_each_prefill_group_uploads_its_own_page_list_while_the_stream_is_busy():
    """The caller's attends back the stream up between groups; a group's queued H2D copy must
    still read that group's pages, not a later group's rewrite of shared staging."""
    rows, pages_per_row = 6, 4
    off = _offloader(num_slots=13, num_logical=64)  # cap = 13 - 0 - 9 = 4 -> one row per group
    sel = torch.arange(rows * pages_per_row, dtype=torch.int32, device="cuda").view(rows, pages_per_row)
    off._counts_dev = torch.full((rows,), pages_per_row, dtype=torch.int32, device="cuda")
    received = []
    off._ensure = lambda q: received.append(q.clone())  # a stream-ordered read of what the copy delivered
    block_table = torch.zeros(1, 8, dtype=torch.int32, device="cuda")

    groups = []
    for span, _eff in off.iter_prefill_groups(sel, torch.empty(0, dtype=torch.int64), block_table):
        groups.append(span)
        torch.cuda._sleep(20_000_000)  # a slow attend: later CPU writes race the queued copies
    torch.cuda.synchronize()

    assert groups == [(r, r + 1) for r in range(rows)]
    for r, got in enumerate(received):
        assert sorted(got.tolist()) == sel[r].tolist(), (r, got.tolist())


@pytest.mark.parametrize("atomics", [False, True])
def test_compacted_selection_flags_rows_that_overflow_max_sel_pages(atomics, monkeypatch):
    """Below sm_70 the dropped-page counter is a plain store, since Triton lowers atomics to
    sm_70+ PTX; a racing row may win, but the counter is non-zero iff some row dropped."""
    import freetoken.kernel.triton.qsa.offload as offload_kernels

    if atomics and torch.cuda.get_device_capability() < (7, 0):
        pytest.skip("atomics need sm_70")
    monkeypatch.setattr(offload_kernels, "device_capability", lambda: (7, 0) if atomics else (6, 1))
    off = _offloader(num_slots=13, num_logical=64)
    off.max_sel_pages, off.page_size, off.dummy_page = 2, 4, 64
    off._sel_all = None
    off._trunc_eager = torch.zeros(1, dtype=torch.int32, device="cuda")
    off._g = {}
    block_table = (torch.arange(8, dtype=torch.int32, device="cuda") + 10).view(1, 8)
    token_to_req = torch.zeros(4, dtype=torch.int32, device="cuda")

    def tokens(*rows):
        return torch.tensor(rows, dtype=torch.int32, device="cuda")

    indices = tokens(
        [0, 1, 2, -1, -1, -1],      # one page
        [0, 4, 8, 12, 16, 20],      # six pages: keeps 2, drops 4
        [4, 5, 8, 12, -1, -1],      # three pages: keeps 2, drops 1
        [-1, -1, -1, -1, -1, -1],   # nothing selected
    )
    sel = off.compact_all(indices, token_to_req, block_table)
    assert sel.tolist() == [[10, 10], [10, 11], [11, 12], [64, 64]]
    assert off._counts_dev[:4].tolist() == [1, 2, 2, 0]
    dropped = off.trunc_count()
    assert dropped == 4 if atomics else dropped in (4, 1)
    assert off.trunc_count() == 0  # read resets

    off.compact_all(indices[[0, 3]], token_to_req[:2], block_table)
    assert off.trunc_count() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("maxp", [8, 72])
def test_compacted_selection_matches_a_reference_on_scattered_selections(maxp):
    """Unsorted, duplicate-heavy selections over a wide table: the distinct pages in column
    order (the first maxp of them), the stored counts, and a mark scratch left zeroed."""
    from freetoken.kernel.triton.qsa.offload import compact_selected_pages

    g = torch.Generator().manual_seed(maxp)
    rows, sel, width, page = 5, 2048, 4097, 64
    block_table = torch.randperm(10 * width, generator=g)[: 2 * width].view(2, width).int().cuda()
    token_to_req = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32, device="cuda")
    runs = torch.randint(0, width * page - 8, (rows, sel // 8), generator=g)
    indices = (runs[:, :, None] + torch.arange(8)).view(rows, sel)
    indices[3, 100:] = -1  # a short selection
    indices[4] = -1  # nothing selected
    indices = indices.int().cuda()
    out = torch.full((rows, maxp), -7, dtype=torch.int32, device="cuda")
    counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    marks = torch.zeros((rows, width), dtype=torch.int8, device="cuda")
    compact_selected_pages(indices, token_to_req, block_table, out, page, 99, counts=counts, marks=marks)
    for r in range(rows):
        toks = indices[r][indices[r] >= 0].cpu()
        cols = sorted(set((toks // page).tolist()))
        want = [int(block_table[token_to_req[r], c]) for c in cols][:maxp]
        pad = want[0] if want else 99
        assert out[r].tolist() == want + [pad] * (maxp - len(want)), r
        assert int(counts[r]) == len(want)
    assert int(marks.abs().sum()) == 0
