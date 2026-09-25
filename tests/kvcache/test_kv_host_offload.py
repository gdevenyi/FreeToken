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
