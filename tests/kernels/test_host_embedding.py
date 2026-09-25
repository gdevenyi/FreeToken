"""--embed-weights host: the UVA row gather from pinned host RAM must equal the GPU table lookup."""

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _embedding(dtype: torch.dtype, rows: int = 1000, dim: int = 2560, device: str = "cuda"):
    from freetoken.layers.embedding import VocabParallelEmbedding

    torch.manual_seed(0)
    emb = VocabParallelEmbedding(rows, dim)
    emb.weight = torch.randn(rows, dim, device=device, dtype=dtype)
    return emb


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_host_rows_equal_the_gpu_lookup(dtype):
    emb = _embedding(dtype)
    ids = torch.randint(0, 1000, (37,), device="cuda", dtype=torch.int32)
    want = emb.forward(ids)
    emb.place_on_host()
    assert not emb.weight.is_cuda and emb.weight.is_pinned()
    got = emb.forward(ids)
    assert got.is_cuda and got.dtype is dtype
    assert torch.equal(got, want)


def test_a_table_the_gather_cannot_read_is_refused_before_pinning():
    emb = _embedding(torch.float64, rows=4, dim=8, device="cpu")
    with pytest.raises(ValueError, match="cannot gather a torch.float64 table"):
        emb.place_on_host()
    assert emb._host_ptr is None


@needs_cuda
def test_host_lookup_replays_in_a_cuda_graph_with_new_ids():
    emb = _embedding(torch.bfloat16)
    table = emb.weight.clone()
    emb.place_on_host()
    ids = torch.zeros(8, device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        emb.forward(ids)  # compile outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = emb.forward(ids)
    for seed in range(3):
        ids.copy_(torch.randint(0, 1000, (8,), generator=torch.Generator().manual_seed(seed)))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, table[ids])
