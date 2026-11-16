"""The QSA backend behind the real Qwen4ExpAttention layer.

(a) dense-oracle equivalence -- while a request sees at most ``index_budget + index_ratio - 1``
    tokens every complete block is selected, so QSA IS dense attention: the selection must be
    exactly the causal prefix and the layer output must match ``TorchDenseQSAReference`` (fp32)
    and a flashinfer dense run over the same pool;
(b) chunked prefill at unaligned cut points matches one-shot prefill to bf16 tolerance (the
    dual-source compress);
(c) a captured decode or verify replay equals the eager step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import Fixture, requires_cuda, parsed_config, selection_spy

QSA_LAYER = 3


def _inputs(fixture: Fixture, lengths, extra: int = 0, seed: int = 11):
    generator = torch.Generator(device=fixture.device).manual_seed(seed)
    return [
        torch.randn(
            n + extra, fixture.config.hidden_size, device=fixture.device,
            dtype=fixture.dtype, generator=generator,
        )
        * 0.5
        for n in lengths
    ]


def _assert_selection_is_causal_prefix(indices: torch.Tensor, positions: torch.Tensor) -> None:
    for row, position in enumerate(positions.tolist()):
        selected = indices[row][indices[row] >= 0]
        assert torch.equal(
            selected.sort().values,
            torch.arange(position + 1, dtype=selected.dtype, device=selected.device),
        ), f"row {row} (position {position}) did not select its whole causal prefix"


@requires_cuda
def test_prefill_is_dense_below_the_budget(monkeypatch):
    """bs=3 ragged prefill, longest request exactly at budget + ratio - 1."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths = [2051, 1000, 137]
    inputs = _inputs(fixture, lengths)
    x = torch.cat([row[:n] for row, n in zip(inputs, lengths)])
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    seen = selection_spy(monkeypatch, fixture.backend)
    batch = fixture.batch(reqs, "prefill")
    got = attn.forward(x, batch)
    _assert_selection_is_causal_prefix(seen["indices"], batch.positions)

    fixture.ctx.attn_backend = _dense_oracle(fixture)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


def _dense_oracle(fixture: Fixture):
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    return TorchDenseQSAReference(
        fixture.config,
        num_slots=fixture.num_req_slots,
        max_len=4096,
        device=fixture.device,
        dtype=fixture.dtype,
    )


@requires_cuda
def test_decode_is_dense_below_the_budget(monkeypatch):
    """Prefill then five decode steps, sparse path vs the fp32 dense oracle."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411, 64], 5
    inputs = _inputs(fixture, lengths, extra=steps)
    oracle = _dense_oracle(fixture)

    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    seen = selection_spy(monkeypatch, fixture.backend)

    steps_x = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    steps_x += [
        torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)
    ]
    for step, x in enumerate(steps_x):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        fixture.ctx.attn_backend = fixture.backend
        got = attn.forward(x, batch)
        _assert_selection_is_causal_prefix(seen["indices"], batch.positions)
        fixture.ctx.attn_backend = oracle
        reference = attn.forward(x, batch)
        torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flashinfer_dense_matches_the_sparse_path():
    """The engine's dense FULL backend over the same pool, as an independent oracle."""
    pytest.importorskip("flashinfer")
    from freetoken.attention.fi import FlashInferBackend

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 500
    x = _inputs(fixture, [length])[0]
    req = fixture.req(0, 0, length)
    got = attn.forward(x, fixture.batch([req], "prefill"))

    dense = FlashInferBackend(config)
    fixture.ctx.attn_backend = SimpleNamespace(
        qsa_forward=lambda q, k, v, index, layer_id, batch: dense.forward(
            q, k, v, layer_id, batch
        )
    )
    batch = fixture.batch([req], "prefill")
    dense.prepare_metadata(batch)
    reference = attn.forward(x, batch)
    torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
@pytest.mark.parametrize("cut", [1001, 4096, 4097], ids=["unaligned", "page-boundary", "boundary+1"])
def test_chunked_prefill_matches_one_shot(cut: int):
    """Cut points that are not multiples of index_ratio exercise the dual-source compress.

    Not bit for bit: the projections see a different M (5000 vs 5000 - cut rows) and cuBLAS
    may pick a different kernel per shape, so the indexer's q and k round differently and a
    few rows flip one block at the top-k margin. Same tolerance as the oracle tests above."""
    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    attn = fixture.layer(QSA_LAYER)
    length = 5000
    x = _inputs(fixture, [length])[0]

    one_shot = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))
    head = fixture.req(1, 0, cut)
    attn.forward(x[:cut], fixture.batch([head], "prefill"))
    tail = fixture.req(1, cut, length)
    got = attn.forward(x[cut:], fixture.batch([tail], "prefill"))
    torch.testing.assert_close(got.float(), one_shot[cut:].float(), rtol=2e-2, atol=2e-2)

    # A wrong key for the group straddling the cut only swaps one block in a few rows, which
    # the output tolerance hides; the unit-scale slab rows show it as an O(1) error.
    ratio = config.qwen4_args.index_ratio
    slab = fixture.pool.cmp_k_cache(0)
    group_starts = slice(0, length - length % ratio, ratio)
    one_shot_rows, chunked_rows = (
        (fixture.page_table[table_idx, group_starts] // ratio).long() for table_idx in (0, 1)
    )
    torch.testing.assert_close(
        slab[chunked_rows].float(), slab[one_shot_rows].float(), rtol=2e-2, atol=2e-2
    )


@requires_cuda
@pytest.mark.parametrize("rows_per_req", [1, 4], ids=["decode", "verify"])
def test_decode_graph_replay_matches_eager(rows_per_req):
    config = parsed_config()
    # the pending ring holds the rows a verify may reject
    fixture = Fixture(config, num_pages=256, num_speculative_tokens=rows_per_req - 1)
    attn = fixture.layer(QSA_LAYER)
    lengths, steps = [300, 411], 4
    bs, phase = len(lengths), "decode" if rows_per_req == 1 else "verify"
    inputs = _inputs(fixture, lengths, extra=steps * rows_per_req)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]
    attn.forward(
        torch.cat([row[:n] for row, n in zip(inputs, lengths)]),
        fixture.batch(reqs, "prefill"),
    )

    fixture.backend.init_capture_graph(max_seq_len=fixture.page_table.shape[1], bs_list=[bs], rows_per_req=rows_per_req)
    dummy = SimpleNamespace(
        table_idx=fixture.num_req_slots - 1, cached_len=1, device_len=1 + rows_per_req, extend_len=rows_per_req
    )
    rows = bs * rows_per_req
    static = {
        "x": torch.zeros(rows, config.hidden_size, device=fixture.device, dtype=fixture.dtype),
        "positions": torch.zeros(rows, dtype=torch.int32, device=fixture.device),
        "out_loc": torch.zeros(rows, dtype=torch.int32, device=fixture.device),
    }
    capture_batch = SimpleNamespace(
        padded_reqs=[dummy] * bs, reqs=[dummy] * bs, phase=phase, size=bs, padded_size=bs,
        is_prefill=False, is_decode=rows_per_req == 1, positions=static["positions"],
        get_attn_positions=lambda: static["positions"],
        out_loc=static["out_loc"], attn_metadata=None, active_table_idx=None,
    )
    fixture.backend.prepare_for_capture(capture_batch)
    attn.forward(static["x"], capture_batch)  # warmup, same metadata object as the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = attn.forward(static["x"], capture_batch)
    torch.cuda.synchronize()
    # GraphRunner drops the capture batch after capture: nothing the graph reads may live only on its metadata (a
    # verify graph once baked in a prefill row-tile map built there, and its replay read freed memory)
    capture_batch.attn_metadata = None
    # refill the freed small blocks (a tile map is a few hundred bytes) with -1 and keep them through the replays
    scrub = [torch.full((128,), -1, dtype=torch.int32, device=fixture.device) for _ in range(8192)]

    for step in range(steps):
        for req in reqs:
            fixture.step(req, rows_per_req)
        start = step * rows_per_req
        x = torch.cat([row[n + start : n + start + rows_per_req] for row, n in zip(inputs, lengths)])
        batch = fixture.batch(reqs, phase)
        static["x"].copy_(x)
        static["positions"].copy_(batch.positions)
        static["out_loc"].copy_(batch.out_loc)
        fixture.backend.prepare_for_replay(batch)
        # replay must stage into the captured buffers, never reallocate them
        md = batch.attn_metadata
        assert md.block_table.data_ptr() == fixture.backend._graph["block_table"].data_ptr()
        graph.replay()
        replayed = captured_out.clone()
        eager = attn.forward(x, fixture.batch(reqs, phase))
        assert torch.equal(replayed, eager), f"graph replay diverged at {phase} step {step}"
    del scrub


@requires_cuda
def test_verify_after_rejected_rows_compresses_like_decode():
    """A verify whose last rows get rejected must leave the pending ring able to close the open group: the next verify compresses it as decode steps do."""
    config = parsed_config()
    rows, ratio = 4, config.qwen4_args.index_ratio
    fixture = Fixture(config, num_pages=64, num_speculative_tokens=rows - 1)
    attn = fixture.layer(QSA_LAYER)
    # group 300..303 is open after the prompt, its first member in the ring
    length = 301
    x = _inputs(fixture, [length], extra=rows + 1)[0]
    spec, decode = fixture.req(0, 0, length), fixture.req(1, 0, length)
    attn.forward(torch.cat([x[:length]] * 2), fixture.batch([spec, decode], "prefill"))
    # verify 301..304 and keep only 301, then verify 302..305, which closes the group at 303
    fixture.step(spec, rows)
    attn.forward(x[length : length + rows], fixture.batch([spec], "verify"))
    spec.device_len = length + 1
    fixture.step(spec, rows)
    attn.forward(x[length + 1 : length + 1 + rows], fixture.batch([spec], "verify"))
    for position in range(length, length + 3):
        fixture.step(decode)
        attn.forward(x[position : position + 1], fixture.batch([decode], "decode"))

    group = length - length % ratio
    slab = fixture.pool.cmp_k_cache(0)
    got, want = (slab[int(fixture.page_table[r.table_idx, group]) // ratio].float() for r in (spec, decode))
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_row_chunked_scoring_matches_one_chunk(monkeypatch):
    """The scoring workspace bound splits long prefills into row chunks."""
    import freetoken.attention.qsa_sparse as qsa_sparse

    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    length = 600
    x = _inputs(fixture, [length])[0]
    whole = attn.forward(x, fixture.batch([fixture.req(0, 0, length)], "prefill"))

    columns = fixture.page_table.shape[1] // config.qwen4_args.index_ratio
    monkeypatch.setattr(qsa_sparse, "_LOGITS_WORKSPACE_BYTES", 64 * columns * 4)
    chunked = attn.forward(x, fixture.batch([fixture.req(1, 0, length)], "prefill"))
    assert torch.equal(chunked, whole)


@requires_cuda
def test_two_qsa_layers_keep_separate_slab_slots(monkeypatch):
    """Both QSA layers of one forward must hit their own slab slot and ring slice."""
    config = parsed_config(num_layers=8)
    assert config.attention_groups[1].layer_ids == (3, 7)
    fixture = Fixture(config, num_pages=64)
    layers = [fixture.layer(layer_id, seed=layer_id) for layer_id in (3, 7)]
    oracle = _dense_oracle(fixture)
    lengths, steps = [200, 71], 3
    inputs = _inputs(fixture, lengths, extra=steps)
    reqs = [fixture.req(i, 0, n) for i, n in enumerate(lengths)]

    xs = [torch.cat([row[:n] for row, n in zip(inputs, lengths)])]
    xs += [torch.stack([row[n + step] for row, n in zip(inputs, lengths)]) for step in range(steps)]
    for step, x in enumerate(xs):
        if step:
            for req in reqs:
                fixture.step(req)
        batch = fixture.batch(reqs, "prefill" if step == 0 else "decode")
        for attn in layers:
            fixture.ctx.attn_backend = fixture.backend
            got = attn.forward(x, batch)
            fixture.ctx.attn_backend = oracle
            reference = attn.forward(x, batch)
            torch.testing.assert_close(got.float(), reference.float(), rtol=2e-2, atol=2e-2)

    slab = fixture.pool.cmp_k_cache
    assert not torch.equal(slab(0), slab(1))


def test_only_prefill_scores_in_row_tiles(monkeypatch):
    """A verify or decode batch never takes the prefill row-tile path, whose host-built tile map a captured graph
    would keep after the capture batch frees it (MTP verify graphs read freed memory on the first step)."""
    import freetoken.attention.qsa_sparse as qsa

    monkeypatch.setattr(qsa, "_CPU_PINNED", {"device": "cpu", "dtype": torch.int32})
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    monkeypatch.setattr(qsa, "get_global_ctx", lambda: SimpleNamespace(page_table=torch.zeros(4, 256, dtype=torch.int32)))
    backend = object.__new__(qsa.QSASparseAttnBackend)
    backend.device, backend.page_size, backend.index_heads = torch.device("cpu"), 64, 4
    reqs = [SimpleNamespace(extend_len=3, device_len=40, table_idx=0), SimpleNamespace(extend_len=3, device_len=9, table_idx=1)]
    for phase in ("prefill", "verify", "decode"):
        batch = SimpleNamespace(phase=phase, reqs=reqs, padded_reqs=reqs)
        backend.prepare_metadata(batch)
        assert batch.attn_metadata.is_prefill is (phase == "prefill"), phase
        assert backend._scores_in_row_tiles(batch.attn_metadata) is (phase == "prefill"), phase
    backend.index_heads = 2  # the tiled kernel is written for 4 index heads
    batch = SimpleNamespace(phase="prefill", reqs=reqs, padded_reqs=reqs)
    backend.prepare_metadata(batch)
    assert not backend._scores_in_row_tiles(batch.attn_metadata)


def test_selection_stats_match_a_token_loop():
    """P0 (KVSTATS qsa): the ages of the selected tokens behind each query row, and the share of a
    request's selection (union over its verify rows) that its previous sample also selected."""
    from freetoken.attention.qsa_sparse import QSA_SELECTION_AGES, qsa_selection_stats

    gen = torch.Generator().manual_seed(3)
    uids = [7, 7, 9, 9, 4, 4]                                     # two verify rows per request
    positions = torch.tensor([300_000, 300_001, 70_000, 70_001, 20_000, 20_001], dtype=torch.int32)

    def sample():
        idx = torch.randint(0, 20_000, (6, 24), generator=gen, dtype=torch.int32)
        idx = idx * (positions.unsqueeze(1) // 20_000) + torch.randint(0, 3, (6, 24), generator=gen, dtype=torch.int32)
        idx[torch.rand(6, 24, generator=gen) < 0.2] = -1        # padding
        idx[1, :12] = idx[0, :12]                                 # verify rows share blocks
        return idx

    first, second = sample(), sample()
    second[2, :10] = first[3, :10]                                # request 9 keeps part of its selection
    counts, prev = qsa_selection_stats(first, positions, uids, {})
    counts, _ = qsa_selection_stats(second, positions, uids, prev)

    want = {"rows": 6, "tokens": 0, "overlap_hits": 0, "overlap_tokens": 0, "overlap_reqs": 0}
    want.update({f"old{a}": 0 for a in QSA_SELECTION_AGES})
    for r in range(6):
        for t in second[r].tolist():
            if t >= 0:
                want["tokens"] += 1
                for a in QSA_SELECTION_AGES:
                    want[f"old{a}"] += int(positions[r]) - t > a
    for uid in dict.fromkeys(uids):
        rows = [r for r in range(6) if uids[r] == uid]
        cur = {t for r in rows for t in second[r].tolist() if t >= 0}
        old = {t for r in rows for t in first[r].tolist() if t >= 0}
        want["overlap_hits"] += len(cur & old)
        want["overlap_tokens"] += len(cur)
        want["overlap_reqs"] += 1
    assert counts == want
    assert want["old16384"] > 0 and want["overlap_hits"] > 0


def test_selection_sampler_takes_every_verify_row():
    """FREETOKEN_QSA_SELECTION_STATS under DFlash: every decode step is a verify step with 4 rows
    per request. The sampler reads all rows of the real requests (not the padding) and takes each
    request's selection as the union over its rows."""
    from freetoken.attention.qsa_sparse import QSASelectionSampler, qsa_selection_stats
    from freetoken.core import Batch, Req, SamplingParams

    rows_per_req, uids, width = 4, [5, 6], 24
    reqs = [Req(input_ids=torch.arange(70_000, dtype=torch.int32), table_idx=i, cached_len=69_996, output_len=8,
                uid=uid, sampling_params=SamplingParams(), cache_handle=None) for i, uid in enumerate(uids)]
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=rows_per_req)
    batch.positions = torch.arange(69_996, 70_000).repeat(3)       # 2 requests and one padding request
    gen = torch.Generator().manual_seed(5)
    steps = [torch.randint(0, 70_000, (12, width), generator=gen, dtype=torch.int32) for _ in range(2)]
    steps[1][4:8, :12] = steps[0][4:8, :12]                         # request 6 keeps part of its selection
    buffer = steps[0].clone()                                       # the static buffer, padding rows included
    backend = SimpleNamespace(selection_view=lambda rows: buffer[:rows])
    sampler = QSASelectionSampler(1)
    sampler.after_forward(batch, backend)
    buffer.copy_(steps[1])
    sampler.after_forward(batch, backend)
    line = sampler.line()

    row_uids = [u for u in uids for _ in range(rows_per_req)]
    first, prev = qsa_selection_stats(steps[0][:8], batch.positions[:8], row_uids, {})
    second, _ = qsa_selection_stats(steps[1][:8], batch.positions[:8], row_uids, prev)
    assert f"samples=2 skipped=0 rows=16 tokens={first['tokens'] + second['tokens']} " in line
    assert line.endswith(f"overlap={second['overlap_hits'] / second['overlap_tokens']:.3f}/2")
