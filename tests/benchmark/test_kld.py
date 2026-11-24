"""The KLD tool against llama.cpp's base-file format and its kl_divergence math (perplexity.cpp)."""

from __future__ import annotations

import math
import os
import struct
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from freetoken.benchmark import kld
from freetoken.benchmark.kld import KldBase, KldCapture


def _nearest_int(f: np.ndarray) -> np.ndarray:
    """perplexity.cpp nearest_int: the 1.5*2^23 float trick, round half to even."""
    v = (f.astype(np.float32) + np.float32(12582912.0)).view(np.int32)
    return (v & 0x007FFFFF) - 0x00400000


def _encode_row(logits: np.ndarray) -> np.ndarray:
    """perplexity.cpp log_softmax (the writer): one stored row of nv uint16 words."""
    logits = logits.astype(np.float32)
    n_vocab = logits.size
    nv = 2 * ((n_vocab + 1) // 2) + 4
    max_l = logits.max()
    min_l = max(logits.min(), np.float32(max_l - np.float32(16)))
    sum_exp = float(np.sum(np.exp(logits - max_l).astype(np.float64)))
    lse = np.float32(math.log(sum_exp))
    min_lp = np.float32(np.float32(min_l - max_l) - lse)
    scale = np.float32(np.float32(max_l - min_l) / np.float32(65535))
    row = np.zeros(nv, dtype=np.uint16)
    row[:4] = np.array([scale, min_lp], dtype="<f4").view(np.uint16)
    if scale:
        q = _nearest_int(np.float32(1) / scale * (logits - min_l))
        row[4 : 4 + n_vocab] = np.where(logits > min_l, q, 0).astype(np.uint16)
    return row


def _write_base(path, tokens: np.ndarray, base_logits: np.ndarray) -> KldBase:
    """A base file as llama-perplexity --kl-divergence-base writes it. base_logits: [n_chunk, n_rows, V]."""
    n_chunk, n_ctx = tokens.shape
    n_vocab = base_logits.shape[2]
    with open(path, "wb") as f:
        f.write(b"_logits_")
        f.write(struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
        f.write(tokens.astype("<i4").tobytes())
        f.writelines(
            _encode_row(base_logits[c, r]).astype("<u2").tobytes()
            for c in range(n_chunk)
            for r in range(base_logits.shape[1])
        )
    return KldBase.open(str(path))


def _reference_row(logits: np.ndarray, row: np.ndarray, tok: int) -> dict:
    """perplexity.cpp log_softmax (the reader), in its float/double order."""
    logits = logits.astype(np.float32)
    n_vocab = logits.size
    imax = int(np.argmax(logits))  # first index of the max, like its strict > scan
    max_l = logits[imax]
    sum_exp = float(np.sum(np.exp(logits - max_l).astype(np.float64)))
    lse = np.float32(math.log(sum_exp))
    scale, min_lp = row[:4].view("<f4")
    q = row[4 : 4 + n_vocab].astype(np.float32)
    nll = np.float32(np.float32(max_l + lse) - logits[tok])
    nll_base = -np.float32(scale * q[tok] + min_lp)
    max_l = np.float32(max_l + lse)
    lpb = (scale * q + min_lp).astype(np.float32)
    imax_base = int(np.argmax(lpb))
    keep = lpb > np.float32(-16)
    terms = (np.exp(lpb[keep]) * (lpb[keep] - logits[keep] + max_l)).astype(np.float32)
    return {
        "kld": float(np.sum(terms.astype(np.float64))),
        "nll": float(nll),
        "nll_base": float(nll_base),
        "same_top": int(imax == imax_base),
        "p_diff": float(np.exp(np.float32(-nll)) - np.exp(np.float32(-nll_base))),
    }


@pytest.fixture
def small_base(tmp_path):
    rng = np.random.default_rng(0)
    n_chunk, n_ctx, vocab = 3, 16, 301  # odd vocab: one pad word per row
    tokens = rng.integers(0, vocab, size=(n_chunk, n_ctx))
    n_rows = n_ctx - 1 - n_ctx // 2
    base_logits = rng.normal(0, 3, size=(n_chunk, n_rows, vocab)).astype(np.float32)
    base_logits[..., 7] += (
        12  # some rows with a peaked distribution: codes below min_logit clamp to 0
    )
    return _write_base(tmp_path / "base.kld", tokens, base_logits), tokens, base_logits


def test_header_and_layout(small_base):
    base, tokens, _ = small_base
    assert (base.n_ctx, base.n_vocab, base.n_chunk) == (16, 301, 3)
    assert base.nv == 2 * 151 + 4 and base.first == 8 and base.n_rows == 7
    assert os.path.getsize(base.path) == base.file_bytes
    np.testing.assert_array_equal(base.tokens(2), tokens[2])
    assert base.rows(1, 2, 5).shape == (3, base.nv)


def test_truncated_and_foreign_files_are_refused(small_base, tmp_path):
    base, _, _ = small_base
    with open(base.path, "rb") as f:
        data = f.read()
    (tmp_path / "short.kld").write_bytes(data[:-2])
    with pytest.raises(ValueError, match="truncated"):
        KldBase.open(str(tmp_path / "short.kld"))
    (tmp_path / "other.bin").write_bytes(b"GGUF" + data[4:])
    with pytest.raises(ValueError, match="magic"):
        KldBase.open(str(tmp_path / "other.bin"))


def test_decode_matches_writer(small_base):
    """A stored code decodes to the writer's log-probability within half a quantization step."""
    base, _, base_logits = small_base
    raw = torch.from_numpy(base.rows(0).view(np.int16))
    got = kld.decode_base_rows(raw, base.n_vocab).numpy()
    lg = base_logits[0].astype(np.float64)
    want = (
        lg
        - np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True))
        - lg.max(1, keepdims=True)
    )
    scale = raw[:, :4].contiguous().view(torch.float32)[:, 0].numpy()
    above = lg > (lg.max(1, keepdims=True) - 16)
    err = np.abs(got - want)[above]
    assert err.max() <= (scale.max() / 2) * 1.01 + 1e-5
    # codes of tokens 16 nats below the top clamp to the row minimum
    assert np.all(
        got[~above] == np.repeat(got.min(1, keepdims=True), got.shape[1], 1)[~above]
    )


@pytest.mark.parametrize("shift", [0.0, 0.3])
def test_row_stats_match_llama_cpp_reader(small_base, shift):
    base, tokens, base_logits = small_base
    rng = np.random.default_rng(1)
    chunk = 1
    model = base_logits[chunk] + shift * rng.normal(
        size=base_logits[chunk].shape
    ).astype(np.float32)
    rows = base.rows(chunk)
    targets = torch.from_numpy(tokens[chunk, base.first + 1 :])
    got = kld.row_stats(
        torch.from_numpy(model), torch.from_numpy(rows.view(np.int16)), targets
    )
    for r in range(base.n_rows):
        want = _reference_row(model[r], rows[r], int(targets[r]))
        assert got["same_top"][r].item() == want["same_top"]
        for name in ("kld", "nll", "nll_base", "p_diff"):
            assert got[name][r].item() == pytest.approx(want[name], abs=2e-6), (name, r)
    if shift == 0.0:
        # the model IS the base up to the uint16 rounding of the stored rows
        assert (
            got["kld"].abs().max().item() < 1e-4
        )  # the uint16 step (16/65535 nats) leaves a small floor


def test_top1_ties_take_the_first_index():
    v = 64
    logits = torch.zeros(1, v)
    logits[0, [5, 9]] = 3.0  # model tie: first index 5
    lg = np.zeros(v, np.float32)
    lg[[5, 9]] = 3.0
    row = _encode_row(lg)  # base tie at 5 and 9 too
    s = kld.row_stats(
        logits, torch.from_numpy(row[None].view(np.int16)), torch.tensor([1])
    )
    assert s["same_top"].item() == 1
    logits[0, 5] = 2.9  # model now picks 9, base keeps 5
    s = kld.row_stats(
        logits, torch.from_numpy(row[None].view(np.int16)), torch.tensor([1])
    )
    assert s["same_top"].item() == 0


class _Head:
    """A CPU stand-in for ParallelLMHead.logits that records the slice shapes."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight, self.calls = weight, []

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(rows.shape))
        return rows.float() @ self.weight.T


def _capture_case(tmp_path, n_ctx=32, vocab=257, hidden=12, n_chunk=2):
    g = torch.Generator().manual_seed(3)
    w = torch.randn(vocab, hidden, generator=g)
    xs = torch.randn(n_chunk, n_ctx, hidden, generator=g)
    tokens = np.random.default_rng(4).integers(0, vocab, size=(n_chunk, n_ctx))
    first = n_ctx // 2
    base_logits = (
        xs[:, first : n_ctx - 1] @ w.T
    ).numpy() * 1.1  # a "base" a bit sharper than the model
    base = _write_base(tmp_path / "cap.kld", tokens, base_logits)
    return base, w, xs, tokens


@pytest.mark.parametrize("slice_rows", [1, 4, 7, 64])
def test_capture_slices_cover_every_scored_row_once(tmp_path, slice_rows):
    base, w, xs, tokens = _capture_case(tmp_path)
    chunk, x = 1, xs[1]
    ids = torch.from_numpy(tokens[chunk])
    result = kld.new_chunk_result(chunk)
    head = _Head(w)
    cap = KldCapture(
        base,
        chunk,
        kld.capture_targets(base, ids),
        slice_rows=slice_rows,
        primary=True,
        result=result,
    )
    cap.consume(x, head)
    cap.consume(x, head)  # a draft head under the same batch: ignored
    assert sum(n for n, _ in head.calls) == base.n_rows
    assert max(n for n, _ in head.calls) == min(slice_rows, base.n_rows)
    assert result["error"] is None and len(result["kld"]) == base.n_rows
    whole = kld.row_stats(
        x[base.first : base.n_ctx - 1] @ w.T,
        torch.from_numpy(base.rows(chunk).view(np.int16)),
        ids[base.first + 1 :],
    )
    for name in kld.ROW_FIELDS:
        np.testing.assert_allclose(
            result[name], whole[name].numpy(), rtol=1e-6, atol=1e-6
        )


def test_non_primary_rank_makes_the_same_collective_calls(tmp_path):
    base, w, xs, tokens = _capture_case(tmp_path)
    ids = torch.from_numpy(tokens[0])
    heads, results = [], []
    for primary in (True, False):
        head, result = _Head(w), kld.new_chunk_result(0)
        KldCapture(
            base,
            0,
            kld.capture_targets(base, ids),
            slice_rows=5,
            primary=primary,
            result=result,
        ).consume(xs[0], head)
        heads.append(head)
        results.append(result)
    assert heads[0].calls == heads[1].calls
    assert len(results[0]["kld"]) == base.n_rows and results[1]["kld"] == []


def test_primary_failure_keeps_the_ranks_in_step(tmp_path):
    base, w, xs, tokens = _capture_case(tmp_path)
    ids = torch.from_numpy(tokens[0])
    head, result = _Head(w), kld.new_chunk_result(0)
    broken = KldBase(
        base.path, base.n_ctx, base.n_vocab + 1, base.n_chunk
    )  # vocab mismatch on the primary only
    KldCapture(
        broken,
        0,
        kld.capture_targets(base, ids),
        slice_rows=5,
        primary=True,
        result=result,
    ).consume(xs[0], head)
    assert sum(n for n, _ in head.calls) == base.n_rows
    assert "logits per row" in result["error"]


def test_admission_checks(small_base):
    base, tokens, _ = small_base
    spec = {"path": base.path, "chunk": 1, "slice_rows": 4}
    ids = tokens[1].tolist()
    assert kld.check_kld_request(spec, ids, base.n_vocab) == base
    for bad_spec, bad_ids, vocab, match in [
        (spec, ids, base.n_vocab + 1, "vocab"),
        (spec, ids[:-1] + [ids[-1] ^ 1], base.n_vocab, "differ"),
        (spec, ids[:-1], base.n_vocab, "tokens"),
        ({**spec, "chunk": 3}, ids, base.n_vocab, "chunk 3"),
        ({**spec, "slice_rows": 0}, ids, base.n_vocab, "slice_rows"),
        ({**spec, "path": base.path + ".missing"}, ids, base.n_vocab, "No such file"),
        ({"chunk": 1}, ids, base.n_vocab, "kld: 'path'"),
    ]:
        with pytest.raises(ValueError, match=match):
            kld.check_kld_request(bad_spec, bad_ids, vocab)
    # llama.cpp replaced token 0 by BOS when the GGUF said add_bos: then the prompt carries the BOS
    bos = [7] + ids[1:]
    with pytest.raises(ValueError, match="differ"):
        kld.check_kld_request(spec, bos, base.n_vocab)
    assert kld.check_kld_request({**spec, "bos_token_id": 7}, bos, base.n_vocab) == base


def test_summary_matches_llama_cpp_formulas():
    rng = np.random.default_rng(5)
    chunks = []
    for c in range(4):
        n = 50
        nll_base = rng.gamma(2.0, 0.6, n)
        chunks.append(
            {
                "chunk": c,
                "error": None,
                "kld": rng.gamma(1.0, 0.01, n).tolist(),
                "nll": (nll_base + rng.normal(0, 0.05, n)).tolist(),
                "nll_base": nll_base.tolist(),
                "same_top": (rng.random(n) < 0.9).astype(int).tolist(),
                "p_diff": rng.normal(0, 0.02, n).tolist(),
            }
        )
    chunks.append({**kld.new_chunk_result(9), "error": "boom"})
    s = kld.summarize(chunks)
    allk = np.concatenate([c["kld"] for c in chunks[:4]])
    alln = np.concatenate([c["nll"] for c in chunks[:4]])
    assert s["count"] == 200 and s["failed_chunks"] == [9]
    assert s["mean_kld"] == pytest.approx(allk.mean())
    assert s["mean_kld_unc"] == pytest.approx(
        allk.std(ddof=0) / math.sqrt(199), rel=1e-9
    )
    assert s["ppl"] == pytest.approx(math.exp(alln.mean()))
    assert s["kld_percentiles"]["0.99"] == pytest.approx(np.percentile(allk, 99))
    assert s["same_top"] == pytest.approx(
        np.mean(np.concatenate([c["same_top"] for c in chunks[:4]]))
    )
    pc = kld.chunk_summary(chunks[0])
    assert (
        pc["mean_kld"] == pytest.approx(np.mean(chunks[0]["kld"])) and pc["count"] == 50
    )


def test_paired_bootstrap():
    rng = np.random.default_rng(6)
    a = [
        {"chunk": c, "error": None, "kld": rng.gamma(1, 0.01, 20).tolist()}
        for c in range(30)
    ]
    same = kld.paired_bootstrap(a, a, n_boot=200)
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]
    b = [{**c, "kld": (np.asarray(c["kld"]) + 0.005).tolist()} for c in a]
    out = kld.paired_bootstrap(a, b, n_boot=500)
    assert out["diff"] == pytest.approx(0.005) and out["ci95"][0] == pytest.approx(
        0.005
    )


def test_base_only_nll_reads_the_target_code(small_base):
    base, tokens, _ = small_base
    nll = kld.base_only_nll(base, [0, 2])
    want = []
    for c in (0, 2):
        raw = torch.from_numpy(base.rows(c).view(np.int16))
        lp = kld.decode_base_rows(raw, base.n_vocab)
        t = torch.from_numpy(tokens[c, base.first + 1 :]).long()
        want.extend((-lp.gather(1, t.view(-1, 1)).squeeze(1)).tolist())
    np.testing.assert_allclose(nll, want, rtol=0, atol=1e-6)


def test_parse_chunks():
    assert kld.parse_chunks(None, 3) == [0, 1, 2]
    assert kld.parse_chunks("0-2,7", 100) == [0, 1, 2, 7]


_REAL_URL = os.environ.get("FREETOKEN_KLD_BASE_URL")


@pytest.mark.skipif(
    not _REAL_URL, reason="set FREETOKEN_KLD_BASE_URL to range-read a real base file"
)
def test_real_base_file_header_by_range_read():
    """Header, chunk 0 tokens and the first row of the real file, by HTTP range reads only."""

    def get(lo: int, hi: int) -> tuple[bytes, str]:
        req = urllib.request.Request(
            _REAL_URL, headers={"Range": f"bytes={lo}-{hi - 1}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(), r.headers.get("Content-Range", "")
        except urllib.error.URLError as exc:
            pytest.skip(f"no network: {exc}")

    head, crange = get(0, kld.HEADER_BYTES)
    n_ctx, n_vocab, n_chunk = KldBase.parse_header(head)
    base = KldBase("remote", n_ctx, n_vocab, n_chunk)
    assert int(crange.rsplit("/", 1)[1]) == base.file_bytes
    toks, _ = get(kld.HEADER_BYTES, kld.HEADER_BYTES + 4 * n_ctx)
    row, _ = get(base.rows_offset, base.rows_offset + 2 * base.nv)
    raw = torch.from_numpy(np.frombuffer(row, "<u2").copy().view(np.int16))[None]
    lp = kld.decode_base_rows(raw, n_vocab)[0].double()
    assert lp.max().item() <= 0 and torch.exp(
        lp[lp > kld.KL_FLOOR]
    ).sum().item() == pytest.approx(1, abs=1e-3)
    if (n_ctx, n_vocab, n_chunk) == (512, 248320, 100):  # flashnext-bf16-base.kld
        assert (
            np.frombuffer(toks, "<i4")[0] == 695
        )  # no BOS at position 0: the base run had add_bos = false


class _GatherComm:
    """Two TP ranks in one process: rank r's all_gather stacks every rank's shard GEMM of r's last input."""

    def __init__(self, heads, rank):
        self.heads, self.rank = heads, rank

    def all_gather(self, local):
        x = self.heads[self.rank].last_x
        return torch.cat([h.shard_logits(x) for h in self.heads])


@pytest.mark.parametrize("tp", [1, 2])
def test_lm_head_hook_scores_full_vocab_rows_and_serves_the_last_row(
    tmp_path, monkeypatch, tp
):
    """ParallelLMHead.forward on a prefill: the capture sees the all-gathered full-vocab logits of every
    scored row (equal to one unsharded head), and the served last-row logits do not change."""
    import freetoken.distributed.info as dist_info
    from freetoken.distributed import DistributedInfo
    from freetoken.layers import ParallelLMHead
    from freetoken.layers import embedding as emb
    from freetoken.models.loader import shard_tensor

    n_ctx, vocab, hidden = 32, 257, 12  # odd vocab: the TP shards pad the last one
    base, w, xs, tokens = _capture_case(
        tmp_path, n_ctx=n_ctx, vocab=vocab, hidden=hidden
    )
    x, ids = xs[0], torch.from_numpy(tokens[0])
    heads = []
    for rank in range(tp):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, tp))
        h = ParallelLMHead(vocab, hidden, prefix="lm_head")
        shard = (
            shard_tensor(
                "lm_head.weight", w, rank=rank, world_size=tp, num_kv_heads=None
            )
            if tp > 1
            else w.clone()
        )
        h.weight = torch.cat(
            [shard, shard.new_zeros(h.num_embeddings_tp - shard.shape[0], hidden)]
        )
        heads.append(h)
    for rank, h in enumerate(heads):
        h.shard_logits = h._local_logits
        h._local_logits = lambda rows, h=h: (
            setattr(h, "last_x", rows) or h.shard_logits(rows)
        )
        h._comm = _GatherComm(heads, rank)
    want = kld.row_stats(
        x[base.first : n_ctx - 1] @ w.T,
        torch.from_numpy(base.rows(0).view(np.int16)),
        ids[base.first + 1 :],
    )

    def forward(head, capture):
        batch = SimpleNamespace(
            size=1,
            features=None,
            is_prefill=True,
            kld=capture,
            attn_metadata=SimpleNamespace(
                get_last_indices=lambda bs: torch.tensor([n_ctx - 1])
            ),
        )
        monkeypatch.setattr(emb, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
        return head.forward(x)

    for rank, h in enumerate(heads):
        served_off = forward(h, None)
        result = kld.new_chunk_result(0)
        cap = KldCapture(
            base,
            0,
            kld.capture_targets(base, ids),
            slice_rows=4,
            primary=rank == 0,
            result=result,
        )
        torch.testing.assert_close(forward(h, cap), served_off, rtol=0, atol=0)
        torch.testing.assert_close(served_off, x[-1:] @ w.T, rtol=1e-5, atol=1e-5)
        if rank == 0:
            assert result["error"] is None
            for name in kld.ROW_FIELDS:
                np.testing.assert_allclose(
                    result[name], want[name].numpy(), rtol=1e-5, atol=1e-5
                )
        else:
            assert result["kld"] == []
