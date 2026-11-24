"""Full-vocabulary KL divergence against a llama.cpp ``--kl-divergence-base`` file.

The base file (llama.cpp ``tools/perplexity/perplexity.cpp``) is::

    "_logits_"  uint32 n_ctx  int32 n_vocab  int32 n_chunk  int32 tokens[n_chunk * n_ctx]
    then per chunk, the rows of positions n_ctx/2 .. n_ctx-2 (n_ctx - 1 - n_ctx/2 rows), each
    nv = 2*((n_vocab+1)/2) + 4 uint16: float32 scale, float32 min_log_prob, n_vocab codes (+ pad)

A code q decodes to the log-probability ``scale * q + min_log_prob``. The statistics follow
llama.cpp's ``kl_divergence``: KL(base || model) summed over the base tokens with log-probability
above -16, top-1 agreement by first-index argmax on both sides, and the two NLLs of the next token.

The server side (``--enable-kld-endpoint``) runs each chunk as one cold prefill and hands the final
hidden rows to :class:`KldCapture`; this module's ``main`` is the client (``ft bench kld``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

MAGIC = b"_logits_"
HEADER_BYTES = 20
# log-probabilities at or below this are left out of the KL sum (perplexity.cpp: `p_log_base > -16.f`)
KL_FLOOR = -16.0
ROW_FIELDS = ("kld", "nll", "nll_base", "same_top", "p_diff")


@dataclass(frozen=True)
class KldBase:
    """The header of a base file; tokens and rows are read on demand."""

    path: str
    n_ctx: int
    n_vocab: int
    n_chunk: int

    @staticmethod
    def parse_header(head: bytes) -> tuple[int, int, int]:
        """(n_ctx, n_vocab, n_chunk) from the first 20 bytes."""
        if len(head) < HEADER_BYTES or head[:8] != MAGIC:
            raise ValueError(
                "not a llama.cpp KL-divergence base file (no _logits_ magic)"
            )
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", head[8:HEADER_BYTES])
        if n_ctx < 4 or n_vocab < 1 or n_chunk < 1:
            raise ValueError(
                f"bad base header: n_ctx={n_ctx} n_vocab={n_vocab} n_chunk={n_chunk}"
            )
        return n_ctx, n_vocab, n_chunk

    @classmethod
    def open(cls, path: str) -> KldBase:
        with open(path, "rb") as f:
            base = cls(path, *cls.parse_header(f.read(HEADER_BYTES)))
        size = os.path.getsize(path)
        if size != base.file_bytes:
            raise ValueError(
                f"{path}: {size} bytes, the header needs {base.file_bytes} (truncated download?)"
            )
        return base

    @property
    def nv(self) -> int:
        """uint16 words per stored row."""
        return 2 * ((self.n_vocab + 1) // 2) + 4

    @property
    def first(self) -> int:
        """The first scored position of a chunk."""
        return self.n_ctx // 2

    @property
    def n_rows(self) -> int:
        """Scored rows per chunk: positions first .. n_ctx-2, each predicting the next token."""
        return self.n_ctx - 1 - self.first

    @property
    def rows_offset(self) -> int:
        return HEADER_BYTES + 4 * self.n_ctx * self.n_chunk

    @property
    def file_bytes(self) -> int:
        return self.rows_offset + 2 * self.nv * self.n_rows * self.n_chunk

    def _pread(self, n: int, offset: int) -> bytes:
        with open(self.path, "rb") as f:
            data = os.pread(f.fileno(), n, offset)
        if len(data) != n:
            raise ValueError(f"{self.path}: short read of {n} bytes at {offset}")
        return data

    def tokens(self, chunk: int) -> np.ndarray:
        """The chunk's n_ctx tokens as stored: before llama.cpp's BOS replacement of position 0."""
        self._check_chunk(chunk)
        data = self._pread(4 * self.n_ctx, HEADER_BYTES + 4 * self.n_ctx * chunk)
        return np.frombuffer(data, dtype="<i4").copy()

    def rows(self, chunk: int, lo: int = 0, hi: int | None = None) -> np.ndarray:
        """Stored rows [lo, hi) of a chunk, ``[hi-lo, nv]`` uint16; row r scores position first + r."""
        self._check_chunk(chunk)
        hi = self.n_rows if hi is None else hi
        if not 0 <= lo <= hi <= self.n_rows:
            raise ValueError(f"rows [{lo}, {hi}) outside 0..{self.n_rows}")
        row_bytes = 2 * self.nv
        data = self._pread(
            (hi - lo) * row_bytes,
            self.rows_offset + (chunk * self.n_rows + lo) * row_bytes,
        )
        return np.frombuffer(data, dtype="<u2").reshape(hi - lo, self.nv).copy()

    def _check_chunk(self, chunk: int) -> None:
        if not 0 <= chunk < self.n_chunk:
            raise ValueError(f"chunk {chunk} outside 0..{self.n_chunk - 1}")

    def describe(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n_ctx": self.n_ctx,
            "n_vocab": self.n_vocab,
            "n_chunk": self.n_chunk,
            "first": self.first,
            "n_rows": self.n_rows,
        }


def decode_base_rows(raw: torch.Tensor, n_vocab: int) -> torch.Tensor:
    """Stored rows (``[S, nv]``, int16 or uint16 bits) to fp32 log-probabilities ``[S, n_vocab]``."""
    if raw.dtype not in (torch.int16, torch.uint16):
        raise TypeError(f"base rows must be 16-bit words, got {raw.dtype}")
    raw = raw.view(torch.int16)
    head = raw[:, :4].contiguous().view(torch.float32)  # [S, 2]: scale, min_log_prob
    q = raw[:, 4 : 4 + n_vocab].to(torch.int32).bitwise_and_(0xFFFF).to(torch.float32)
    return q.mul_(head[:, 0:1]).add_(head[:, 1:2])


def row_stats(
    logits: torch.Tensor, base_raw: torch.Tensor, targets: torch.Tensor
) -> dict[str, torch.Tensor]:
    """llama.cpp's per-row statistics of model ``logits`` ``[S, V]`` against stored base rows.

    ``targets`` holds the token each row predicts. Returns 1-D tensors: kld = KL(base || model) in
    nats, nll / nll_base = -log p(target), same_top = argmax agreement, p_diff = p(target) - p_base(target).
    """
    n_vocab = logits.shape[1]
    logits = logits.float()
    t = targets.to(device=logits.device, dtype=torch.int64).view(-1, 1)
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    nll = (lse - logits.gather(1, t)).squeeze(1)
    top = logits.argmax(dim=1)
    logq = logits - lse
    base = decode_base_rows(base_raw.to(logits.device), n_vocab)
    nll_base = -base.gather(1, t).squeeze(1)
    top_base = base.argmax(dim=1)
    terms = torch.where(base > KL_FLOOR, base.exp() * (base - logq), 0.0)
    kld = terms.sum(dim=1, dtype=torch.float64)
    return {
        "kld": kld,
        "nll": nll,
        "nll_base": nll_base,
        "same_top": (top == top_base).to(torch.int32),
        "p_diff": torch.exp(-nll) - torch.exp(-nll_base),
    }


def new_chunk_result(chunk: int) -> dict[str, Any]:
    out: dict[str, Any] = {"chunk": chunk, "error": None}
    out.update({name: [] for name in ROW_FIELDS})
    return out


class KldCapture:
    """The scored rows of one base chunk, from the final hidden rows of its cold single-pass prefill.

    Every TP rank builds one with the same arguments and ``ParallelLMHead.forward`` calls ``consume``
    on each, so the vocab all-gather inside ``logits_fn`` runs in lockstep. Only the primary rank
    reads base rows and computes statistics. Its failures go into ``result["error"]`` and never
    skip a ``logits_fn`` call: an exception on one rank would leave the other rank in the collective.
    """

    def __init__(
        self,
        base: KldBase,
        chunk: int,
        targets: torch.Tensor,
        *,
        slice_rows: int,
        primary: bool,
        result: dict[str, Any],
    ) -> None:
        self.base, self.chunk, self.targets = base, chunk, targets
        self.slice_rows, self.primary, self.result = slice_rows, primary, result
        self.done = False

    def consume(
        self, x: torch.Tensor, logits_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """``x``: the pass's final hidden rows, row p = position p. A second call (a draft head under the same batch) does nothing."""
        if self.done:
            return
        self.done = True
        base, first = self.base, self.base.first
        for lo in range(0, base.n_rows, self.slice_rows):
            hi = min(lo + self.slice_rows, base.n_rows)
            logits = logits_fn(x[first + lo : first + hi].contiguous())
            if not self.primary or self.result["error"] is not None:
                continue
            try:
                self._score(logits, lo, hi)
            except Exception as exc:  # noqa: BLE001 - reported to the client, the other ranks keep going
                self.result["error"] = (
                    f"chunk {self.chunk} rows [{lo}, {hi}): {type(exc).__name__}: {exc}"
                )
            del logits

    def _score(self, logits: torch.Tensor, lo: int, hi: int) -> None:
        if logits.shape[1] != self.base.n_vocab:
            raise ValueError(
                f"model gives {logits.shape[1]} logits per row, the base file {self.base.n_vocab}"
            )
        raw = torch.from_numpy(self.base.rows(self.chunk, lo, hi).view(np.int16))
        stats = row_stats(logits, raw, self.targets[lo:hi])
        for name in ROW_FIELDS:
            self.result[name].extend(stats[name].cpu().tolist())


def check_kld_request(
    spec: dict[str, Any], input_ids: Sequence[int], vocab_size: int
) -> KldBase:
    """The base file of a KLD chunk request, or ValueError saying why it cannot run.

    It reads only the file, so every TP rank reaches the same answer."""
    try:
        base = KldBase.open(str(spec["path"]))
        chunk = int(spec["chunk"])
        tokens = base.tokens(chunk)
        slice_rows = int(spec.get("slice_rows", 0))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError(f"kld: {exc}") from exc
    if slice_rows < 1:
        raise ValueError("kld: slice_rows must be at least 1")
    if base.n_vocab != vocab_size:
        raise ValueError(f"kld: base vocab {base.n_vocab} != model vocab {vocab_size}")
    ids = np.asarray(list(input_ids), dtype=np.int64)
    if ids.shape[0] != base.n_ctx:
        raise ValueError(
            f"kld: prompt has {ids.shape[0]} tokens, the base chunk {base.n_ctx}"
        )
    expect = tokens.astype(np.int64)
    if spec.get("bos_token_id") is not None:
        expect[0] = int(spec["bos_token_id"])
    if not np.array_equal(ids, expect):
        raise ValueError(f"kld: prompt tokens differ from chunk {chunk} of {base.path}")
    return base


def capture_targets(base: KldBase, input_ids: torch.Tensor) -> torch.Tensor:
    """The token each scored row predicts: positions first+1 .. n_ctx-1."""
    return input_ids[base.first + 1 : base.n_ctx].to(torch.int64)


# ---------------------------------------------------------------- summary (llama.cpp's report)


def _mean_unc(values: np.ndarray) -> tuple[float, float]:
    """perplexity.cpp mean_and_uncertainty: the mean and the standard error of the mean."""
    n = values.size
    if n < 1:
        return 0.0, 0.0
    f = float(values.sum() / n)
    df = float((values * values).sum() / n - f * f)
    return f, (math.sqrt(df / (n - 1)) if df > 0 and n > 10 else 0.0)


def _cov(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    if n < 10:
        return 0.0
    return float(((a * b).sum() / n - (a.sum() / n) * (b.sum() / n)) / (n - 1))


def _percentile(values: np.ndarray, fraction: float) -> float:
    s = np.sort(values)
    if fraction <= 0:
        return float(s[0])
    if fraction >= 1:
        return float(s[-1])
    p = fraction * (s.size - 1)
    i = int(p)
    p -= i
    return float((1 - p) * s[i] + p * s[min(i + 1, s.size - 1)])


def _rows(chunks: Sequence[dict[str, Any]], name: str) -> np.ndarray:
    return (
        np.concatenate([np.asarray(c[name], dtype=np.float64) for c in chunks])
        if chunks
        else np.zeros(0)
    )


def chunk_summary(c: dict[str, Any]) -> dict[str, Any]:
    """Per-chunk means: the unit a paired bootstrap between two arms resamples."""
    n = len(c["kld"])
    out = {"chunk": c["chunk"], "count": n, "error": c.get("error")}
    if n:
        nll, nll_base = (
            np.asarray(c["nll"], np.float64),
            np.asarray(c["nll_base"], np.float64),
        )
        out.update(
            mean_kld=float(np.mean(c["kld"])),
            same_top=float(np.mean(c["same_top"])),
            mean_nll=float(nll.mean()),
            mean_nll_base=float(nll_base.mean()),
            ppl=math.exp(nll.mean()),
            ppl_base=math.exp(nll_base.mean()),
        )
    return out


def summarize(chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The numbers llama-perplexity --kl-divergence prints, over the chunks without an error."""
    good = [c for c in chunks if not c.get("error") and len(c["kld"])]
    kld, nll, nll_base = _rows(good, "kld"), _rows(good, "nll"), _rows(good, "nll_base")
    same, p_diff = _rows(good, "same_top"), _rows(good, "p_diff")
    n = int(kld.size)
    out: dict[str, Any] = {
        "count": n,
        "chunks": len(good),
        "failed_chunks": [c["chunk"] for c in chunks if c.get("error")],
    }
    if n == 0:
        return out
    lp, lp_unc = _mean_unc(nll)
    lb, lb_unc = _mean_unc(nll_base)
    cov = _cov(nll, nll_base)
    ratio_unc = math.sqrt(max(lp_unc**2 + lb_unc**2 - 2 * cov, 0.0))
    ppl, ppl_base = math.exp(lp), math.exp(lb)
    k, k_unc = _mean_unc(kld)
    pd, pd_unc = _mean_unc(p_diff)
    mse, mse_unc = _mean_unc(p_diff * p_diff)
    rms = math.sqrt(mse)
    top = float(same.sum() / n)
    out.update(
        ppl=ppl,
        ppl_unc=ppl * lp_unc,
        ppl_base=ppl_base,
        ppl_base_unc=ppl_base * lb_unc,
        cor_ln_ppl=(cov / (lp_unc * lb_unc) if lp_unc and lb_unc else 0.0),
        ln_ppl_ratio=lp - lb,
        ln_ppl_ratio_unc=ratio_unc,
        ppl_ratio=math.exp(lp - lb),
        ppl_ratio_unc=math.exp(lp - lb) * ratio_unc,
        ppl_diff=ppl - ppl_base,
        ppl_diff_unc=math.sqrt(
            max(
                (ppl * lp_unc) ** 2
                + (ppl_base * lb_unc) ** 2
                - 2 * ppl * ppl_base * cov,
                0.0,
            )
        ),
        mean_kld=k,
        mean_kld_unc=k_unc,
        kld_percentiles={
            str(f): _percentile(kld, f)
            for f in (0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
        },
        mean_p_diff=pd,
        mean_p_diff_unc=pd_unc,
        rms_p_diff=rms,
        rms_p_diff_unc=(0.5 / rms * mse_unc if rms else 0.0),
        max_abs_p_diff=float(np.abs(p_diff).max()),
        same_top=top,
        same_top_unc=(math.sqrt(top * (1 - top) / (n - 1)) if n > 1 else 0.0),
    )
    return out


def paired_bootstrap(
    a: Sequence[dict[str, Any]],
    b: Sequence[dict[str, Any]],
    metric: str = "kld",
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Arm b minus arm a of a row metric's mean, resampling the chunks both arms scored."""
    by_a = {c["chunk"]: c for c in a if not c.get("error") and len(c[metric])}
    by_b = {c["chunk"]: c for c in b if not c.get("error") and len(c[metric])}
    common = sorted(set(by_a) & set(by_b))
    if not common:
        raise ValueError("the two runs share no scored chunk")
    sums = np.array(
        [
            [np.sum(by_a[i][metric]), np.sum(by_b[i][metric]), len(by_a[i][metric])]
            for i in common
        ],
        np.float64,
    )
    if np.any(sums[:, 2] != np.array([len(by_b[i][metric]) for i in common])):
        raise ValueError("the two runs scored a different number of rows in a chunk")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(n_boot, len(common)))
    s = sums[idx].sum(axis=1)
    boot = (s[:, 1] - s[:, 0]) / s[:, 2]
    diff = float((sums[:, 1].sum() - sums[:, 0].sum()) / sums[:, 2].sum())
    return {
        "metric": metric,
        "chunks": len(common),
        "diff": diff,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def base_only_nll(base: KldBase, chunks: Sequence[int]) -> list[float]:
    """-log p_base(next token) of every scored row, from the file alone (base PPL check, no model)."""
    out: list[float] = []
    row_bytes = 2 * base.nv
    with open(base.path, "rb") as f:
        fd = f.fileno()
        for chunk in chunks:
            tokens = base.tokens(chunk)
            for r in range(base.n_rows):
                off = base.rows_offset + (chunk * base.n_rows + r) * row_bytes
                scale, min_lp = np.frombuffer(os.pread(fd, 8, off), dtype="<f4")
                tok = int(tokens[base.first + 1 + r])
                q = np.frombuffer(os.pread(fd, 2, off + 8 + 2 * tok), dtype="<u2")[0]
                out.append(-float(scale * np.float32(q) + min_lp))
    return out


# ---------------------------------------------------------------- client: ft bench kld


def parse_chunks(text: str | None, n_chunk: int) -> list[int]:
    """ "0-9,20" -> [0..9, 20]; None -> every chunk."""
    if not text:
        return list(range(n_chunk))
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def _post(
    url: str, body: dict[str, Any], api_key: str | None, timeout: float
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"{url}: HTTP {exc.code}: {exc.read().decode(errors='replace')}"
        ) from exc


def _cmd_run(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/") + "/v1/internal/kld"
    # the server reads the file; an empty chunk list returns its header only
    info = _post(
        url, {"base_path": args.base, "chunks": []}, args.api_key, args.timeout
    )
    todo = parse_chunks(args.chunks, info["base"]["n_chunk"])
    results: list[dict[str, Any]] = []
    for i in range(0, len(todo), args.batch):
        part = todo[i : i + args.batch]
        body = {
            "base_path": args.base,
            "chunks": part,
            "slice_rows": args.slice_rows,
            "bos_token_id": args.bos_token_id,
        }
        reply = _post(url, body, args.api_key, args.timeout)
        info = {k: v for k, v in reply.items() if k not in ("chunks", "summary")}
        results.extend(reply["chunks"])
        s = summarize(results)
        print(
            f"[{len(results)}/{len(todo)}] KLD {s.get('mean_kld', float('nan')):.6f} "
            f"top1 {s.get('same_top', float('nan')):.4f} PPL {s.get('ppl', float('nan')):.4f} "
            f"base {s.get('ppl_base', float('nan')):.4f} failed {s['failed_chunks']}",
            file=sys.stderr,
        )
    out = {
        **info,
        "label": args.label,
        "summary": summarize(results),
        "per_chunk": [chunk_summary(c) for c in results],
        "chunks": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(json.dumps(out["summary"], indent=1))
    return 0 if not out["summary"]["failed_chunks"] else 1


def _cmd_summary(args: argparse.Namespace) -> int:
    with open(args.result) as f:
        print(json.dumps(summarize(json.load(f)["chunks"]), indent=1))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    with open(args.a) as f:
        a = json.load(f)["chunks"]
    with open(args.b) as f:
        b = json.load(f)["chunks"]
    out = [
        paired_bootstrap(a, b, m, n_boot=args.n_boot, seed=args.seed)
        for m in ("kld", "same_top", "nll")
    ]
    print(json.dumps(out, indent=1))
    return 0


def _cmd_base_ppl(args: argparse.Namespace) -> int:
    base = KldBase.open(args.base)
    nll = np.asarray(
        base_only_nll(base, parse_chunks(args.chunks, base.n_chunk)), np.float64
    )
    m, unc = _mean_unc(nll)
    print(
        json.dumps(
            {
                **base.describe(),
                "rows": int(nll.size),
                "ppl_base": math.exp(m),
                "ppl_base_unc": math.exp(m) * unc,
            },
            indent=1,
        )
    )
    return 0


def main(argv: list[str] | None = None, prog: str = "ft bench kld") -> int:
    p = argparse.ArgumentParser(
        prog=prog,
        description="KL divergence of a running server against a llama.cpp base file.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "run", help="score chunks through a server started with --enable-kld-endpoint"
    )
    r.add_argument("--url", default="http://127.0.0.1:1919")
    r.add_argument("--base", required=True, help="base file path as the SERVER sees it")
    r.add_argument("--chunks", default=None, help='e.g. "0-99" (default: all)')
    r.add_argument("--batch", type=int, default=10, help="chunks per HTTP request")
    r.add_argument(
        "--slice-rows", type=int, default=16, help="rows per lm_head slice on the GPU"
    )
    r.add_argument(
        "--bos-token-id",
        type=int,
        default=None,
        help="only when the base run replaced token 0 by BOS",
    )
    r.add_argument("--api-key", default=os.environ.get("FREETOKEN_API_KEY"))
    r.add_argument("--timeout", type=float, default=1800.0)
    r.add_argument("--label", default=None)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summary", help="print the summary of a result file")
    s.add_argument("result")
    c = sub.add_parser("compare", help="paired chunk bootstrap of arm b minus arm a")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--n-boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    bp = sub.add_parser(
        "base-ppl",
        help="base PPL from the file alone (expect 3.2749 for flashnext-bf16-base)",
    )
    bp.add_argument("base")
    bp.add_argument("--chunks", default=None)
    args = p.parse_args(argv)
    return {
        "run": _cmd_run,
        "summary": _cmd_summary,
        "compare": _cmd_compare,
        "base-ppl": _cmd_base_ppl,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
