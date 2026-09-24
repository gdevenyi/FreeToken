"""Disk-backed PLE table (--ple-backend disk): the C++ store hashes n-gram windows and batch-reads rows from the checkpoint's fp8 shard tensors into pinned staging; the captured ``lookup`` is a fixed-shape H2D copy + dequant.

Hash windows are pure functions of ``req.input_ids`` + ``device_len`` (prefix hits, restores and COW forks need no bookkeeping); the decode input token lives device-side under overlap scheduling and is read back here.
"""

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import safetensors
import torch

from freetoken.mm import MM_PAD_SHIFT_VALUE, restore_placeholder

from freetoken.core import Batch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

from .weight import (
    _PLE_SCALE_SUFFIX,
    _PLE_SHARD_RE,
    _PLE_ST_DTYPE,
    _ple_table_files,
    _safetensors_header,
)

_IO_URING_ENV = "FREETOKEN_PLE_IO_URING"
_SYNC_ENV = "FREETOKEN_PLE_SYNC"  # auto | wait | gate

logger = init_logger(__name__)


def _context(ids: torch.Tensor, position: int, eos: int) -> list[int]:
    """The two token ids before ``position``; eos pads past the start."""
    return [int(ids[position - 2]) if position >= 2 else eos,
            int(ids[position - 1]) if position >= 1 else eos]


@dataclass(frozen=True)
class PleRowSource:
    """On-disk row layout: equal extents, row i of an extent at ``base + i * row_stride`` (a repacked flat file is one extent with its own stride)."""

    paths: list[str]
    extent_file: list[int]
    extent_base: list[int]
    rows_per_extent: int
    row_bytes: int
    row_stride: int
    scale: float

    @property
    def total_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent


def source_from_safetensors(folder: str) -> PleRowSource:
    """Map the checkpoint's ``ngram_embedding.shard_<i>`` tensors in place: one extent per shard, no copy."""
    rows = cols = 0
    scale: torch.Tensor | None = None
    paths: list[str] = []
    path_idx: dict[str, int] = {}
    shards: dict[int, tuple[int, int]] = {}
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE shard {key} has dtype {meta['dtype']}, expected {_PLE_ST_DTYPE}")
            if rows and tuple(meta["shape"]) != (rows, cols):
                raise ValueError(f"PLE shard {key} is {meta['shape']}, expected {[rows, cols]}")
            rows, cols = meta["shape"]
            if path not in path_idx:
                path_idx[path] = len(paths)
                paths.append(path)
            idx = int(match.group("shard"))
            if idx in shards:
                raise ValueError(f"duplicate PLE shard {idx} in {path}")
            shards[idx] = (path_idx[path], base + meta["data_offsets"][0])
    if sorted(shards) != list(range(len(shards))) or not shards:
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(shards)[:8]}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")
    order = [shards[i] for i in range(len(shards))]
    return PleRowSource(paths, [f for f, _ in order], [b for _, b in order], rows, cols, cols, float(scale))


def resolve_row_source(folder: str) -> PleRowSource:
    """Pick the row source for a checkpoint; the seam where a repacked format would plug in."""
    return source_from_safetensors(folder)


class _Filler:
    """The one thread that drives the PleStore (engine-thread-only C++, no locks), FIFO.

    Graph fills run here, submitted before the launch: the fill releases the captured decode's
    WAIT, and cuGraphLaunch may block until the GPU drains, so it must not wait for the launch
    to return (a large graph deadlocked that way; fork issue #65)."""

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="ple-filler", daemon=True)
        self._thread.start()

    def submit(self, fn) -> Future:
        fut: Future = Future()
        self._jobs.put((fn, fut))
        return fut

    def _run(self) -> None:
        while (item := self._jobs.get()) is not None:
            fn, fut = item
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 -- surfaced to the engine thread
                fut.set_exception(exc)

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5)


class _PendingFill:
    """A graph fill submitted before its launch; ``cancel`` after a failed launch."""

    def __init__(self, table: "DiskRowTable", future: Future, cancelled: threading.Event) -> None:
        self._table, self._future, self._cancelled = table, future, cancelled

    def cancel(self) -> None:
        # no WAIT consumed the signal: clear it, or the next step's WAIT passes on stale rows
        self._cancelled.set()
        try:
            self._future.result()
        except BaseException:  # noqa: BLE001 -- the launch error is the one to surface
            pass
        self._table._flag.zero_()


class DiskRowTable:
    """``PLETableBackend`` whose rows are read from disk per fill (--ple-backend disk)."""

    def __init__(
        self,
        source: PleRowSource,
        hash_constants: dict,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from freetoken.kernel import _ple_store

        self.num_rows = source.total_rows
        self.head_dim = source.row_bytes  # fp8: one byte per element
        self.dtype = dtype
        self.heads = int(hash_constants["num_ngram_heads"])
        self.scale = source.scale
        self.eos_token_id = int(hash_constants["eos_token_id"])
        self.image_token_id = hash_constants.get("image_token_id")
        sizes = [int(x) for x in hash_constants["per_head_vocab_sizes"]]
        offsets = [int(x) for x in hash_constants["per_head_offsets"]]
        need = max(o + s for o, s in zip(offsets, sizes))
        if need > source.total_rows:
            raise ValueError(
                f"PLE row source holds {source.total_rows} rows but the hash addresses {need}; incomplete checkpoint?"
            )
        self._store = _ple_store.PleStore(
            paths=list(source.paths),
            extent_file=list(source.extent_file),
            extent_base=list(source.extent_base),
            rows_per_extent=source.rows_per_extent,
            row_bytes=source.row_bytes,
            row_stride=source.row_stride,
            multipliers=[int(x) for x in hash_constants["layer_multipliers"]],
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            eos_token_id=self.eos_token_id,
            use_io_uring=os.getenv(_IO_URING_ENV, "1") != "0",
        )
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._token_bytes = self.heads * self.head_dim
        # allocated up front: pinned alloc inside stream capture is illegal; one replay consumes it at a time
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * self._token_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()  # padded decode lanes read whatever sits here
        # outlives any one graph: a cache rebuild recaptures against the same pointer
        self._graph_dev = torch.empty(
            max_graph_rows * self._token_bytes, dtype=torch.uint8, device=self._device
        )
        eager_bytes = max_extend_tokens * self._token_bytes
        self._eager_pinned = alloc_pinned_tensor(eager_bytes, dtype=torch.uint8)
        self._eager_pinned.zero_()  # the warmup prefill stages nothing and reads whatever sits here
        self._eager_dev = torch.empty(eager_bytes, dtype=torch.uint8, device=self._device)
        # probe picks flag-sync (graph WAITs at the consume, host fills then signals) or launch-gating
        self._wait_sync = self._probe_wait_sync(os.getenv(_SYNC_ENV, "auto"))
        # one flag for all graphs: the readback event orders a fill after the previous graph, so signals never overlap
        self._flag = alloc_pinned_tensor(1, dtype=torch.int64)
        self._flag.zero_()
        self._token_readback = alloc_pinned_tensor(max_graph_rows, dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        sync = "wait-sync" if self._wait_sync else "launch-gating"
        logger.info_rank0(f"PLE disk backend: {self._store.io_backend()}, {sync}")

    def _probe_wait_sync(self, mode: str) -> bool:
        from freetoken.kernel import _ple_store

        if mode == "gate":
            return False
        scratch = alloc_pinned_tensor(1, dtype=torch.int64)
        scratch.zero_()
        stream = torch.cuda.current_stream(self._device)
        ok = (
            _ple_store.memop_write(stream.cuda_stream, scratch.data_ptr(), 7) == 0
            and _ple_store.memop_wait_geq(stream.cuda_stream, scratch.data_ptr(), 7) == 0
        )
        if ok:
            stream.synchronize()
            ok = int(scratch[0]) == 7
        if mode == "wait" and not ok:
            raise RuntimeError("FREETOKEN_PLE_SYNC=wait but stream memops are unavailable")
        return ok

    # ---------------- host side (engine thread, before the forward launches) ----------------

    def _filler(self) -> _Filler:
        filler = self.__dict__.get("_filler_thread")
        if filler is None:
            filler = self._filler_thread = _Filler()
        return filler

    def close(self) -> None:
        filler = self.__dict__.pop("_filler_thread", None)
        if filler is not None:
            filler.close()

    def _current_stream(self):
        return torch.cuda.current_stream(self._device)

    def _raise_failed_fill(self) -> None:
        pending = self.__dict__.pop("_pending_fill", None)
        if pending is not None and pending.done() and pending.exception() is not None:
            raise pending.exception()
        if pending is not None and not pending.done():
            self._pending_fill = pending

    def _fill_now(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        """A fill that completes before its launch, on the store's thread (after any queued graph fill)."""
        self._filler().submit(lambda: self.fill(runs, graph=graph)).result()

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        """Stage per-request token runs (two context ids, then the new tokens) in batch order."""
        pinned = self._graph_pinned if graph else self._eager_pinned
        offset = 0
        for run in runs:
            self._store.stage(run.data_ptr(), run.numel() - 2, pinned.data_ptr() + offset * self._token_bytes)
            offset += run.numel() - 2
        self._store.flush(self._flag.data_ptr() if graph and self._wait_sync else 0)

    def _ple_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.image_token_id is None:
            return input_ids
        return restore_placeholder(input_ids, self.image_token_id)

    def _ple_context(self, ids: torch.Tensor, position: int) -> list[int]:
        if self.image_token_id is None:
            return _context(ids, position, self.eos_token_id)
        return [self.image_token_id if t >= MM_PAD_SHIFT_VALUE else t for t in _context(ids, position, self.eos_token_id)]

    def host_fill_batch(self, batch: Batch, use_graph: bool) -> _PendingFill | None:
        """Stage this batch's rows before the dispatch. Under flag-sync a graph fill is only submitted:
        it completes on the filler thread, independently of the launch; returns its handle, else None."""
        self._raise_failed_fill()
        if batch.is_decode:
            reqs = list(batch.reqs)
            # the context comes from the request as it is NOW: the engine mutates it once the launch returns
            contexts = [self._ple_context(r.input_ids, r.device_len - 1) for r in reqs]
            if use_graph and self._wait_sync:
                bs = batch.padded_size
                self._token_readback[:bs].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(self._current_stream())
                cancelled = threading.Event()

                def _job() -> None:
                    try:
                        self._readback_event.synchronize()
                        tokens = self._token_readback[:bs].to(torch.int64).tolist()
                        if cancelled.is_set():
                            return
                        runs = [torch.tensor([*ctx, t], dtype=torch.int64) for ctx, t in zip(contexts, tokens)]
                        self.fill(runs, graph=True)
                    except BaseException:
                        from freetoken.kernel import _ple_store

                        # unblock the stream before surfacing; the step's output is discarded
                        _ple_store.signal_flag(self._flag.data_ptr())
                        raise

                self._pending_fill = self._filler().submit(_job)
                return _PendingFill(self, self._pending_fill, cancelled)
            # launch-gating: this D2H is the step's readback and orders the fill after sampling
            tokens = batch.input_ids.to("cpu").to(torch.int64).tolist()
            runs = [torch.tensor([*ctx, t], dtype=torch.int64) for ctx, t in zip(contexts, tokens)]
            self._fill_now(runs, graph=use_graph)
            return None
        runs = [
            torch.cat((
                torch.tensor(self._ple_context(req.input_ids, req.cached_len), dtype=torch.int64),
                self._ple_ids(req.input_ids[req.cached_len : req.device_len]).to(torch.int64),
            ))
            for req in batch.padded_reqs
        ]
        self._fill_now(runs, graph=False)
        return None

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one dispatch: stage (or submit the graph fill) on enter; cancel a submitted fill if the launch fails."""
        pending = self.host_fill_batch(batch, use_graph)
        try:
            yield
        except BaseException:
            if pending is not None:
                pending.cancel()
            raise

    # ---------------- device side (PLETableBackend protocol) ----------------

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = row_ids.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self._wait_sync:
            from freetoken.kernel import _ple_store

            _ple_store.memop_wait_reset(
                torch.cuda.current_stream(self._device).cuda_stream, self._flag.data_ptr()
            )
        pinned, dev = (
            (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned, self._eager_dev)
        )
        nbytes = rows * self._token_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        values = dev[:nbytes].view(torch.float8_e4m3fn).to(self.dtype)
        if self.scale != 1.0:
            values = values * self.scale
        values = values.view(*row_ids.shape[:-1], -1)
        if out is None:
            return values
        out.copy_(values)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None
