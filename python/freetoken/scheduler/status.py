from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable

from freetoken.core import Batch

# KVSTATS accept-len buckets: verify steps by the request's context length before the step
ACCEPT_BUCKETS = (32768, 131072, 262144)
_ACCEPT_NAMES = ("<32k", "32-128k", "128-256k", ">=256k")
REFUSAL_CAUSES = ("kv", "table", "gdn", "swa")


@dataclass
class KVStats:
    """One window of the counters behind the ``KVSTATS`` log line (KV host tier design, P0).

    Counts on TP rank 0 only (``enabled``); every rank takes the same decisions without them.
    """

    enabled: bool = True
    admits: int = 0
    walk_tokens: int = 0     # matched KV tokens before the snapshot cut
    kept_tokens: int = 0     # reused tokens after it
    cut_admits: int = 0      # admissions whose KV match went deeper than their snapshot
    cut_tokens: int = 0
    evicted_admits: int = 0  # admissions whose prompt continued into an evicted node
    evicted_tokens: int = 0  # the evicted nodes' lengths (a lower bound of the loss)
    cold_admits: int = 0     # no KV match at all
    refusals: dict = field(default_factory=dict)  # cause -> (events, set of uids)
    accept: list = field(default_factory=lambda: [[0, 0] for _ in _ACCEPT_NAMES])  # [steps, tokens]

    def note_admission(self, walk_len: int, cached_len: int, evicted_len: int) -> None:
        if not self.enabled:
            return
        self.admits += 1
        self.walk_tokens += walk_len
        self.kept_tokens += cached_len
        if walk_len > cached_len:
            self.cut_admits += 1
            self.cut_tokens += walk_len - cached_len
        if walk_len == 0:
            self.cold_admits += 1
        if evicted_len:
            self.evicted_admits += 1
            self.evicted_tokens += evicted_len

    def note_refusal(self, cause: str, uid: int) -> None:
        if not self.enabled:
            return
        events, uids = self.refusals.setdefault(cause, (0, set()))
        uids.add(uid)
        self.refusals[cause] = (events + 1, uids)

    def note_accept(self, context_len: int, committed: int) -> None:
        if not self.enabled:
            return
        bucket = self.accept[sum(context_len >= b for b in ACCEPT_BUCKETS)]
        bucket[0] += 1
        bucket[1] += committed

    def line(self, tree: Mapping[str, int] | None = None, extra: str | None = None) -> str | None:
        """The window's ``KVSTATS`` line, or None when nothing happened; starts a new window."""
        tree = dict(tree or {})
        if not (self.admits or self.refusals or any(tree.values()) or any(s for s, _ in self.accept) or extra):
            return None
        refuse = " ".join(
            f"{c}={self.refusals.get(c, (0, ()))[0]}/{len(self.refusals.get(c, (0, ()))[1])}" for c in REFUSAL_CAUSES
        )
        accept = " ".join(
            f"{name}={(tokens / steps if steps else 0):.2f}/{steps}" for name, (steps, tokens) in zip(_ACCEPT_NAMES, self.accept)
        )
        parts = [
            (
                f"KVSTATS admit={self.admits} walk={self.walk_tokens} kept={self.kept_tokens} "
                f"cut={self.cut_admits}/{self.cut_tokens} evicted={self.evicted_admits}/{self.evicted_tokens} "
                f"cold={self.cold_admits}"
            ),
            f"refuse {refuse}",
            "tree " + (" ".join(f"{k}={v}" for k, v in sorted(tree.items())) or "-"),
            f"accept {accept}",
        ]
        if extra:
            parts.append(extra)
        self.__init__(enabled=self.enabled)
        return " | ".join(parts)


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    # --moe-collect-stats: one more line per decode log line (the stats since the previous one), or None to skip
    moe_stats: Callable[[], str | None] | None = None
    # the KVSTATS line of the window (None when it is empty), at most once per kv_stats_interval seconds
    kv_stats: Callable[[], str | None] | None = None
    kv_stats_interval: float = 60.0
    _last_prefill_time: float = field(init=False)
    _last_kv_stats_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)
    # speculative steps since the last decode log: requests stepped and tokens they committed
    _spec_req_steps: int = field(default=0, init=False)
    _spec_tokens: int = field(default=0, init=False)
    # the same since the last take_spec_counts(), which the decode log leaves alone
    _spec_counts: tuple[int, int] = field(default=(0, 0), init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self._last_kv_stats_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def report_batch(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        generated: int | None = None,
    ) -> None:
        """``generated``: tokens a decode step committed; None means one per request."""
        if batch.is_prefill:
            self._report_prefill(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
            )
        else:
            self._report_decode(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                generated=generated,
            )
        if self.kv_stats is not None:
            now = self.clock()
            if now - self._last_kv_stats_time >= self.kv_stats_interval:
                self._last_kv_stats_time = now
                if line := self.kv_stats():
                    self.log(line)

    def take_spec_counts(self) -> tuple[int, int]:
        """Requests stepped and tokens they committed over the speculative decode steps since the last call."""
        counts, self._spec_counts = self._spec_counts, (0, 0)
        return counts

    def _report_prefill(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        now = self.clock()
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        # Read the schedule-time snapshot: by report time the forward's complete_one() has
        # advanced each req's cached_len to device_len, so reading the reqs here would log
        # decode-state values (#new-token == #reqs, #cached-token == full prompt).
        new_tokens = batch.log_new_tokens
        cached_tokens = batch.log_cached_tokens
        input_throughput = new_tokens / gap if gap > 0 else 0.0
        self.log(
            f"Prefill batch, "
            f"#new-seq: {len(batch.reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"input throughput (token/s): {input_throughput:.2f}"
        )

    def _report_decode(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        generated: int | None = None,
    ) -> None:
        self._decode_forward_count += 1
        self._decode_generated_tokens += len(batch.reqs) if generated is None else generated
        if generated is not None:
            self._spec_req_steps += len(batch.reqs)
            self._spec_tokens += generated
            steps, tokens = self._spec_counts
            self._spec_counts = (steps + len(batch.reqs), tokens + generated)
        if self._decode_forward_count % self.decode_log_interval != 0:
            return

        now = self.clock()
        gap = now - self._last_decode_time
        self._last_decode_time = now
        gen_throughput = self._decode_generated_tokens / gap if gap > 0 else 0.0
        self._decode_generated_tokens = 0
        accept = f"accept len: {self._spec_tokens / self._spec_req_steps:.2f}, " if self._spec_req_steps else ""
        self._spec_req_steps = self._spec_tokens = 0
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"{accept}"
            f"#queue-req: {queue_reqs}"
        )
        if self.moe_stats is not None and (line := self.moe_stats()):
            self.log(line)


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


def _mamba_msg(mamba_slots: tuple[int, int] | None) -> str:
    """GDN-state (mamba) pool occupancy for hybrid models; empty for the rest."""
    if mamba_slots is None:
        return ""
    used, total = mamba_slots
    return f"#mamba-slot: {used}/{total}, mamba usage: {_usage_ratio(used, total):.2f}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window (swa) pool occupancy for SWA models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"#swa-token: {used}/{total}, swa usage: {_usage_ratio(used, total):.2f}, "
