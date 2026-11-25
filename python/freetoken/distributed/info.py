from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None
_TP_CPU_GROUP = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def set_tp_cpu_group(group) -> None:
    global _TP_CPU_GROUP
    _TP_CPU_GROUP = group


def try_get_tp_cpu_group():
    return _TP_CPU_GROUP


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


__all__ = [
    "DistributedInfo",
    "get_tp_info",
    "set_tp_cpu_group",
    "set_tp_info",
    "try_get_tp_cpu_group",
    "try_get_tp_info",
]
