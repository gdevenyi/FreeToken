from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


@dataclass
class PcieSysmemDistributedImpl(DistributedImpl):
    """Two-tier pinned-sysmem PCIe TP=2 AllReduce with PyNCCL fallback for AllGather / oversized tensors."""

    pcie_comm: Any
    fallback: DistributedImpl
    max_bytes: int

    _SUPPORTED_DTYPES: ClassVar[tuple[torch.dtype, ...]] = (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.int32,
    )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if (
            x.is_cuda
            and x.is_contiguous()
            and (x.data_ptr() & 15) == 0
            and x.dtype in self._SUPPORTED_DTYPES
            and x.nbytes <= self.max_bytes
        ):
            self.pcie_comm.all_reduce(x.view(-1))
            return x
        return self.fallback.all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.fallback.all_gather(x)


class DistributedCommunicator:
    plugins: ClassVar[list[DistributedImpl]] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_bytes: int,
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism
    (and optional pinned-sysmem PCIe TP=2 AllReduce when FREETOKEN_PCIE_ALLREDUCE=1).
    """
    if tp_info.size == 1:
        return
    from freetoken.kernel import (
        init_pcie_allreduce,
        init_pynccl,
        pcie_allreduce_enabled,
    )

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )
    pynccl_impl = PyNCCLDistributedImpl(comm)
    DistributedCommunicator.plugins.append(pynccl_impl)

    if tp_info.size == 2 and pcie_allreduce_enabled():
        pcie_comm = init_pcie_allreduce(
            tp_rank=tp_info.rank,
            tp_size=tp_info.size,
            tp_cpu_group=tp_cpu_group,
        )
        if pcie_comm is not None:
            DistributedCommunicator.plugins.append(
                PcieSysmemDistributedImpl(
                    pcie_comm=pcie_comm,
                    fallback=pynccl_impl,
                    max_bytes=pcie_comm.max_bytes,
                )
            )


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    from .info import set_tp_cpu_group

    for plugin in reversed(DistributedCommunicator.plugins):
        pcie_comm = getattr(plugin, "pcie_comm", None)
        if pcie_comm is not None and hasattr(pcie_comm, "close"):
            pcie_comm.close()
    DistributedCommunicator.plugins = [TorchDistributedImpl()]
    set_tp_cpu_group(None)
