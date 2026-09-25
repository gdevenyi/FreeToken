"""Load-time requantization of a checkpoint's bf16 dense projections (--dense-quant, --lm-head-quant).

A card without the VRAM for a model's bf16 projections can still hold them as MXFP8: e4m3
elements with one power-of-two scale per 32 inputs, served W8A16 by the triton kernels that
already decode e4m3 in software below sm_89. ``LoadTimeQuantConfig`` wraps the checkpoint's own
QuantConfig so the model builder allocates MXFP8 buffers for the chosen modules and the weight
reader, which sees the same config, quantizes their bf16 tensors as they stream in.
"""

from __future__ import annotations

import re
from typing import ClassVar

import torch

from ..scheme import MX_GROUP, QuantKind, QuantScheme, mxfp8_scheme
from .base import QuantConfig, Stored

_E4M3_MAX = 448.0
_E8M0_BIAS = 127


class LoadTimeQuantConfig(QuantConfig):
    """The checkpoint's QuantConfig, plus MXFP8 for bf16 modules whose checkpoint name matches a target."""

    dialect = "load-time"
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {}

    def __init__(self, base: QuantConfig, targets: tuple[str, ...]) -> None:
        super().__init__(base.name_map)
        self.base = base
        self.unquantized = base.unquantized
        self.targets = tuple(re.compile(t) for t in targets)
        self.scheme = mxfp8_scheme()
        self.STORAGE = {**base.STORAGE, QuantKind.MXFP8: {"weight": "weight", "weight_scale_inv": "weight_scale_inv"}}

    def _targeted(self, name: str) -> bool:
        return any(t.search(name) for t in self.targets)

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        scheme = self.base.scheme_for_name(name)
        if scheme is not None or not self._targeted(name):
            return scheme
        return self.scheme

    def quantize_at_load(self, prefix: str) -> bool:
        """Whether the bf16 tensor of model module ``prefix`` is to be requantized while it loads."""
        names = self.name_map.to_checkpoint(prefix)
        return all(
            self.base.scheme_for_name(n) is None and not self.unquantized(n) and self._targeted(n)
            for n in names
        )


def quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N, K]`` float weight -> (``[N, K]`` e4m3, ``[N, K // 32]`` uint8 e8m0 codes).

    Each 32-wide block takes the smallest power-of-two scale that maps its amax onto e4m3's
    finite range, so no element saturates (OCP's floor rule clips the block max by up to 12.5%).
    """
    n, k = weight.shape
    if k % MX_GROUP:
        raise ValueError(f"mxfp8 needs K divisible by {MX_GROUP}, got {k}")
    q = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=weight.device)
    codes = torch.empty(n, k // MX_GROUP, dtype=torch.uint8, device=weight.device)
    # row chunks bound the fp32 transient (a whole 248K-row lm_head would need 2.4 GiB)
    rows = max(1, (32 << 20) // (4 * k))
    for r in range(0, n, rows):
        blocks = weight[r : r + rows].float().reshape(-1, k // MX_GROUP, MX_GROUP)
        amax = blocks.abs().amax(dim=-1)
        mant, exp = torch.frexp(amax / _E4M3_MAX)
        # ceil(log2(x)) from x = mant * 2**exp with mant in [0.5, 1): exact at powers of two
        log2 = exp - (mant == 0.5).to(exp.dtype)
        code = torch.where(amax > 0, log2 + _E8M0_BIAS, torch.zeros_like(log2)).clamp_(0, 254)
        scale = torch.ldexp(torch.ones_like(amax), code - _E8M0_BIAS)
        blocks = (blocks / scale.unsqueeze(-1)).clamp_(-_E4M3_MAX, _E4M3_MAX)
        q[r : r + rows] = blocks.reshape(-1, k).to(torch.float8_e4m3fn)
        codes[r : r + rows] = code.to(torch.uint8)
    return q, codes


__all__ = ["LoadTimeQuantConfig", "quantize_mxfp8"]
