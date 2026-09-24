"""Opt-in post-load FP8 for the qwen4_exp projections the dense FP8 path leaves in bf16.

At batch size 1 these skinny GEMMs run at ~600 GB/s on the 5080, so halving their weight
bytes halves their time: the shared expert (``FREETOKEN_FP8_SHARED_EXPERT=1``, ~0.4 ms/token)
and the hyper-connection mixes (``FREETOKEN_FP8_HC=1``, ~1 ms/token, two per layer). They are
quantized after ``load_state_dict`` from the loaded bf16 weights, with the same per-tensor
quantizer as the dense path, so HF and FTW checkpoints alike need no layout change. TP=1 only.
The router stays bf16: routing decisions are the precision-sensitive part.
"""

from __future__ import annotations

import os

import torch
from freetoken.distributed import get_tp_info
from freetoken.layers.base import BaseOP
from freetoken.layers.fp8_dynamic import Fp8DynamicLinear
from freetoken.models.config import _ENV_TRUE
from freetoken.utils import init_logger

logger = init_logger(__name__)


def fp8_shared_expert_enabled() -> bool:
    return os.getenv("FREETOKEN_FP8_SHARED_EXPERT", "0").strip().lower() in _ENV_TRUE


def fp8_hc_enabled() -> bool:
    return os.getenv("FREETOKEN_FP8_HC", "0").strip().lower() in _ENV_TRUE


def _fp8_from(linear: BaseOP) -> Fp8DynamicLinear:
    from .weight import _quantize_per_tensor

    weight = linear.weight
    out_features, in_features = weight.shape
    op = Fp8DynamicLinear(in_features, out_features)
    op.weight, op.weight_scale = _quantize_per_tensor(weight)
    return op


def _walk(root: BaseOP):
    seen: set[int] = set()
    stack = [root]
    while stack:
        op = stack.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        yield op
        for value in vars(op).values():
            if isinstance(value, BaseOP):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(v for v in value if isinstance(v, BaseOP))


def quantize_after_load(root: BaseOP) -> int:
    """Swap the opted-in bf16 projections under ``root`` for FP8 ones; returns how many."""
    shared, hc = fp8_shared_expert_enabled(), fp8_hc_enabled()
    if not (shared or hc):
        return 0
    if get_tp_info().size > 1:
        logger.warning("FREETOKEN_FP8_SHARED_EXPERT / FREETOKEN_FP8_HC are TP=1 only; ignored")
        return 0
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert

    from .hc import GatedResidual

    swapped = 0
    for op in list(_walk(root)):
        if shared and isinstance(op, _SharedExpert):
            names = ("gate_up_proj", "down_proj")
        elif hc and isinstance(op, GatedResidual):
            names = ("input_mix_weight_down_block_inject", "input_mix_weight_down", "input_mix_weight_up")
        else:
            continue
        for name in names:
            linear = getattr(op, name, None)
            if linear is None or getattr(linear, "weight", None) is None or linear.weight.dtype != torch.bfloat16:
                continue
            setattr(op, name, _fp8_from(linear))
            swapped += 1
    if swapped:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # the dropped bf16 weights are what the cache planner sees
        logger.info_rank0(
            f"post-load FP8: {swapped} projections (shared expert={shared}, hyper-connections={hc})"
        )
    return swapped


__all__ = ["fp8_hc_enabled", "fp8_shared_expert_enabled", "quantize_after_load"]
