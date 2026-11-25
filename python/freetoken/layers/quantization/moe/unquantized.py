"""bf16 experts: the fused Triton MoE over gate_up / down banks."""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import (
    BankSpec,
    ExpertView,
    MoEConfig,
    MoEKernel,
    MoEMethod,
    fused_piece,
    gated_epilogue_reason,
    is_resident,
    limit_or_inf,
)


class FusedMoEKernel(MoEKernel):
    name = "fused"
    cpu_format = "bf16"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = gated_epilogue_reason(cfg)
        return f"bf16 fused MoE: {reason}" if reason else None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i = cfg.local_intermediate
        # resident experts compute in the model dtype; offload banks stay bf16, the FTW and CPU executor format
        dt = cfg.dtype if cfg.strategy == "resident" else torch.bfloat16
        return {
            "gate_up": BankSpec((2 * i, cfg.hidden), dt),
            "down": BankSpec((cfg.hidden, i), dt),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
        out["down"].copy_(pieces["down"])
        return {}

    def apply(
        self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool
    ):
        from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

        extra = {"has_inactive": True} if getattr(layer, "ep_size", 1) > 1 else {}
        # the resident layer runs the in-place prefill kernel for both phases
        if is_prefill or is_resident(layer):
            return fused_experts_impl(
                x,
                view.tensors["gate_up"],
                view.tensors["down"],
                topk_weights,
                topk_ids,
                layer.activation,
                layer.apply_router_weight_on_input,
                float(layer.alpha),
                limit_or_inf(layer),
                **extra,
            )
        return fused_experts_decode_impl(
            x,
            view.tensors["gate_up"],
            view.tensors["down"],
            topk_weights,
            topk_ids,
            layer.activation,
            layer.apply_router_weight_on_input,
            float(layer.alpha),
            limit_or_inf(layer),
        )


@register_method(QuantKind.NONE, LayerKind.MOE)
class UnquantizedMoEMethod(MoEMethod):
    candidates = (FusedMoEKernel,)
