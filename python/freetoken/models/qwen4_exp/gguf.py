"""Qwen3.8-Flash-Next from a llama.cpp ``qwen4exp`` GGUF: config from the GGUF metadata, tensor names, and the converter's transforms undone.

The llama.cpp converter (conversion/qwen.py, conversion/qwen4exp.py) stores every zero-centred norm as ``w + 1``, ``A_log`` as ``-exp(A_log)``, the squeezed conv kernels, the indexer projection split into q and k, and the GDN value heads in tiled order (k-head minor). Row reorders are undone on the packed bytes; ``ssm_out`` keeps its tiled columns and the GDN layer permutes its input instead (``Qwen4ExpArgs.gdn_out_heads_tiled``).

With :func:`gguf_float_dense` the dense projections load dequantized to bf16 in the safetensors reader's layout (``ssm_out`` columns untiled too) and go through its TP shard and load-time FP8 (``weight.dense_emitter``); only the routed experts and the PLE table stay packed ggml. TP > 1 needs it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from freetoken.layers.quantization.ggml_types import is_quantized
from freetoken.models.gguf.mmproj import find_mmproj, log_mmproj
from freetoken.models.gguf.reader import GgufTensorInfo, gguf_tensor_index
from freetoken.models.gguf.tensors import (
    check_scales,
    dequantized_tensor,
    float_tensor,
    fused,
    matrix,
    weight_scale,
)
from freetoken.models.qwen3_vl.gguf import gguf_mm_config, iter_mmproj_weights

from .config import parse_config

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen4exp"

# HF config fields the GGUF metadata does not carry (Qwen3.8-Flash-Next's own config.json values)
_FAMILY_DEFAULTS = dict(
    hidden_act="silu",
    output_gate_type="sigmoid",
    indexer_kv_heads=1,
    ngram_vocab_size_base=20000000,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=128,
    norm_topk_prob=True,
)

_LAYER_LEAVES: dict[str, tuple[str, ...]] = {
    "linear_attn.in_proj_qkv": ("attn_qkv",),
    "linear_attn.in_proj_z": ("attn_gate",),
    "linear_attn.in_proj_b": ("ssm_beta",),
    "linear_attn.in_proj_a": ("ssm_alpha",),
    "linear_attn.out_proj": ("ssm_out",),
    "self_attn.q_proj": ("attn_q",),
    "self_attn.k_proj": ("attn_k",),
    "self_attn.v_proj": ("attn_v",),
    "self_attn.o_proj": ("attn_output",),
    "self_attn.indexer.index_q_proj": ("indexer.q_proj",),
    "self_attn.indexer.index_k_proj": ("indexer.k_proj",),
    "mlp.gate": ("ffn_gate_inp",),
    "mlp.shared_expert_gate": ("ffn_gate_inp_shexp",),
    "mlp.shared_expert.gate_proj": ("ffn_gate_shexp",),
    "mlp.shared_expert.up_proj": ("ffn_up_shexp",),
    "mlp.shared_expert.down_proj": ("ffn_down_shexp",),
    "mlp.experts.0.gate_proj": ("ffn_gate_exps",),
    "mlp.experts.0.up_proj": ("ffn_up_exps",),
    "mlp.experts.0.down_proj": ("ffn_down_exps",),
    "attn_hyper_connection.input_mix_weight_down": ("hc_attn_down",),
    "attn_hyper_connection.block_inject_weight": ("hc_attn_inject",),
    "attn_hyper_connection.input_mix_weight_up": ("hc_attn_up",),
    "mlp_hyper_connection.input_mix_weight_down": ("hc_ffn_down",),
    "mlp_hyper_connection.block_inject_weight": ("hc_ffn_inject",),
    "mlp_hyper_connection.input_mix_weight_up": ("hc_ffn_up",),
    "ple.key_proj": ("ple_key",),
    "ple.value_proj": ("ple_value",),
}
_TOP_LEAVES: dict[str, tuple[str, ...]] = {
    "model.language_model.embed_tokens": ("token_embd",),
    "lm_head": ("output",),
    "model.language_model.hyper_connection_mixer.input_mix_weight_down": (
        "output_hc_down",
    ),
    "model.language_model.hyper_connection_mixer.input_mix_weight_up": (
        "output_hc_up",
    ),
}
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")


def to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """Checkpoint module name -> the GGUF tensors (without ``.weight``) that store it; empty for modules the GGUF has no matrix for."""
    if name in _TOP_LEAVES:
        return _TOP_LEAVES[name]
    m = _LAYER_RE.match(name)
    if m is None:
        return ()
    return tuple(
        f"blk.{m.group(1)}.{leaf}" for leaf in _LAYER_LEAVES.get(m.group(2), ())
    )


# checkpoint-name globs of every dense module: all but the routed experts and the PLE table (ple_embedding)
_FLOAT_DENSE_MODULES = (
    "*embed_tokens",
    "lm_head",
    "*hyper_connection*",
    "*.self_attn",
    "*.linear_attn",
    "*.mlp.gate",
    "*.shared_expert",
    "*.shared_expert_gate",
    "*.ple.key_proj",
    "*.ple.value_proj",
)


def gguf_float_dense() -> bool:
    """Whether the dense projections load dequantized to bf16 instead of as packed ggml rows (FREETOKEN_GGUF_FLOAT_DENSE=1):
    the TP shard runs on float rows, so TP > 1 needs it, and FREETOKEN_FP8_DENSE / _LMHEAD then requantize on top as for a
    bf16 checkpoint. Without it the file's own ggml types win, FP8 switches included. An env switch, not the TP size, so the
    builder (QuantConfig, parsed before TP is set) and the reader always get one answer."""
    return os.getenv("FREETOKEN_GGUF_FLOAT_DENSE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def gguf_float_modules() -> tuple[str, ...]:
    """``ModelSpec.gguf_float_modules``: the modules the builder makes float, so the GGUF QuantConfig gives them no scheme."""
    return _FLOAT_DENSE_MODULES if gguf_float_dense() else ()


def parse_gguf_config(shim: GgufConfigShim) -> ModelConfig:
    check_scales(shim.model_path)
    # a metadata-only GGUF (an FTW dir's source_metadata.gguf) has no tensor table to check
    index = gguf_tensor_index(shim.model_path)
    # the PLE row sources read no weight scale, so refuse one instead of serving the table unscaled
    if "per_layer_token_embd.scale" in index:
        raise NotImplementedError(
            f"{shim.model_path}: a PLE table with a weight scale (per_layer_token_embd.scale) is not supported"
        )
    if index and "blk.0.hc_attn_down.weight" not in index:
        raise NotImplementedError(
            f"{shim.model_path}: no decoder layer tensors (blk.0.hc_attn_down), e.g. an MTP-only draft GGUF"
        )
    md = shim.metadata

    def kv(key: str):
        return md[f"{_ARCH}.{key}"]

    head_dim = int(kv("attention.key_length"))
    sections = [int(s) for s in kv("rope.dimension_sections")][:3]
    ratios = [int(r) for r in kv("attention.compress_ratios")]
    n_v = int(kv("ssm.time_step_rank"))
    text = SimpleNamespace(
        **_FAMILY_DEFAULTS,
        hidden_size=int(kv("embedding_length")),
        # block_count also counts the MTP layers a GGUF may carry after the decoder
        num_hidden_layers=int(kv("block_count"))
        - int(md.get(f"{_ARCH}.nextn_predict_layers", 0)),
        num_attention_heads=int(kv("attention.head_count")),
        num_key_value_heads=int(kv("attention.head_count_kv")),
        head_dim=head_dim,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(kv("rope.freq_base")),
            "partial_rotary_factor": int(kv("rope.dimension_count")) / head_dim,
            "mrope_section": sections,
            "mrope_interleaved": True,
        },
        max_position_embeddings=int(kv("context_length")),
        full_attention_interval=int(kv("full_attention_interval")),
        rms_norm_eps=float(kv("attention.layer_norm_rms_epsilon")),
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        num_experts=int(kv("expert_count")),
        num_experts_per_tok=int(kv("expert_used_count")),
        moe_intermediate_size=int(kv("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(kv("expert_shared_feed_forward_length")),
        linear_num_key_heads=int(kv("ssm.group_count")),
        linear_num_value_heads=n_v,
        linear_key_head_dim=int(kv("ssm.state_size")),
        linear_value_head_dim=int(kv("ssm.inner_size")) // n_v,
        linear_conv_kernel_dim=int(kv("ssm.conv_kernel")),
        hc_count=int(kv("hyper_connection.count")),
        hc_lowrank=int(kv("hyper_connection.low_rank")),
        indexer_n_heads=int(kv("attention.indexer.head_count")),
        indexer_head_dim=int(kv("attention.indexer.key_length")),
        indexer_budget=int(kv("attention.indexer.top_k")),
        indexer_compress_ratio=max(ratios),
        ple_layer_ids=[int(i) + 1 for i in kv("ple.layers")],
        # the GGUF key is one n-gram head's row; HF counts all (ngram_size - 1) * heads_per_ngram heads
        ple_embed_dim=int(kv("embedding_length_per_layer_input"))
        * (int(kv("ple.ngram_size")) - 1)
        * int(kv("ple.heads_per_ngram")),
        ple_conv_kernel_size=int(kv("ple.conv_kernel")),
        ngram_size=int(kv("ple.ngram_size")),
        heads_per_ngram=int(kv("ple.heads_per_ngram")),
        eos_token_id=int(kv("ple.eos_token_id")),
    )
    # with an mmproj, the id the image processor places; ple.image_token_id names the same token for text-only files
    image_token_id = getattr(shim, "image_token_id", None)
    if image_token_id is None:
        image_token_id = md.get(f"{_ARCH}.ple.image_token_id")
    hf_like = SimpleNamespace(
        text_config=text,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        image_token_id=None if image_token_id is None else int(image_token_id),
        vision_config=getattr(shim, "vision_config", None),
    )
    config = parse_config(hf_like)
    float_dense = gguf_float_dense()
    tiled = not float_dense  # the float reader untiles ssm_out's columns
    return replace(
        config,
        attn_quant=config.attn_quant if float_dense else "none",
        expert_quant="gguf",
        qwen4_args=replace(config.qwen4_args, gdn_out_heads_tiled=tiled),
    )


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------


def _untile(
    t: torch.Tensor, num_k_heads: int, head_dim: int, dim: int = 0
) -> torch.Tensor:
    """Tiled value heads (k-head minor) back to grouped (k-head major) along ``dim``."""
    n = t.shape[dim]
    r = n // (num_k_heads * head_dim)
    idx = torch.arange(n).view(r, num_k_heads, head_dim).transpose(0, 1).reshape(n)
    return t.index_select(dim, idx)


def iter_gguf_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The dense state dict from the GGUF; the routed experts load as offload banks (``freetoken.models.gguf.experts``)."""
    if not include_non_moe:
        return
    from freetoken.distributed import try_get_tp_info
    from freetoken.layers.quantization import get_quant_config
    from freetoken.utils import cached_load_hf_config

    config = parse_gguf_config(cached_load_hf_config(model_path))
    tp = try_get_tp_info()
    rank, world = (0, 1) if tp is None else (tp.rank, tp.size)
    float_dense = gguf_float_dense()
    if world > 1 and not float_dense:
        raise ValueError(
            "a qwen4exp GGUF at --tp-size > 1 shards its dense projections as floats: set FREETOKEN_GGUF_FLOAT_DENSE=1 "
            "(bf16; add FREETOKEN_FP8_DENSE=1 / FREETOKEN_FP8_LMHEAD=1 for load-time FP8, as the safetensors path)"
        )
    if float_dense:
        from .weight import dense_emitter

        emit, _, _ = dense_emitter(
            replace(config, quant=get_quant_config()), rank, world
        )
    else:

        def emit(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
            yield name, tensor

    for name, tensor in _iter_gguf_dense(
        model_path, config, float_dense, include_vision
    ):
        yield from emit(name, tensor)


class _DenseReader:
    """Tensor readers over one GGUF index: packed ggml rows per ggml-type run, or with ``float_dense`` bf16 in the
    safetensors layout. Shared by the trunk (``_iter_gguf_dense``) and the MTP sidecar (``iter_sidecar_weights``)."""

    def __init__(
        self, index: dict[str, GgufTensorInfo], config: ModelConfig, float_dense: bool
    ) -> None:
        self.index, self.config, self.float_dense = index, config, float_dense
        self.hidden = config.hidden_size

    def t(self, name: str) -> GgufTensorInfo:
        return self.index[name if name in self.index else f"{name}.weight"]

    def norm(self, name: str) -> torch.Tensor:
        return float_tensor(self.t(name)).float() - 1.0

    def mat(self, info: GgufTensorInfo) -> torch.Tensor:
        if not (self.float_dense and is_quantized(info.ggml_type)):
            return matrix(info)
        w = dequantized_tensor(info)
        # an NVFP4 tensor's weight_scale_2 (``.scale``), which the packed path applies per output row in the GEMM
        if (s := weight_scale(info)) is not None:
            if s.numel() == 1:
                w = w * s
            elif s.numel() == w.shape[0]:
                w = w * s.unsqueeze(1)
            else:
                raise NotImplementedError(
                    f"{info.name}: {s.numel()} weight scales for {w.shape[0]} rows"
                )
        return w.to(torch.bfloat16)

    def fz(
        self,
        prefix: str,
        infos: list[GgufTensorInfo],
        parts: list[torch.Tensor],
        k: int,
        pad_to: int = 0,
    ):
        # no infos: fused takes every part as a float row block
        return fused(prefix, [] if self.float_dense else infos, parts, k, pad_to=pad_to)

    def hc(self, prefix: str, src: str) -> Iterator[tuple[str, torch.Tensor]]:
        t, mat, args = self.t, self.mat, self.config.qwen4_args
        down, inject, up = (t(f"{src}_{p}") for p in ("down", "inject", "up"))
        yield from self.fz(
            f"{prefix}.input_mix_weight_down_block_inject",
            [down, inject],
            [mat(down), mat(inject)],
            self.hidden * args.hc_count,
            pad_to=16,
        )
        yield from self.fz(
            f"{prefix}.input_mix_weight_up", [up], [mat(up)], args.hc_lowrank
        )
        yield f"{prefix}.hc_norm.weight", self.norm(f"{src}_norm")

    def attention(self, blk: str, sa: str) -> Iterator[tuple[str, torch.Tensor]]:
        t, mat, norm = self.t, self.mat, self.norm
        parts = [t(f"{blk}.attn_{p}") for p in ("q", "k", "v")]
        yield from self.fz(
            f"{sa}.qkv_proj", parts, [mat(p) for p in parts], self.hidden
        )
        out = t(f"{blk}.attn_output")
        yield from self.fz(f"{sa}.o_proj", [out], [mat(out)], int(out.ne[0]))
        yield f"{sa}.q_norm.weight", norm(f"{blk}.attn_q_norm")
        yield f"{sa}.k_norm.weight", norm(f"{blk}.attn_k_norm")
        iq, ik = t(f"{blk}.indexer.q_proj"), t(f"{blk}.indexer.k_proj")
        yield from self.fz(
            f"{sa}.indexer.index_qk_proj", [iq, ik], [mat(iq), mat(ik)], self.hidden
        )
        yield f"{sa}.indexer.q_layernorm.weight", norm(f"{blk}.indexer.q_norm")
        yield f"{sa}.indexer.k_layernorm.weight", norm(f"{blk}.indexer.k_norm")

    def mlp_and_hc(self, blk: str, pre: str) -> Iterator[tuple[str, torch.Tensor]]:
        t, mat = self.t, self.mat
        mlp = f"{pre}.mlp"
        yield f"{mlp}.gate.weight", float_tensor(t(f"{blk}.ffn_gate_inp"))
        sg, su, sd = (t(f"{blk}.ffn_{p}_shexp") for p in ("gate", "up", "down"))
        yield from self.fz(
            f"{mlp}.shared_expert.gate_up_proj",
            [sg, su],
            [mat(sg), mat(su)],
            self.hidden,
        )
        yield from self.fz(
            f"{mlp}.shared_expert.down_proj",
            [sd],
            [mat(sd)],
            self.config.shared_expert_intermediate_size,
        )
        yield (
            f"{mlp}.shared_expert_gate.weight",
            float_tensor(t(f"{blk}.ffn_gate_inp_shexp")).reshape(1, -1),
        )
        yield from self.hc(f"{pre}.attn_hyper_connection", f"{blk}.hc_attn")
        yield from self.hc(f"{pre}.mlp_hyper_connection", f"{blk}.hc_ffn")


def _iter_gguf_dense(
    model_path: str, config: ModelConfig, float_dense: bool, include_vision: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    """Unsharded dense tensors: packed ggml rows per ggml-type run, or with ``float_dense`` bf16 in the safetensors layout."""
    from freetoken.utils import cached_load_hf_config

    vision = include_vision and config.is_multimodal
    log_mmproj(model_path, vision)
    md = cached_load_hf_config(model_path).metadata
    g = config.linear_attention_group()
    hidden = config.hidden_size
    r = _DenseReader(gguf_tensor_index(model_path), config, float_dense)
    t, norm, mat, fz = r.t, r.norm, r.mat, r.fz

    yield from fz(
        "model.embed_tokens", [t("token_embd")], [mat(t("token_embd"))], hidden
    )
    yield from fz("lm_head", [t("output")], [mat(t("output"))], hidden)
    for proj, k in (
        ("down", hidden * config.qwen4_args.hc_count),
        ("up", config.qwen4_args.hc_lowrank),
    ):
        info = t(f"output_hc_{proj}")
        yield from fz(
            f"model.hyper_connection_mixer.input_mix_weight_{proj}",
            [info],
            [mat(info)],
            k,
        )
    yield "model.hyper_connection_mixer.hc_norm.weight", norm("output_hc_norm")

    for layer in range(config.num_layers):
        blk, pre = f"blk.{layer}", f"model.layers.{layer}"
        if config.is_linear_layer(layer):
            la = f"{pre}.linear_attn"
            nk, dk = g.num_key_heads, g.key_head_dim
            qkv, gate = t(f"{blk}.attn_qkv"), t(f"{blk}.attn_gate")
            qkv_rows = mat(qkv)
            v_start = 2 * nk * dk
            qkv_rows = torch.cat(
                [qkv_rows[:v_start], _untile(qkv_rows[v_start:], nk, g.value_head_dim)]
            )
            z_rows = _untile(mat(gate), nk, g.value_head_dim)
            beta_info, alpha_info = t(f"{blk}.ssm_beta"), t(f"{blk}.ssm_alpha")
            beta, alpha = (_untile(mat(i), nk, 1) for i in (beta_info, alpha_info))
            if not float_dense and (
                is_quantized(qkv.ggml_type) or is_quantized(gate.ggml_type)
            ):
                yield from fused(
                    f"{la}.in_proj_qkvz", [qkv, gate], [qkv_rows, z_rows], hidden
                )
                yield from fused(
                    f"{la}.in_proj_ba", [beta_info, alpha_info], [beta, alpha], hidden
                )
            else:
                yield from fz(
                    f"{la}.in_proj",
                    [qkv, gate, beta_info, alpha_info],
                    [qkv_rows, z_rows, beta, alpha],
                    hidden,
                )
            conv = float_tensor(t(f"{blk}.ssm_conv1d")).float()
            conv = torch.cat(
                [conv[:v_start], _untile(conv[v_start:], nk, g.value_head_dim)]
            )
            yield f"{la}.conv1d.weight", conv.unsqueeze(1)
            yield (
                f"{la}.dt_bias",
                _untile(float_tensor(t(f"{blk}.ssm_dt.bias")).float(), nk, 1),
            )
            yield (
                f"{la}.A_log",
                _untile(torch.log(-float_tensor(t(f"{blk}.ssm_a")).float()), nk, 1),
            )
            yield f"{la}.norm.weight", float_tensor(t(f"{blk}.ssm_norm"))
            out_info = t(f"{blk}.ssm_out")
            out = mat(out_info)
            if float_dense:
                # the row-parallel shard and the untiled GDN forward both want grouped value-head columns
                out = _untile(out, nk, g.value_head_dim, dim=1)
            yield from fz(
                f"{la}.out_proj",
                [out_info],
                [out],
                g.num_value_heads * g.value_head_dim,
            )
        else:
            yield from r.attention(blk, f"{pre}.self_attn")
        yield from r.mlp_and_hc(blk, pre)

        if layer in config.qwen4_args.ple_layer_ids:
            ple = f"{pre}.ple"
            for proj in ("key", "value"):
                info = t(f"{blk}.ple_{proj}")
                yield from fz(
                    f"{ple}.{proj}_proj",
                    [info],
                    [mat(info)],
                    config.qwen4_args.ple_embed_dim,
                )
            for n in ("key", "query", "conv"):
                yield f"{ple}.norm_{n}.weight", norm(f"{blk}.ple_norm_{n}")
            yield (
                f"{ple}.conv1d.weight",
                float_tensor(t(f"{blk}.ple_conv1d")).float().unsqueeze(1),
            )
            yield (
                f"{ple}.ple_embedding.layer_multipliers",
                torch.tensor(
                    [int(x) for x in md[f"{_ARCH}.ple.layer_multipliers"]],
                    dtype=torch.int64,
                ),
            )
            yield (
                f"{ple}.ple_embedding.ngram_heads_vocab_sizes",
                torch.tensor(
                    [int(x) for x in md[f"{_ARCH}.ple.head_vocab_sizes"]],
                    dtype=torch.int64,
                ),
            )
            yield (
                f"{ple}.ple_embedding.ngram_heads_offsets",
                torch.tensor(
                    [int(x) for x in md[f"{_ARCH}.ple.head_offsets"]], dtype=torch.int64
                ),
            )

    if vision:
        yield from iter_mmproj_weights(find_mmproj(model_path))


# ----------------------------------------------------------------------------------------------
# MTP sidecar: llama.cpp's nextn block (Unsloth mtp-*.gguf)
# ----------------------------------------------------------------------------------------------

# the draft's dense modules, all float (bf16, or load-time FP8 through dense_emitter); only its routed experts stay packed
_SIDECAR_FLOAT_MODULES = (
    "mtp.fc_embedding",
    "mtp.fc_hidden",
    "mtp.hyper_connection_mixer",
    "*.self_attn",
    "*hyper_connection*",
    "*.mlp.gate",
    "*.shared_expert",
    "*.shared_expert_gate",
)


def sidecar_block(model_path: str) -> int | None:
    """The ``blk.N`` of the nextn block a GGUF carries, or None."""
    for name in gguf_tensor_index(model_path):
        if name.endswith(".nextn.eh_proj.weight"):
            return int(name.split(".")[1])
    return None


def sidecar_to_gguf(name: str, tensors: dict[str, int]) -> tuple[str, ...]:
    """A draft module's checkpoint name (``mtp.*``) -> the sidecar tensors that store it."""
    blk = next(
        n.split(".nextn.")[0] for n in tensors if n.endswith(".nextn.eh_proj.weight")
    )
    if name in ("mtp.fc_embedding", "mtp.fc_hidden"):
        return (f"{blk}.nextn.eh_proj",)
    if name.startswith("mtp.hyper_connection_mixer.input_mix_weight_"):
        return (f"{blk}.nextn.hc_head_{name.rsplit('_', 1)[1]}",)
    leaf = name.removeprefix("mtp.layers.0.")
    return tuple(f"{blk}.{x}" for x in _LAYER_LEAVES.get(leaf, ()))


def sidecar_quant(model_path: str):
    """The draft's QuantConfig: the sidecar's ggml types for its routed experts, every dense module float."""
    from freetoken.layers.quantization import GGUFConfig, NameMap
    from freetoken.models.register import get_model_spec

    packed = get_model_spec("Qwen4ExpForConditionalGeneration").packed_modules_mapping
    types = {
        name: info.ggml_type for name, info in gguf_tensor_index(model_path).items()
    }
    return GGUFConfig(
        types,
        sidecar_to_gguf,
        name_map=NameMap(packed=packed),
        unquantized=_SIDECAR_FLOAT_MODULES,
    )


def iter_sidecar_weights(
    model_path: str, draft_config: ModelConfig
) -> Iterator[tuple[str, torch.Tensor]]:
    """The draft's dense tensors from a nextn block, unsharded, under their checkpoint names (``mtp.*``), in bf16.

    ``eh_proj`` is ``[fc_embedding | fc_hidden]`` along its input (embedding first, as llama.cpp concatenates
    ``enorm(e)`` and ``hnorm(h)``); ``hc_head_*`` is the draft's final mixer. Norms are stored ``w + 1``."""
    blk = f"blk.{sidecar_block(model_path)}"
    r = _DenseReader(gguf_tensor_index(model_path), draft_config, float_dense=True)
    hidden = draft_config.hidden_size
    eh = r.mat(r.t(f"{blk}.nextn.eh_proj"))
    if tuple(eh.shape) != (hidden, 2 * hidden):
        raise ValueError(
            f"{model_path}: {blk}.nextn.eh_proj is {tuple(eh.shape)}, expected {(hidden, 2 * hidden)}"
        )
    yield "mtp.fc_embedding.weight", eh[:, :hidden].clone()
    yield "mtp.fc_hidden.weight", eh[:, hidden:].clone()
    yield "mtp.pre_fc_norm_embedding.weight", r.norm(f"{blk}.nextn.enorm")
    yield "mtp.pre_fc_norm_hidden.weight", r.norm(f"{blk}.nextn.hnorm")
    yield from r.attention(blk, "mtp.layers.0.self_attn")
    yield from r.mlp_and_hc(blk, "mtp.layers.0")
    for proj in ("down", "up"):
        yield (
            f"mtp.hyper_connection_mixer.input_mix_weight_{proj}.weight",
            r.mat(r.t(f"{blk}.nextn.hc_head_{proj}")),
        )
    yield (
        "mtp.hyper_connection_mixer.hc_norm.weight",
        r.norm(f"{blk}.nextn.hc_head_norm"),
    )


def ple_table_info(model_path: str) -> GgufTensorInfo:
    info = gguf_tensor_index(model_path).get("per_layer_token_embd.weight")
    if info is None:
        raise ValueError(f"{model_path} has no per_layer_token_embd.weight tensor")
    return info


def check_ple_table(
    path: str, *, head_dim: int | None = None, rows_needed: int | None = None
) -> GgufTensorInfo:
    """A --ple-table file: one 2-D ``per_layer_token_embd`` of ``head_dim``-wide rows, no weight scale (no PLE reader applies
    one), at least ``rows_needed`` rows (the checkpoint pads its table to whole shards: 320,001,536 rows for 320,001,446 hashed)."""
    info = ple_table_info(path)
    if "per_layer_token_embd.scale" in gguf_tensor_index(path):
        raise NotImplementedError(
            f"--ple-table {path}: a PLE table with a weight scale (per_layer_token_embd.scale) is not supported"
        )
    if len(info.ne) != 2:
        raise ValueError(
            f"--ple-table {path}: per_layer_token_embd has shape {info.ne}, expected [row width, rows]"
        )
    if head_dim is not None and info.ne[0] != head_dim:
        raise ValueError(
            f"--ple-table {path}: rows are {info.ne[0]} wide, this model's are {head_dim}"
        )
    if rows_needed is not None and info.ne[1] < rows_needed:
        raise ValueError(
            f"--ple-table {path} has {info.ne[1]} rows; this model's n-gram hash addresses {rows_needed}"
        )
    return info


__all__ = [
    "check_ple_table",
    "gguf_float_dense",
    "gguf_float_modules",
    "gguf_mm_config",
    "iter_gguf_weights",
    "iter_sidecar_weights",
    "parse_gguf_config",
    "ple_table_info",
    "sidecar_block",
    "sidecar_quant",
    "sidecar_to_gguf",
    "to_gguf",
]
