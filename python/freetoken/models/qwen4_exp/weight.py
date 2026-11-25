"""Qwen3.8-Flash-Next checkpoint reader (bf16, block-fp8, ModelOpt and llm-compressor exports).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseReader``.
* :func:`load_ple_table` -- the n-gram table (47.7 GiB fp8 or 95.4 GiB bf16), 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named under the checkpoint's dialect, for the offload cache's expert reader.

Dropped: ``mtp.*`` (speculative head, read by :func:`iter_mtp_weights`); ``model.visual.*`` is kept only when the model built the tower.
"""

from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass, replace

import safetensors
import torch
from tqdm import tqdm

from freetoken.distributed import get_tp_info
from freetoken.layers.int8_g32 import quantize_int8_g32
from freetoken.layers.quantization import (
    QuantConfig,
    QuantKind,
    QuantScheme,
    get_quant_config,
)
from freetoken.models.config import (
    VISION_KEY_PREFIXES,
    fp8_dense_enabled,
    fp8_dense_scale,
    fp8_lmhead_enabled,
    fp8_lmhead_scale,
    hc_int8_enabled,
    int8_dense_enabled,
)
from freetoken.models.loader import iter_weight_files, shard_tensor
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.models.qwen3_vl.weight import rename_vl_prefix
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, build_shared_bank, read_range_into
from freetoken.utils import (
    cached_load_hf_config,
    div_even,
    download_hf_weight,
    init_logger,
)
from freetoken.utils.progress import byte_bar

from .config import parse_config, use_fp8_dense, use_fp8_lmhead

logger = init_logger(__name__)

# Routed NVFP4 experts: per-expert, un-fused. Matched against the RAW weight_map key in nvfp4_banks.
# The ``model.language_model.`` anchor excludes the MTP head's stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {
    "weight": "weight",
    "weight_scale": "weight_scale",
    "weight_global": "weight_scale_2",
}

# leaves the model builds as quant-aware Linear layers: their tensors are read under the QuantConfig
_LINEAR_LEAVES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "index_qk_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_mix_weight_down",
        "block_inject_weight",
        "input_mix_weight_up",
        "key_proj",
        "value_proj",
        "lm_head",
        "fc_embedding",
        "fc_hidden",
    }
)
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# The per-layer HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM pads the merged rows to a multiple of 16 for cuBLAS (hyperconnection.py pad_size).
# The top-level hyper_connection_mixer has no injection and never fuses.
_PAD_TO = {"input_mix_weight_down_block_inject": 16}
_HC_WITH_INJECT = (".attn_hyper_connection", ".mlp_hyper_connection")


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(
            f"scale has {flat.numel()} elements for a weight with {rows} output rows (shape {tuple(scale.shape)}); expected 1 or {rows}"
        )
    return flat.contiguous()


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in.

    The part table is the family's packed_modules_mapping. The QuantConfig picks the GDN in_proj layout and, through its dialect's storage names, which checkpoint tensor fills which role.
    """

    def __init__(
        self, quant: QuantConfig | None, packed: tuple[tuple[str, tuple[str, ...]], ...]
    ) -> None:
        self.quant = quant
        self.groups = {
            fused: parts for fused, parts in packed if fused != "experts"
        }  # experts: bank reader
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores}, scheme)
        self.pending: dict[
            str,
            tuple[
                int,
                dict[int, dict[str, torch.Tensor]],
                dict[int, set[str]],
                QuantScheme | None,
            ],
        ] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from b|a; same test as layers/linear_attn/gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        if fused in _PAD_TO and not parent.endswith(_HC_WITH_INJECT):
            return module, 0, 1
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        scheme = self.scheme(module)
        roles = (
            {"weight": "weight"}
            if scheme is None
            else {n: r for r, e in self.quant.storage(scheme).items() for n in e.names}
        )
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(
                f"{name}: the checkpoint's quant config declares {module} {scheme or 'unquantized'}, stored as {sorted(roles)}"
            )
        if role == "weight":
            self._check_weight(name, module, scheme, tensor)
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, scheme))
        part = parts.setdefault(idx, {})
        if role in part:
            raise ValueError(
                f"{name}: {module} already has its {role} from another checkpoint tensor"
            )
        part[role] = tensor
        expected[idx] = set(roles.values())
        if scheme is not None and set(part) == expected[idx]:
            parts[idx] = self._finish(module, scheme, part)
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], scheme)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, _scheme) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}")
        return lines

    @staticmethod
    def _check_weight(
        name: str, module: str, scheme: QuantScheme | None, tensor: torch.Tensor
    ) -> None:
        if scheme is None:
            if tensor.dtype in _QUANT_DTYPES:
                raise ValueError(
                    f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized"
                )
            return
        elem = _ELEM_DTYPES.get(scheme.weight.elem)
        if elem is None:
            raise ValueError(f"{name}: qwen4_exp has no dense reader for {scheme}")
        if tensor.dtype is not elem:
            raise ValueError(
                f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}"
            )
        rows, cols = (
            tensor.shape[0],
            tensor.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1),
        )
        block_rows, block_cols = scheme.weight.group or (1, 1)
        if block_rows > 1 and rows % block_rows or block_cols > 1 and cols % block_cols:
            raise ValueError(
                f"{name}: {rows}x{cols} is not a multiple of the {block_rows}x{block_cols} scale block of {module}"
            )

    def _finish(
        self, module: str, scheme: QuantScheme, part: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Validate one complete part against ``scheme`` and put its scales in the layer's form."""
        storage = self.quant.storage(scheme)
        part = {
            role: 1.0 / t.to(torch.float32) if storage[role].reciprocal else t
            for role, t in part.items()
        }
        weight = part["weight"]
        rows, cols = (
            weight.shape[0],
            weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1),
        )
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = (
            "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        )
        out = dict(part)
        if block_cols < 0:
            try:
                out[scale_role] = _per_row_scale(part[scale_role], rows)
            except ValueError as exc:
                raise ValueError(f"{module}.{scale_role}: {exc}") from None
        else:
            want = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != want:
                raise ValueError(
                    f"{module}: {scale_role} is {tuple(part[scale_role].shape)}, expected {want} for {scheme}"
                )
            if (
                scheme.weight.scale == "e4m3"
                and part[scale_role].dtype is not torch.float8_e4m3fn
            ):
                raise ValueError(
                    f"{module}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales"
                )
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(
                    f"{module}: weight_global has {g.numel()} elements, expected one per-tensor scale"
                )
            out["weight_global"] = g.expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out

    @staticmethod
    def _emit(
        target: str, parts: list[dict[str, torch.Tensor]], scheme: QuantScheme | None
    ) -> list[tuple[str, torch.Tensor]]:
        pad_to = _PAD_TO.get(target.rpartition(".")[2], 0)
        if pad_to and scheme is not None:
            raise ValueError(
                f"{target}: qwen4_exp pads this fusion to {pad_to} rows and reads it bf16 only, the checkpoint's quant config declares {scheme}"
            )
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so one range covers them: max is exact then and safe if they drift
                out.append((f"{target}.{role}", torch.stack(tensors).max()))
                continue
            pad = (-sum(t.shape[0] for t in tensors)) % pad_to if pad_to else 0
            if pad:
                tensors.append(
                    torch.zeros(
                        pad,
                        *tensors[0].shape[1:],
                        dtype=tensors[0].dtype,
                        device=tensors[0].device,
                    )
                )
            out.append(
                (
                    f"{target}.{role}",
                    tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0),
                )
            )
        return out


def _shard_rows(
    t: torch.Tensor, parts: list[tuple[int, int]], rank: int, world: int
) -> torch.Tensor:
    """Column-parallel slice (dim 0) of a ``[part0 | part1 | ...]`` fusion; ``parts`` gives each
    part as ``(heads, rows_per_head)``. Heads split evenly across ranks; a part with fewer heads
    than ranks (GQA kv) replicates head ``rank * heads // world``, the ``div_even(...,
    allow_replicate=True)`` convention of the TP-aware layers."""
    out, off = [], 0
    for heads, rows in parts:
        local = div_even(heads, world, allow_replicate=True)
        first = rank * heads // world
        out.append(t[off + first * rows : off + (first + local) * rows])
        off += heads * rows
    assert off == t.shape[0], (
        f"fusion parts {parts} cover {off} rows, tensor has {t.shape[0]}"
    )
    return torch.cat(out, dim=0)


def _shard(name: str, t: torch.Tensor, config, rank: int, world: int) -> torch.Tensor:
    """TP shard of one state-dict tensor (fused projections included); identity at TP=1.

    Column-parallel (dim 0, by head): attention ``qkv_proj`` [q|gate per head | k | v], GDN
    ``in_proj`` [q | k | v | z | b | a] and the matching ``conv1d`` channels, ``A_log`` /
    ``dt_bias``, shared-expert ``gate_up_proj``. Row-parallel (dim 1): ``o_proj``,
    ``out_proj``, shared-expert ``down_proj``. Vocab rows: ``embed_tokens`` / ``lm_head``.
    The vision tower follows qwen3_vl/vision.py (see ``_shard_vision``).
    Everything else (router, indexer, norms, HC, PLE, shared-expert gate) is replicated.
    """
    if world == 1:
        return t
    if name.startswith(VISION_KEY_PREFIXES):
        return _shard_vision(name, t, config.vision_config, rank, world)
    if name.endswith(".self_attn.qkv_proj.weight"):
        q = (config.num_qo_heads, 2 * config.head_dim)
        kv = (config.num_kv_heads, config.head_dim)
        return _shard_rows(t, [q, kv, kv], rank, world)
    if ".linear_attn." in name:
        g = config.linear_attention_group()
        k = (g.num_key_heads, g.key_head_dim)
        v = (g.num_value_heads, g.value_head_dim)
        if name.endswith(".in_proj.weight"):
            return _shard_rows(t, [k, k, v, v, (v[0], 1), (v[0], 1)], rank, world)
        if name.endswith(".conv1d.weight"):
            return _shard_rows(t, [k, k, v], rank, world)
        if name.endswith((".A_log", ".dt_bias")):
            return _shard_rows(t, [(v[0], 1)], rank, world)
        if name.endswith(".out_proj.weight"):
            return t.chunk(world, dim=1)[rank].clone()
        return t
    if name.endswith(".shared_expert.gate_up_proj.weight"):
        half = t.shape[0] // 2
        return _shard_rows(t, [(half, 1), (half, 1)], rank, world)
    # o_proj / down_proj: dim 1; embed_tokens / lm_head: vocab rows; others unchanged.
    return shard_tensor(name, t, rank=rank, world_size=world, num_kv_heads=None)


def _shard_vision(
    name: str, t: torch.Tensor, vc, rank: int, world: int
) -> torch.Tensor:
    """TP shard of a vision-tower tensor, in the layout of the TP-aware ops in qwen3_vl/vision.py.

    ``attn.qkv`` (LinearQKVMerged, [q | k | v] of ``num_heads`` heads each): by head within each part.
    ``attn.proj`` and every ``linear_fc2`` (LinearOProj / LinearRowParallel): the weight by input column; the bias on rank 0 only, zeros elsewhere, because those layers add their bias inside the GEMM on every rank and then all-reduce.
    Every ``linear_fc1`` (column-parallel, block MLP and patch merger): weight and bias by output row. Patch / position embeddings and norms are replicated.
    """
    if name.endswith((".attn.qkv.weight", ".attn.qkv.bias")):
        head = (vc.num_heads, vc.hidden_size // vc.num_heads)
        return _shard_rows(t, [head, head, head], rank, world)
    if name.endswith((".attn.proj.weight", ".linear_fc2.weight")):
        assert t.shape[1] % world == 0, (name, tuple(t.shape), world)
        return t.chunk(world, dim=1)[rank].clone()
    if name.endswith((".attn.proj.bias", ".linear_fc2.bias")):
        return t if rank == 0 else torch.zeros_like(t)
    if name.endswith((".linear_fc1.weight", ".linear_fc1.bias")):
        assert t.shape[0] % world == 0, (name, tuple(t.shape), world)
        return t.chunk(world, dim=0)[rank].clone()
    return t


# Load-time FP8, per tensor or per channel (use_fp8_dense, fp8_dense_scale, layers/fp8_dynamic.py): these keep their name and gain a sibling ``weight_scale``;
# GDN ``in_proj`` splits into the fp8 ``in_proj_qkvz`` and the bf16 ``in_proj_ba`` (the gate rows stay bf16, as in block-fp8 checkpoints).
_FP8_DENSE_SUFFIXES = (
    ".self_attn.qkv_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.out_proj.weight",
)
# behind its own flag: this one moves the logits (see models.config.fp8_lmhead_enabled)
_FP8_LMHEAD_SUFFIX = "lm_head.weight"
_E4M3_MAX = 448.0


def _quantize_per_tensor(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(e4m3 weight, fp32 scale ())`` with ``w ~= weight * scale``."""
    w = w.float()
    scale = (w.abs().amax() / _E4M3_MAX).clamp_min(1e-12)
    return (w / scale).clamp_(-_E4M3_MAX, _E4M3_MAX).to(
        torch.float8_e4m3fn
    ), scale.reshape(())


def _quantize_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(e4m3 weight, fp32 scale [rows])`` with ``w ~= weight * scale[:, None]``: one scale per output row.

    E4M3 rounds relative to the value (3 mantissa bits), so a row scale only helps rows whose range is far
    below the tensor's (per-tensor scaling makes them subnormal); for normal weights the error is the same."""
    w = w.float()
    scale = (w.abs().amax(dim=1) / _E4M3_MAX).clamp_min(1e-12)
    w8 = (w / scale[:, None]).clamp_(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
    return w8, scale


def _fp8_dense(
    name: str,
    t: torch.Tensor,
    config,
    world: int,
    *,
    dense: bool = True,
    lm_head: bool = False,
    dense_channel: bool = False,
    lmhead_channel: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """The (already sharded) dense tensor as the fp8_dynamic model expects it.

    ``dense_channel`` / ``lmhead_channel``: one weight scale per output row of this rank's shard (see
    ``fp8_dense_scale``); the default is one scale per tensor per rank."""
    dense_quant = _quantize_per_channel if dense_channel else _quantize_per_tensor
    if dense and name.endswith(".linear_attn.in_proj.weight"):
        g = config.linear_attention_group()
        nk = div_even(g.num_key_heads, world, allow_replicate=True)
        nv = div_even(g.num_value_heads, world, allow_replicate=True)
        qkvz = (
            2 * nk * g.key_head_dim + 2 * nv * g.value_head_dim
        )  # [q | k | v | z] local rows
        assert t.shape[0] == qkvz + 2 * nv, (name, t.shape, qkvz, nv)
        base = name[: -len("in_proj.weight")]
        w8, scale = dense_quant(t[:qkvz])
        yield base + "in_proj_qkvz.weight", w8
        yield base + "in_proj_qkvz.weight_scale", scale
        # clone, not contiguous(): a contiguous row slice comes back as a view, which keeps the whole bf16 in_proj resident
        yield base + "in_proj_ba.weight", t[qkvz:].clone()
    elif dense and name.endswith(_FP8_DENSE_SUFFIXES):
        w8, scale = dense_quant(t)
        yield name, w8
        yield name[: -len("weight")] + "weight_scale", scale
    elif lm_head and name.endswith(_FP8_LMHEAD_SUFFIX):
        w8, scale = (_quantize_per_channel if lmhead_channel else _quantize_per_tensor)(
            t
        )
        yield name, w8
        yield name[: -len("weight")] + "weight_scale", scale
    else:
        yield name, t


def _fp8_dense_applies(name: str, config) -> bool:
    """The builder's use_fp8_dense answer for the mixer that owns this tensor."""
    mixer = name.rpartition(".")[0].rpartition(".")[0]
    if name.endswith((".linear_attn.in_proj.weight", ".linear_attn.out_proj.weight")):
        return use_fp8_dense(config, f"{mixer}.in_proj_qkvz")
    if name.endswith((".self_attn.qkv_proj.weight", ".self_attn.o_proj.weight")):
        return use_fp8_dense(config, f"{mixer}.qkv_proj")
    return False


def _int8_dense_applies(name: str, config) -> bool:
    """The builder's ``use_int8_dense`` answer for the mixer that owns this tensor."""
    from .config import use_int8_dense

    mixer = name.rpartition(".")[0].rpartition(".")[0]
    if name.endswith((".linear_attn.in_proj.weight", ".linear_attn.out_proj.weight")):
        return use_int8_dense(config, f"{mixer}.in_proj_qkvz")
    if name.endswith((".self_attn.qkv_proj.weight", ".self_attn.o_proj.weight")):
        return use_int8_dense(config, f"{mixer}.qkv_proj")
    return False


_HC_INT8_SUFFIXES = (
    ".input_mix_weight_down_block_inject.weight",
    ".input_mix_weight_down.weight",
    ".input_mix_weight_up.weight",
)


def _int8_dense_emit(
    name: str, t: torch.Tensor, config, world: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """Quantize a TP-sharded QSA/GDN dense or HC projection into ``int8 g32`` (`weight` + `weight_scale`)."""
    if name.endswith(".linear_attn.in_proj.weight"):
        g = config.linear_attention_group()
        nk = div_even(g.num_key_heads, world, allow_replicate=True)
        nv = div_even(g.num_value_heads, world, allow_replicate=True)
        qkvz = 2 * nk * g.key_head_dim + 2 * nv * g.value_head_dim
        assert t.shape[0] == qkvz + 2 * nv, (name, t.shape, qkvz, nv)
        base = name[: -len("in_proj.weight")]
        qw, sc = quantize_int8_g32(t[:qkvz])
        yield base + "in_proj_qkvz.weight", qw
        yield base + "in_proj_qkvz.weight_scale", sc
        yield base + "in_proj_ba.weight", t[qkvz:].clone()
    else:
        qw, sc = quantize_int8_g32(t)
        yield name, qw
        yield name[: -len("weight")] + "weight_scale", sc


def dense_emitter(config, rank: int, world: int):
    """``(emit, quantized_dense, lmhead)``: ``emit(name, tensor)`` yields one dense state-dict tensor as this rank loads it,
    TP-sharded and then load-time FP8 or ``int8 g32`` where the builder made quantized modules."""
    lmhead = config is not None and use_fp8_lmhead(config)
    fp8 = config is not None and config.attn_quant == "fp8_dynamic"
    int8 = config is not None and config.attn_quant == "int8_g32"
    hc8 = config is not None and hc_int8_enabled()
    dense_channel = fp8 and fp8_dense_scale() == "channel"
    lmhead_channel = lmhead and fp8_lmhead_scale() == "channel"
    if int8:
        logger.info(
            "qwen4_exp dense projections: load-time int8 g32 (W8A16 C4b), FREETOKEN_INT8_DENSE=1"
        )
    elif fp8:
        logger.info(
            f"qwen4_exp dense projections: load-time per-{'channel' if dense_channel else 'tensor'} FP8 "
            "(W8A8 via _scaled_mm), FREETOKEN_FP8_DENSE=1"
        )
    if hc8:
        logger.info(
            "qwen4_exp hyper-connections: load-time int8 g32 (hc8), FREETOKEN_HC_INT8=1"
        )
    if lmhead:
        logger.info(
            f"qwen4_exp lm_head: load-time per-{'channel' if lmhead_channel else 'tensor'} FP8, FREETOKEN_FP8_LMHEAD=1"
        )

    def emit(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        tensor = _shard(name, tensor, config, rank, world)
        if int8 and _int8_dense_applies(name, config):
            yield from _int8_dense_emit(name, tensor, config, world)
            return
        if hc8 and name.endswith(_HC_INT8_SUFFIXES):
            prefix = name[: -len(".weight")]
            if config.quant is None or config.quant.scheme_for(prefix) is None:
                yield from _int8_dense_emit(name, tensor, config, world)
                return
        dense = fp8 and _fp8_dense_applies(name, config)
        if dense or lmhead:
            yield from _fp8_dense(
                name,
                tensor,
                config,
                world,
                dense=dense,
                lm_head=lmhead,
                dense_channel=dense_channel,
                lmhead_channel=lmhead_channel,
            )
        else:
            yield name, tensor

    return emit, (fp8 or int8 or hc8), lmhead


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the model's state dict minus the routed experts.
    Each Linear is read under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, per-channel / per-tensor fp8, block-fp8, MXFP8 and NVFP4 projections in any mix land as the model's state dict.
    Fusions, per role: attention q|k|v -> ``qkv_proj``; GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, or ``in_proj_qkvz`` + ``in_proj_ba`` when qkv|z is quantized; shared-expert gate|up -> ``gate_up_proj``; each per-layer HC's ``input_mix_weight_down`` | ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.
    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the routed experts are NVFP4 and always come from the offload cache's expert reader.
    """
    if not include_non_moe:
        return

    tp = get_tp_info()
    hf_config = cached_load_hf_config(model_path)
    spec = get_model_spec(hf_config.architectures[0])
    quant = get_quant_config()
    reader = _DenseReader(quant, spec.packed_modules_mapping)
    # the sharding geometry and the fp8 split need the parsed config, under the QuantConfig the builder saw; a plain TP=1 read never touches it
    needs_config = (
        tp.size > 1
        or fp8_dense_enabled()
        or fp8_lmhead_enabled()
        or int8_dense_enabled()
        or hc_int8_enabled()
    )
    config = replace(parse_config(hf_config), quant=quant) if needs_config else None
    emit, fp8, lmhead = dense_emitter(config, tp.rank, tp.size)

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not tp.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = (
                    None
                    if name.startswith(VISION_KEY_PREFIXES)
                    else reader.add(name, tensor)
                )
                if emitted is None:
                    yield from emit(name, tensor)
                else:
                    for fused_name, fused_tensor in emitted:
                        yield from emit(fused_name, fused_tensor)

    if reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (
            f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else ""
        )
        raise ValueError(
            f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}"
        )
    if (fp8 or lmhead) and device.type == "cuda":
        # hand the bf16 originals and fp32 temporaries back, so the expert-cache planner (free VRAM after load) sees the halved dense footprint
        torch.cuda.empty_cache()


# routed experts: expert_pieces.iter_extra_layer_expert_pieces
_MTP_EXPERTS_PREFIX = "mtp.layers.0.mlp.experts."
_MTP_EXPERT_RE = re.compile(
    re.escape(_MTP_EXPERTS_PREFIX)
    + r"(?:(?P<expert>\d+)\.(?P<proj>gate|up|down)_proj\.(?P<kind>weight|weight_scale_inv)|(?P<stacked>gate_up|down)_proj)$"
)


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a 2D block-fp8 weight ``[rows, cols]`` with ``scale [sr, sc]`` to bf16."""
    rows, cols = weight.shape
    sr, sc = scale.shape
    br = (rows + sr - 1) // sr
    bc = (cols + sc - 1) // sc
    s = (
        scale.to(torch.float32)
        .repeat_interleave(br, dim=0)[:rows]
        .repeat_interleave(bc, dim=1)[:, :cols]
    )
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def _dequant_emitted(
    emitted: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    by_name = dict(emitted)
    weight_key = next((k for k in by_name if k.endswith(".weight")), None)
    if weight_key is None:
        return emitted
    base = weight_key[: -len(".weight")]
    w = by_name[weight_key]
    if f"{base}.weight_scale_inv" in by_name:
        return [
            (weight_key, _dequant_fp8_block(w, by_name[f"{base}.weight_scale_inv"]))
        ]
    if f"{base}.weight_scale" in by_name:
        if f"{base}.weight_global" in by_name:
            raise ValueError(
                f"{weight_key}: NVFP4 MTP dense projection cannot be TP-dequantized on the fly"
            )
        s = by_name[f"{base}.weight_scale"].to(torch.float32)
        if s.ndim == 1:
            s = s.unsqueeze(1)
        return [(weight_key, (w.to(torch.float32) * s).to(torch.bfloat16))]
    return emitted


def _shard_mtp_expert_pieces(pieces, rank: int, world: int):
    for bank_layer, e0, e1, piece in pieces:
        if "gate_up" in piece:
            gate, up = piece["gate_up"].chunk(2, dim=1)
            down = piece["down"]
        else:
            gate, up, down = piece["gate"], piece["up"], piece["down"]
            if "gate_scale" in piece:
                gate = torch.stack(
                    [
                        _dequant_fp8_block(w, s)
                        for w, s in zip(gate, piece["gate_scale"])
                    ]
                )
                up = torch.stack(
                    [_dequant_fp8_block(w, s) for w, s in zip(up, piece["up_scale"])]
                )
                down = torch.stack(
                    [
                        _dequant_fp8_block(w, s)
                        for w, s in zip(down, piece["down_scale"])
                    ]
                )
        local_i = div_even(gate.shape[1], world)
        sl = slice(rank * local_i, (rank + 1) * local_i)
        yield (
            bank_layer,
            e0,
            e1,
            {
                "gate_up": torch.cat(
                    [gate[:, sl, :], up[:, sl, :]], dim=1
                ).contiguous(),
                "down": down[:, :, sl].contiguous(),
            },
        )


def iter_mtp_weights(
    model_path: str, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    """The ``mtp.*`` head but its routed experts, named as Qwen4ExpMTPDraft names it and fused like the target's dense weights."""
    from .mtp import draft_config

    tp = get_tp_info()
    hf_config = cached_load_hf_config(model_path)
    quant = get_quant_config()
    reader = _DenseReader(
        quant, get_model_spec(hf_config.architectures[0]).packed_modules_mapping
    )
    needs_config = tp.size > 1 or fp8_dense_enabled()
    draft_cfg = (
        draft_config(replace(parse_config(hf_config), quant=quant), tp_size=tp.size)
        if needs_config
        else None
    )
    fp8 = draft_cfg is not None and draft_cfg.attn_quant == "fp8_dynamic"
    dense_channel = fp8 and fp8_dense_scale() == "channel"

    def emit(
        raw_out_name: str, tensor: torch.Tensor
    ) -> Iterator[tuple[str, torch.Tensor]]:
        name = raw_out_name.removeprefix("mtp.")
        if draft_cfg is not None:
            tensor = _shard(name, tensor, draft_cfg, tp.rank, tp.size)
        if fp8 and _fp8_dense_applies(raw_out_name, draft_cfg):
            yield from _fp8_dense(
                name,
                tensor,
                draft_cfg,
                tp.size,
                dense=True,
                lm_head=False,
                dense_channel=dense_channel,
            )
        else:
            yield name, tensor

    seen = False
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if not raw_name.startswith("mtp.") or raw_name.endswith(
                    (".k_scale", ".v_scale", ".q_scale", ".prob_scale")
                ):
                    continue
                seen = True
                if raw_name.startswith(_MTP_EXPERTS_PREFIX):
                    continue
                tensor = f.get_tensor(raw_name)
                emitted = reader.add(raw_name, tensor)
                if emitted is None:
                    yield from emit(raw_name, tensor)
                else:
                    if tp.size > 1 and emitted:
                        emitted = _dequant_emitted(emitted)
                    for fused_name, fused_tensor in emitted:
                        yield from emit(fused_name, fused_tensor)
    if not seen:
        raise ValueError(f"{model_path} has no mtp.* weights")
    if reader.pending:
        raise ValueError(
            f"{model_path}: incomplete MTP projections: {reader.missing()}"
        )


def iter_mtp_expert_pieces(model_path: str):
    """The MTP head's routed experts: the stacked bf16 pair, or per-expert projections with their block-fp8 scales."""
    from freetoken.moe.expert_pieces import iter_extra_layer_expert_pieces

    def locate(name: str):
        m = _MTP_EXPERT_RE.match(name)
        if m is None:
            return None
        if m["stacked"]:
            return None, m["stacked"]
        return int(m["expert"]), m["proj"] + (
            "_scale" if m["kind"] == "weight_scale_inv" else ""
        )

    pieces = iter_extra_layer_expert_pieces(model_path, _MTP_EXPERTS_PREFIX, locate)
    tp = get_tp_info()
    if tp.size > 1:
        return _shard_mtp_expert_pieces(pieces, tp.rank, tp.size)
    return pieces


def iter_vision_weights(
    model_path: str, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus per-tensor scale or 4-bit lut4 codebook."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)
    weight_lut: torch.Tensor | None = None  # [16] float32 codebook for lut4-gptq tables
    group_size: int = 32
    head_dim: int | None = None

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, cols]`` view of the bank in the stored dtype."""
        return self.bank.tensor


_PLE_LUT4_PACKED_RE = re.compile(
    r"ngram_embedding\.shard_(?P<shard>\d+)\.weight_packed$"
)
_PLE_LUT4_SCALE_RE = re.compile(r"ngram_embedding\.shard_(?P<shard>\d+)\.weight_scale$")
_PLE_LUT4_LUT_RE = re.compile(r"ngram_embedding\.shard_(?P<shard>\d+)\.weight_lut$")


def is_lut4_ple_path(path: str) -> bool:
    """True when ``path`` is a directory or safetensors file containing ``lut4-gptq`` PLE shards."""
    if os.path.isdir(path):
        files = _ple_table_files(path)
    elif os.path.isfile(path) and path.endswith(".safetensors"):
        files = [path]
    else:
        return False
    for fpath in files:
        header, _ = _safetensors_header(fpath)
        if any(_PLE_LUT4_PACKED_RE.search(k) for k in header if k != "__metadata__"):
            return True
    return False


def check_ple_table_lut4(
    path: str, *, head_dim: int, rows_needed: int | None = None
) -> tuple[int, int]:
    """Validate ``lut4-gptq`` PLE shards at ``path`` and return ``(total_rows, group_size)``."""
    files = _ple_table_files(path) if os.path.isdir(path) else [path]
    packed_shapes: dict[int, tuple[int, int]] = {}
    scale_shapes: dict[int, tuple[int, int]] = {}
    for fpath in files:
        header, _ = _safetensors_header(fpath)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            mp = _PLE_LUT4_PACKED_RE.search(key)
            if mp is not None:
                if meta["dtype"] != "U8":
                    raise ValueError(f"LUT4 PLE {key} must be U8, got {meta['dtype']}")
                packed_shapes[int(mp.group("shard"))] = tuple(meta["shape"])
                continue
            ms = _PLE_LUT4_SCALE_RE.search(key)
            if ms is not None:
                if meta["dtype"] != "F16":
                    raise ValueError(f"LUT4 PLE {key} must be F16, got {meta['dtype']}")
                scale_shapes[int(ms.group("shard"))] = tuple(meta["shape"])
    if not packed_shapes or set(packed_shapes) != set(scale_shapes):
        raise ValueError(f"Incomplete lut4-gptq PLE shards at {path}")
    _shard0_rows, packed_cols = packed_shapes[min(packed_shapes)]
    if packed_cols * 2 != head_dim:
        raise ValueError(
            f"LUT4 PLE packed width {packed_cols} != head_dim/2 ({head_dim // 2})"
        )
    _, num_groups = scale_shapes[min(scale_shapes)]
    if num_groups <= 0 or head_dim % num_groups != 0:
        raise ValueError(
            f"LUT4 PLE scale groups {num_groups} does not divide head_dim {head_dim}"
        )
    total_rows = sum(r for r, _ in packed_shapes.values())
    if rows_needed is not None and total_rows < rows_needed:
        raise ValueError(
            f"LUT4 PLE at {path} has {total_rows} rows, model needs {rows_needed}"
        )
    return total_rows, head_dim // num_groups


def load_ple_table_lut4(
    path: str,
    qwen4_args,
    *,
    pin: bool = True,
    tp_cpu_group=None,
    shm_dir: str = "/dev/shm",
    device=None,
) -> PleTable:
    """Load ``lut4-gptq`` PLE shards (`weight_packed` U8 + `weight_scale` F16 + `weight_lut` F32)
    into one shared ``/dev/shm`` host bank of ``[total_rows, packed_cols + scale_bytes]`` uint8."""
    files = _ple_table_files(path) if os.path.isdir(path) else [path]
    packed_files: dict[int, tuple[str, str]] = {}
    scale_files: dict[int, tuple[str, str]] = {}
    lut_tensor: torch.Tensor | None = None
    for fpath in files:
        header, _ = _safetensors_header(fpath)
        for key in header:
            if key == "__metadata__":
                continue
            mp = _PLE_LUT4_PACKED_RE.search(key)
            if mp is not None:
                packed_files[int(mp.group("shard"))] = (fpath, key)
                continue
            ms = _PLE_LUT4_SCALE_RE.search(key)
            if ms is not None:
                scale_files[int(ms.group("shard"))] = (fpath, key)
                continue
            ml = _PLE_LUT4_LUT_RE.search(key)
            if ml is not None:
                with safetensors.safe_open(fpath, framework="pt", device="cpu") as sf:
                    cur_lut = sf.get_tensor(key).to(torch.float32).reshape(-1)
                if cur_lut.numel() != 16:
                    raise ValueError(
                        f"LUT4 PLE {key} must have 16 entries, got {cur_lut.numel()}"
                    )
                if lut_tensor is not None and not torch.allclose(lut_tensor, cur_lut):
                    raise ValueError(f"Inconsistent LUT4 PLE codebook in {key}")
                lut_tensor = cur_lut

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(packed_files) != list(range(expected)) or sorted(scale_files) != list(
        range(expected)
    ):
        raise ValueError(
            f"LUT4 PLE needs shards 0..{expected - 1}, found packed={sorted(packed_files)[:8]}"
        )
    if lut_tensor is None:
        raise ValueError(f"LUT4 PLE at {path} has no weight_lut tensor")

    head_dim = int(qwen4_args.ngram_head_dim)
    p0_path, p0_key = packed_files[0]
    s0_path, s0_key = scale_files[0]
    h_p0, _ = _safetensors_header(p0_path)
    h_s0, _ = _safetensors_header(s0_path)
    shard_rows, packed_cols = h_p0[p0_key]["shape"]
    s_rows, num_groups = h_s0[s0_key]["shape"]
    if (
        packed_cols * 2 != head_dim
        or s_rows != shard_rows
        or head_dim % num_groups != 0
    ):
        raise ValueError(
            f"Invalid LUT4 PLE shard shapes: packed={(shard_rows, packed_cols)} scale={(s_rows, num_groups)}"
        )
    group_size = head_dim // num_groups
    row_bytes = packed_cols + num_groups * 2
    total_rows = expected * shard_rows

    def _fill(buf: memoryview, show_bar: bool) -> None:
        bar = (
            byte_bar(total_rows * row_bytes, "Loading LUT4 PLE table")
            if show_bar
            else None
        )
        out = torch.frombuffer(
            buf, dtype=torch.uint8, count=total_rows * row_bytes
        ).view(total_rows, row_bytes)
        try:
            for shard in range(expected):
                pf, pk = packed_files[shard]
                sf_path, sk = scale_files[shard]
                with safetensors.safe_open(pf, framework="pt", device="cpu") as fp:
                    wp = fp.get_tensor(pk)
                with safetensors.safe_open(sf_path, framework="pt", device="cpu") as fs:
                    ws = fs.get_tensor(sk)
                if wp.shape != (shard_rows, packed_cols) or ws.shape != (
                    shard_rows,
                    num_groups,
                ):
                    raise ValueError(
                        f"LUT4 PLE shard {shard} shape mismatch: {wp.shape}, {ws.shape}"
                    )
                r0 = shard * shard_rows
                out[r0 : r0 + shard_rows, :packed_cols].copy_(wp)
                out[r0 : r0 + shard_rows, packed_cols:].copy_(
                    ws.contiguous().view(torch.uint8)
                )
                if bar is not None:
                    bar.update(shard_rows * row_bytes)
        finally:
            if bar is not None:
                bar.close()

    bank = build_shared_bank(
        (total_rows, row_bytes),
        torch.uint8,
        _fill,
        tp_cpu_group=tp_cpu_group,
        shm_dir=shm_dir,
        device=device,
    )
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(
        bank=bank,
        weight_scale=torch.tensor(1.0, dtype=torch.bfloat16),
        weight_lut=lut_tensor,
        group_size=group_size,
        head_dim=head_dim,
    )


_PLE_ST_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16}


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the PLE n-gram table tensors, and only those, into ``ple-table-*.safetensors`` next to an FTW checkpoint.

    The table is served from safetensors files in the checkpoint dir (see load_ple_table), not from FTW entries."""
    from safetensors.torch import save_file

    folder = download_hf_weight(model_path)
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush():
        nonlocal batch, size
        if batch:
            name = f"ple-table-{len(written):05d}.safetensors"
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(folder):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_FILE_BYTES:
                    flush()
    flush()
    return written


def load_ple_table(
    model_path: str,
    qwen4_args,
    *,
    pin: bool = True,
    workers: int = 8,
    chunk: int = 8 << 20,
    tp_cpu_group=None,
    shm_dir: str = "/dev/shm",
    device=None,
) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB (fp8) and must not also sit in the page cache while the bank holds the same bytes.
    An fp8 table needs the checkpoint's scalar ``weight_scale``; a bf16 table without one is unscaled.
    """
    folder = download_hf_weight(model_path)
    parts: dict[
        int, tuple[str, int, int]
    ] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    dtype: torch.dtype | None = None
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
            shard_dtype = _PLE_ST_DTYPES.get(meta["dtype"])
            if shard_dtype is None or dtype not in (None, shard_dtype):
                raise ValueError(
                    f"PLE table shard {key} has dtype {meta['dtype']}, expected one of {sorted(_PLE_ST_DTYPES)} for every shard"
                )
            dtype = shard_dtype
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(
                    f"PLE table shard {key} is {shape}, expected {[rows, cols]}"
                )
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(
            f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}"
        )
    if scale is None:
        if dtype is torch.float8_e4m3fn:
            raise ValueError("PLE table has no weight_scale")
        scale = torch.tensor(1.0, dtype=torch.bfloat16)

    shard_bytes = rows * cols * dtype.itemsize

    def _fill(buf: memoryview, show_bar: bool) -> None:
        bar = (
            byte_bar(expected * shard_bytes, "Loading PLE table") if show_bar else None
        )
        try:
            for shard in range(expected):
                path, offset, nbytes = parts[shard]
                assert nbytes == shard_bytes, (
                    f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
                )
                read_range_into(
                    buf,
                    path,
                    file_offset=offset,
                    nbytes=nbytes,
                    dest_offset=shard * shard_bytes,
                    workers=workers,
                    chunk=chunk,
                )
                if bar is not None:
                    bar.update(nbytes)
        finally:
            if bar is not None:
                bar.close()

    bank = build_shared_bank(
        (expected * rows, cols),
        dtype,
        _fill,
        tp_cpu_group=tp_cpu_group,
        shm_dir=shm_dir,
        device=device,
    )
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


def load_ple_table_gguf(
    path: str,
    qwen4_args,
    *,
    pin: bool = True,
    workers: int = 8,
    chunk: int = 8 << 20,
    piece: int = 4 << 30,
    tp_cpu_group=None,
    shm_dir: str = "/dev/shm",
    device=None,
) -> PleTable:
    """The ``per_layer_token_embd`` table of a one-tensor GGUF (``--ple-table``, e.g. Unsloth's lossless
    engram BF16 file) as one pinned host bank, read O_DIRECT like :func:`load_ple_table`.

    BF16 only: the UVA gather reads fp8 or bf16 rows; a packed ggml table is served by ``--ple-backend disk``."""
    from freetoken.layers.quantization.ggml_types import type_name
    from freetoken.third_party.gguf import GGMLQuantizationType

    from .gguf import check_ple_table

    info = check_ple_table(path, head_dim=qwen4_args.ngram_head_dim)
    if info.ggml_type != GGMLQuantizationType.BF16:
        raise ValueError(
            f"--ple-backend pinned reads a BF16 PLE table; {path} stores {type_name(info.ggml_type)}, "
            "which --ple-backend disk serves"
        )
    cols, rows = info.ne
    assert info.nbytes == rows * cols * 2, (info.nbytes, rows, cols)

    def _fill(buf: memoryview, show_bar: bool) -> None:
        bar = byte_bar(info.nbytes, "Loading PLE table") if show_bar else None
        try:
            for start in range(0, info.nbytes, piece):
                n = min(piece, info.nbytes - start)
                read_range_into(
                    buf,
                    info.path,
                    file_offset=info.offset + start,
                    nbytes=n,
                    dest_offset=start,
                    workers=workers,
                    chunk=chunk,
                )
                if bar is not None:
                    bar.update(n)
        finally:
            if bar is not None:
                bar.close()

    bank = build_shared_bank(
        (rows, cols),
        torch.bfloat16,
        _fill,
        tp_cpu_group=tp_cpu_group,
        shm_dir=shm_dir,
        device=device,
    )
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=torch.tensor(1.0, dtype=torch.bfloat16))


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(
            _EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))
        ),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.8-Flash-Next NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


__all__ = [
    "PleTable",
    "iter_mtp_expert_pieces",
    "iter_mtp_weights",
    "iter_weights",
    "load_ple_table",
    "nvfp4_expert_spec",
]
