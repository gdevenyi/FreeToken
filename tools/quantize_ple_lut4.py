#!/usr/bin/env python3
"""Offline CPU/GPU converter to 4-bit lut4-gptq PLE shards (`90 B/head`, 3.56x smaller than BF16).

Produces warp34-compatible safetensors shards (`ngram_embedding.shard_<i>.{weight_packed,weight_scale,weight_lut}`)
consumable directly via `ft serve --ple-table <out_dir> --ple-backend pinned`.
Safe to run on a serving host's CPU with `nice -n 19 ionice -c 3` (`--threads` caps intra-op parallelism).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file

_NF4_SEED = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def fit_lut16_codebook(
    samples: torch.Tensor, group_size: int = 32, iters: int = 12
) -> torch.Tensor:
    """Fit a 16-entry float32 Lloyd-Max codebook over normalized ``[-1, 1]`` group-32 blocks."""
    assert samples.ndim == 2 and samples.shape[1] % group_size == 0
    blocks = samples.to(torch.float32).reshape(-1, group_size)
    amax = blocks.abs().amax(dim=1, keepdim=True)
    non_zero = (amax.squeeze(1) > 1e-12).nonzero(as_tuple=False).squeeze(1)
    if non_zero.numel() == 0:
        return torch.tensor(_NF4_SEED, dtype=torch.float32)
    normed = (
        blocks.index_select(0, non_zero) / amax.index_select(0, non_zero)
    ).reshape(-1)
    if normed.numel() > 2_000_000:
        step = normed.numel() // 2_000_000
        normed = normed[::step]

    lut = torch.tensor(_NF4_SEED, dtype=torch.float32)
    for _ in range(iters):
        idx = (normed.unsqueeze(1) - lut.unsqueeze(0)).abs().argmin(dim=1)
        new_lut = lut.clone()
        for k in range(16):
            mask = idx == k
            if bool(mask.any()):
                new_lut[k] = normed[mask].mean()
        max_abs = float(new_lut.abs().max().item())
        if max_abs > 0:
            new_lut = new_lut / max_abs
        lut, _ = torch.sort(new_lut)
    return lut


def quantize_rows_lut4(
    rows: torch.Tensor,
    lut: torch.Tensor,
    *,
    group_size: int = 32,
    refine_steps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, D]`` float rows into ``(weight_packed uint8 [N, D//2], weight_scale fp16 [N, D//group_size])``."""
    n, d = rows.shape
    assert d % 2 == 0 and d % group_size == 0
    num_groups = d // group_size
    x = rows.to(torch.float32).view(n, num_groups, group_size)
    lut = lut.to(dtype=torch.float32, device=x.device)
    max_code = float(lut.abs().max().item()) or 1.0

    scales = (x.abs().amax(dim=-1) / max_code).clamp(min=1e-8)
    codes_idx = torch.zeros(
        (n, num_groups, group_size), dtype=torch.uint8, device=x.device
    )
    for _ in range(max(1, refine_steps)):
        normed = x / scales.unsqueeze(-1)
        dist = (normed.unsqueeze(-1) - lut).abs()
        codes_idx = dist.argmin(dim=-1).to(torch.uint8)
        chosen = lut[codes_idx.long()]
        denom = (chosen * chosen).sum(dim=-1).clamp(min=1e-12)
        opt_scale = (x * chosen).sum(dim=-1) / denom
        # Round to fp16 inside the alternating loop so codes adapt to exact fp16 scale rounding.
        scales = (
            opt_scale.clamp(min=0.0, max=65504.0)
            .to(torch.float16)
            .to(torch.float32)
            .clamp(min=1e-8)
        )

    normed = x / scales.unsqueeze(-1)
    codes_idx = (normed.unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    flat_idx = codes_idx.view(n, d)
    lo = flat_idx[:, 0::2] & 0x0F
    hi = (flat_idx[:, 1::2] & 0x0F) << 4
    packed = (lo | hi).contiguous()
    scale_f16 = scales.to(torch.float16).contiguous()
    return packed, scale_f16


def quantize_ple_directory(
    src_model_dir: str | Path,
    out_dir: str | Path,
    *,
    group_size: int = 32,
    chunk_rows: int = 262_144,
    sample_rows_per_shard: int = 8_192,
) -> dict[str, float | int]:
    """Convert checkpoint ``ngram_embedding.shard_<i>.weight`` safetensors shards to ``lut4-gptq``."""
    src_dir = Path(src_model_dir)
    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    index_path = src_dir / "model.safetensors.index.json"
    if index_path.is_file():
        wmap = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(
            {src_dir / fn for k, fn in wmap.items() if "ngram_embedding" in k}
        )
    else:
        files = sorted(src_dir.glob("*.safetensors"))

    shards: dict[int, tuple[Path, str]] = {}
    weight_scale = 1.0
    for fpath in files:
        with safetensors.safe_open(str(fpath), framework="pt", device="cpu") as sf:
            for k in sf.keys():  # noqa: SIM118
                if k.endswith("ngram_embedding.weight_scale"):
                    weight_scale = float(sf.get_tensor(k).to(torch.float32).item())
                elif ".shard_" in k and k.endswith(".weight"):
                    part = int(k.split(".shard_")[1].split(".")[0])
                    shards[part] = (fpath, k)

    if not shards:
        raise ValueError(
            f"No ngram_embedding.shard_<i>.weight tensors found in {src_dir}"
        )

    # 1. Sample rows across shards to fit global 16-entry codebook.
    sampled: list[torch.Tensor] = []
    for part in sorted(shards):
        fpath, key = shards[part]
        with safetensors.safe_open(str(fpath), framework="pt", device="cpu") as sf:
            sl = sf.get_slice(key)
            nrows = sl.get_shape()[0]
            take = min(nrows, sample_rows_per_shard)
            stride = max(1, nrows // take)
            sub = sl[0 : take * stride : stride].to(torch.float32) * weight_scale
            sampled.append(sub)
    lut = fit_lut16_codebook(torch.cat(sampled, dim=0), group_size=group_size)

    # 2. Stream-quantize each shard and write safetensors + index.json.
    weight_map: dict[str, str] = {}
    total_rows = 0
    sq_err = 0.0
    sq_ref = 0.0
    for part in sorted(shards):
        fpath, key = shards[part]
        with safetensors.safe_open(str(fpath), framework="pt", device="cpu") as sf:
            sl = sf.get_slice(key)
            nrows, head_dim = sl.get_shape()
            packed_out = torch.empty((nrows, head_dim // 2), dtype=torch.uint8)
            scale_out = torch.empty(
                (nrows, head_dim // group_size), dtype=torch.float16
            )
            for r0 in range(0, nrows, chunk_rows):
                r1 = min(nrows, r0 + chunk_rows)
                chunk = sl[r0:r1].to(torch.float32) * weight_scale
                pk, sc = quantize_rows_lut4(chunk, lut, group_size=group_size)
                packed_out[r0:r1].copy_(pk)
                scale_out[r0:r1].copy_(sc)
                if r0 == 0:
                    lo = (pk & 0x0F).long()
                    hi = ((pk >> 4) & 0x0F).long()
                    nib = torch.stack((lo, hi), dim=-1).reshape(r1 - r0, head_dim)
                    rec = (
                        lut[nib].view(r1 - r0, head_dim // group_size, group_size)
                        * sc.to(torch.float32).unsqueeze(-1)
                    ).reshape(r1 - r0, head_dim)
                    sq_err += float(((chunk - rec) ** 2).sum().item())
                    sq_ref += float((chunk**2).sum().item())
        total_rows += nrows
        out_name = f"ple-lut4-{part:05d}-of-{len(shards):05d}.safetensors"
        base_pfx = key[: key.index(".shard_")]
        pfx = f"{base_pfx}.shard_{part}"
        tensors = {
            f"{pfx}.weight_packed": packed_out,
            f"{pfx}.weight_scale": scale_out,
            f"{pfx}.weight_lut": lut.clone(),
        }
        save_file(tensors, str(dst_dir / out_name))
        for tname in tensors:
            weight_map[tname] = out_name

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"format": "lut4-gptq"}, "weight_map": weight_map}, indent=2
        ),
        encoding="utf-8",
    )
    rel_rmse = math.sqrt(sq_err / max(sq_ref, 1e-12))
    return {"shards": len(shards), "total_rows": total_rows, "rel_rmse": rel_rmse}


def _pread_exact(fd: int, nbytes: int, offset: int) -> bytearray:
    buf = bytearray(nbytes)
    mv = memoryview(buf)
    read = 0
    while read < nbytes:
        chunk = os.pread(fd, nbytes - read, offset + read)
        if not chunk:
            raise EOFError(
                f"short pread at offset={offset + read}: wanted {nbytes} B, got {read} B"
            )
        mv[read : read + len(chunk)] = chunk
        read += len(chunk)
    return buf


def _drop_page_cache(fd: int, offset: int, length: int) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def quantize_ple_gguf(
    gguf_path: str | Path,
    out_dir: str | Path,
    *,
    head_dim: int = 160,
    split_parts: int = 128,
    group_size: int = 32,
    chunk_rows: int = 262_144,
    sample_rows_per_shard: int = 8_192,
) -> dict[str, float | int]:
    """Stream-quantize a lossless BF16 PLE GGUF (`per_layer_token_embd.weight`) into ``lut4-gptq`` shards.

    Uses ``os.pread`` + ``POSIX_FADV_DONTNEED`` so reading a 95.4 GiB GGUF does not pollute host page cache.
    """
    from freetoken.models.qwen4_exp.gguf import check_ple_table
    from freetoken.third_party.gguf import GGMLQuantizationType

    info = check_ple_table(str(gguf_path), head_dim=head_dim)
    if info.ggml_type != GGMLQuantizationType.BF16:
        raise ValueError(f"Expected BF16 PLE GGUF, got ggml_type={info.ggml_type}")
    cols, total_rows = info.ne
    if cols != head_dim or total_rows % split_parts != 0:
        raise ValueError(
            f"GGUF shape {(total_rows, cols)} incompatible with head_dim={head_dim}, split_parts={split_parts}"
        )
    shard_rows = total_rows // split_parts
    row_bytes = cols * 2
    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    fd = os.open(info.path, os.O_RDONLY)
    try:
        sampled: list[torch.Tensor] = []
        for part in range(split_parts):
            take = min(shard_rows, sample_rows_per_shard)
            # Read a contiguous prefix of `take` rows per shard to avoid random seeks.
            nbytes = take * row_bytes
            off = info.offset + part * shard_rows * row_bytes
            raw = _pread_exact(fd, nbytes, off)
            _drop_page_cache(fd, off, nbytes)
            sub = (
                torch.frombuffer(raw, dtype=torch.bfloat16)
                .view(take, cols)
                .to(torch.float32)
            )
            sampled.append(sub)
        lut = fit_lut16_codebook(torch.cat(sampled, dim=0), group_size=group_size)
        del sampled

        weight_map: dict[str, str] = {}
        sq_err = 0.0
        sq_ref = 0.0
        for part in range(split_parts):
            packed_out = torch.empty((shard_rows, cols // 2), dtype=torch.uint8)
            scale_out = torch.empty(
                (shard_rows, cols // group_size), dtype=torch.float16
            )
            shard_base = info.offset + part * shard_rows * row_bytes
            for r0 in range(0, shard_rows, chunk_rows):
                r1 = min(shard_rows, r0 + chunk_rows)
                nbytes = (r1 - r0) * row_bytes
                off = shard_base + r0 * row_bytes
                raw = _pread_exact(fd, nbytes, off)
                _drop_page_cache(fd, off, nbytes)
                chunk = (
                    torch.frombuffer(raw, dtype=torch.bfloat16)
                    .view(r1 - r0, cols)
                    .to(torch.float32)
                )
                pk, sc = quantize_rows_lut4(chunk, lut, group_size=group_size)
                packed_out[r0:r1].copy_(pk)
                scale_out[r0:r1].copy_(sc)
                if r0 == 0:
                    lo = (pk & 0x0F).long()
                    hi = ((pk >> 4) & 0x0F).long()
                    nib = torch.stack((lo, hi), dim=-1).reshape(r1 - r0, cols)
                    rec = (
                        lut[nib].view(r1 - r0, cols // group_size, group_size)
                        * sc.to(torch.float32).unsqueeze(-1)
                    ).reshape(r1 - r0, cols)
                    sq_err += float(((chunk - rec) ** 2).sum().item())
                    sq_ref += float((chunk**2).sum().item())
            out_name = f"ple-lut4-{part:05d}-of-{split_parts:05d}.safetensors"
            out_path = dst_dir / out_name
            pfx = f"model.layers.1.ple.ple_embedding.ngram_embedding.shard_{part}"
            tensors = {
                f"{pfx}.weight_packed": packed_out,
                f"{pfx}.weight_scale": scale_out,
                f"{pfx}.weight_lut": lut.clone(),
            }
            save_file(tensors, str(out_path))
            out_fd = os.open(str(out_path), os.O_RDONLY)
            try:
                _drop_page_cache(out_fd, 0, 0)
            finally:
                os.close(out_fd)
            for tname in tensors:
                weight_map[tname] = out_name
            print(
                f"[ple-lut4] wrote shard {part + 1}/{split_parts}: {out_name}",
                flush=True,
            )
    finally:
        os.close(fd)

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"format": "lut4-gptq"}, "weight_map": weight_map}, indent=2
        ),
        encoding="utf-8",
    )
    rel_rmse = math.sqrt(sq_err / max(sq_ref, 1e-12))
    return {"shards": split_parts, "total_rows": total_rows, "rel_rmse": rel_rmse}


def _infer_ple_geometry(model_dir: str | None) -> tuple[int, int]:
    if not model_dir:
        return 160, 128
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return 160, 128
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    tc = raw.get("text_config", raw)
    ple_dim = int(tc.get("ple_embed_dim", 2560))
    ngram_size = int(tc.get("ngram_size", 3))
    heads_per = int(tc.get("heads_per_ngram", 8))
    num_heads = (ngram_size - 1) * heads_per
    head_dim = ple_dim // num_heads
    split_parts = int(tc.get("split_ngram_parts", 128))
    return head_dim, split_parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir",
        default=None,
        help="Source checkpoint directory (with PLE shards or config.json when --ple-gguf is used)",
    )
    ap.add_argument(
        "--ple-gguf",
        default=None,
        help="Optional lossless BF16 PLE GGUF file (avoids FP8->LUT4 double quantization)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Destination directory for lut4-gptq PLE shards",
    )
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--chunk-rows", type=int, default=262_144)
    ap.add_argument(
        "--threads", type=int, default=8, help="PyTorch intra-op CPU threads"
    )
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.ple_gguf:
        head_dim, split_parts = _infer_ple_geometry(args.model_dir)
        stats = quantize_ple_gguf(
            args.ple_gguf,
            args.out_dir,
            head_dim=head_dim,
            split_parts=split_parts,
            group_size=args.group_size,
            chunk_rows=args.chunk_rows,
        )
    elif args.model_dir:
        stats = quantize_ple_directory(
            args.model_dir,
            args.out_dir,
            group_size=args.group_size,
            chunk_rows=args.chunk_rows,
        )
    else:
        ap.error("Provide either --ple-gguf or --model-dir")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
