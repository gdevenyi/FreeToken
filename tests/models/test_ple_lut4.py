"""End-to-end unit tests for 4-bit lut4-gptq PLE quantization, /dev/shm loading, and gather."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from freetoken.models.qwen4_exp.ple import PinnedUVATable
from freetoken.models.qwen4_exp.weight import (
    check_ple_table_lut4,
    is_lut4_ple_path,
    load_ple_table_lut4,
)
from safetensors.torch import save_file


def _load_quantizer_module():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "quantize_ple_lut4", root / "tools" / "quantize_ple_lut4.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lut4_ple_quantize_load_and_gather_roundtrip(tmp_path):
    qmod = _load_quantizer_module()
    torch.manual_seed(7)

    src_dir = tmp_path / "src_ple"
    out_dir = tmp_path / "lut4_ple"
    src_dir.mkdir()

    # Create a synthetic 2-shard BF16 PLE checkpoint (2 x 256 rows x 160 head_dim)
    head_dim = 160
    shard_rows = 256
    s0 = torch.randn((shard_rows, head_dim), dtype=torch.bfloat16) * 0.25
    s1 = torch.randn((shard_rows, head_dim), dtype=torch.bfloat16) * 0.25
    ref_full = torch.cat([s0, s1], dim=0).to(torch.float32)

    save_file(
        {
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": s0,
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight": s1,
        },
        str(src_dir / "model-ple-00001-of-00001.safetensors"),
    )

    stats = qmod.quantize_ple_directory(
        src_dir, out_dir, group_size=32, chunk_rows=128, sample_rows_per_shard=256
    )
    assert stats["shards"] == 2
    assert stats["total_rows"] == 512
    assert stats["rel_rmse"] < 0.14  # 4-bit Normal Lloyd-Max theoretical floor ~0.11

    assert is_lut4_ple_path(str(out_dir))
    total_rows, group_size = check_ple_table_lut4(
        str(out_dir), head_dim=head_dim, rows_needed=512
    )
    assert (total_rows, group_size) == (512, 32)

    qwen4_args = SimpleNamespace(split_ngram_parts=2, ngram_head_dim=head_dim)
    ple_table = load_ple_table_lut4(
        str(out_dir), qwen4_args, pin=False, shm_dir=str(tmp_path)
    )
    # 90 bytes/row vs 320 bytes/row BF16 = 3.556x memory reduction!
    assert ple_table.bank.tensor.shape == (512, 90)
    assert ple_table.bank.tensor.dtype == torch.uint8

    uva = PinnedUVATable(
        ple_table.bank.tensor,
        weight_lut=ple_table.weight_lut,
        group_size=ple_table.group_size,
        head_dim=ple_table.head_dim,
        device=torch.device("cpu"),
        prefetch=False,
    )
    query_ids = torch.tensor([[0, 17, 255, 511], [-1, 512, 300, 42]], dtype=torch.int64)
    gathered = uva.lookup(query_ids)
    assert gathered.shape == (2, 4 * head_dim)
    gathered_rows = gathered.view(2, 4, head_dim).to(torch.float32)

    # Out-of-range rows (-1 and 512) must gather exact zeros!
    assert torch.all(gathered_rows[1, 0] == 0.0)
    assert torch.all(gathered_rows[1, 1] == 0.0)

    # Valid rows match reference within expected 4-bit group-32 tolerance
    valid_ids = torch.tensor([0, 17, 255, 511, 300, 42], dtype=torch.long)
    got_valid = torch.cat([gathered_rows[0], gathered_rows[1, 2:]], dim=0)
    ref_valid = ref_full.index_select(0, valid_ids)
    rel_err = torch.linalg.norm(got_valid - ref_valid) / torch.linalg.norm(ref_valid)
    assert float(rel_err.item()) < 0.14


def test_quantize_ple_gguf_streams_bf16_to_lut4_shards(tmp_path, monkeypatch):
    mod = _load_quantizer_module()

    torch.manual_seed(99)
    total_rows, head_dim, split_parts = 512, 160, 4
    ref = (torch.randn(total_rows, head_dim) * 0.08).to(torch.bfloat16)
    raw_file = tmp_path / "fake_ple.gguf_file"
    header_pad = b"GGUF_FAKE_HEADER" * 4  # 64-byte offset
    raw_file.write_bytes(header_pad + bytes(ref.contiguous().view(torch.uint8).numpy()))

    from freetoken.third_party.gguf import GGMLQuantizationType

    fake_info = SimpleNamespace(
        path=str(raw_file),
        offset=len(header_pad),
        ne=(head_dim, total_rows),
        ggml_type=GGMLQuantizationType.BF16,
        nbytes=total_rows * head_dim * 2,
    )
    import freetoken.models.qwen4_exp.gguf as q4_gguf

    monkeypatch.setattr(q4_gguf, "check_ple_table", lambda *_a, **_kw: fake_info)

    out_dir = tmp_path / "ple_lut4_from_gguf"
    stats = mod.quantize_ple_gguf(
        raw_file,
        out_dir,
        head_dim=head_dim,
        split_parts=split_parts,
        group_size=32,
        chunk_rows=128,
        sample_rows_per_shard=64,
    )
    assert stats["shards"] == split_parts
    assert stats["total_rows"] == total_rows
    assert float(stats["rel_rmse"]) < 0.13

    args = SimpleNamespace(split_ngram_parts=split_parts, ngram_head_dim=head_dim)
    table = load_ple_table_lut4(str(out_dir), args, pin=False, shm_dir=str(tmp_path))
    assert table.bank.tensor.shape == (total_rows, 90)
    assert table.weight_lut is not None and table.weight_lut.shape == (16,)
