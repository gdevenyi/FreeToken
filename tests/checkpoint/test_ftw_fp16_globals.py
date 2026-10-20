"""An FTW converted before #739 stores NVFP4 global scales as fp16: the reader upcasts them to the fp32 the kernels read."""

import logging

import torch

from freetoken.checkpoint import ftw
from freetoken.checkpoint.ftw import FTWReader, FTWWriter, iter_ftw_weights, layer_bank_entry_name, load_ftw_banks

E = 4


def _globals(rows: int, seed: int) -> torch.Tensor:
    # the served magnitudes (2e-5 to 4e-4) sit at fp16's normal minimum
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(E, rows, generator=g) * 4e-4 + 2e-5).to(torch.float16)


def _write(path, *, per_layer: bool) -> dict[str, torch.Tensor]:
    src = {
        "gate_up_global": _globals(6, 0),
        "down_global": _globals(8, 1),
        "gate_up": torch.arange(E * 6 * 4, dtype=torch.uint8).view(E, 6, 4),
    }
    writer = FTWWriter(str(path))
    writer.add_tensor("model.layers.0.mlp.shared_expert.gate_proj.weight_global", torch.tensor(3e-4, dtype=torch.float16))
    writer.add_tensor("model.layers.0.input_layernorm.weight", torch.ones(8, dtype=torch.float16))
    for name, tensor in src.items():
        writer.add_tensor(layer_bank_entry_name(name, 0) if per_layer else name, tensor, kind="experts_bank")
    writer.finalize({})
    return src


def test_fp16_globals_load_as_fp32(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.setenv("FREETOKEN_BANK_CUDA_ALLOC", "0")
    monkeypatch.setattr(ftw, "_FP16_GLOBALS_WARNED", set(), raising=False)
    for per_layer in (True, False):
        path = tmp_path / ("per_layer" if per_layer else "flat")
        src = _write(path, per_layer=per_layer)

        with caplog.at_level(logging.WARNING):
            caplog.clear()
            FTWReader(str(path)).close()
            dense = dict(iter_ftw_weights(str(path)))
            # the engine loads the banks under inference mode, which the bank reader threads do not inherit
            with torch.inference_mode():
                banks = load_ftw_banks(str(path), num_layers=1)
        assert sum("fp16" in r.getMessage() for r in caplog.records) == 1

        g = dense["model.layers.0.mlp.shared_expert.gate_proj.weight_global"]
        assert g.dtype == torch.float32 and g.item() == torch.tensor(3e-4, dtype=torch.float16).item()
        # only the NVFP4 globals are upcast
        assert dense["model.layers.0.input_layernorm.weight"].dtype == torch.float16

        for name in ("gate_up_global", "down_global"):
            bank = banks.sources[name][0]
            assert bank.dtype == torch.float32
            assert torch.equal(bank, src[name].float())
        assert torch.equal(banks.sources["gate_up"][0], src["gate_up"])
