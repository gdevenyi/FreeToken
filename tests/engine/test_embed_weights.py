"""--embed-weights host: a setup the host gather cannot serve is refused while the config is still a dataclass, not after the weight load."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig


def _config(monkeypatch, *, tie: bool, tp: int = 1, embed_weights: str = "host"):
    import freetoken.engine.config as engine_config

    hf = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, hidden_size=64, vocab_size=128, intermediate_size=128, rms_norm_eps=1e-6,
        max_position_embeddings=1024, hidden_act="silu", tie_word_embeddings=tie,
    )
    monkeypatch.setattr(engine_config, "cached_load_hf_config", lambda path: hf)
    monkeypatch.setattr(engine_config, "checkpoint_quant_config", lambda *args: None)
    return EngineConfig(
        model_path="/fake", tp_info=DistributedInfo(rank=0, size=tp), dtype=torch.bfloat16, embed_weights=embed_weights
    )


def test_a_tied_embedding_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="needs an untied embedding"):
        _config(monkeypatch, tie=True).model_config


def test_tensor_parallel_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="single GPU only"):
        _config(monkeypatch, tie=False, tp=2).model_config


def test_an_untied_single_gpu_model_and_the_gpu_default_pass(monkeypatch):
    assert not _config(monkeypatch, tie=False).model_config.tie_word_embeddings
    assert _config(monkeypatch, tie=True, tp=2, embed_weights="gpu").model_config.tie_word_embeddings
