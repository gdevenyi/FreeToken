"""DFlash 2 draft checkpoint reader: safetensors with HF names, q/k/v and gate/up fused as the model names them."""

from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import MergeRule, iter_merged_tensors, iter_weight_files, shard_tensor

_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


def iter_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """This TP rank's tensors: q/k/v, gate/up, o and down sharded before the merge, so a merged tensor holds this rank's heads of each part; fc, the convs, the selector and the norms stay whole."""
    tp = get_tp_info()

    def tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in iter_weight_files(model_path):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    # shard_tensor clones; the config check made the TP size divide the heads, so no KV head is replicated
                    if tp.size > 1:
                        tensor = shard_tensor(name, tensor, rank=tp.rank, world_size=tp.size, num_kv_heads=None)
                    yield name, tensor

    yield from iter_merged_tensors(tensors(), _MERGE_RULES, model_name="dflash")
