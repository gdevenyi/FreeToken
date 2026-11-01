"""Does qwen4_exp's iter_weights emit exactly the state dict the engine builds? (names, shapes, dtypes)

CPU only, no tensor data: the model is built on the meta device the way EngineConfig builds it,
and the REAL loader runs over the checkpoint with safe_open patched to hand out meta tensors
shaped and typed from each shard's safetensors header. So the check follows the loader's own
renames, fusions, TP sharding and load-time FP8 splits; nothing is re-derived here.

usage: FREETOKEN_FP8_DENSE=0|1 FREETOKEN_FP8_LMHEAD=0|1 python keycheck2.py <snapshot> <rank> <world>
Routed experts (offload banks) and the vision tower (streamed separately) are out of scope.
"""

import sys

import torch

folder, rank, world = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

from freetoken.distributed import set_tp_info

set_tp_info(rank=rank, size=world)

from freetoken.distributed import try_get_tp_info
from freetoken.engine.config import EngineConfig
from freetoken.engine.engine import _decode_target
from freetoken.layers import rotary
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
from freetoken.models import create_model
from freetoken.models.qwen4_exp import weight as W
from freetoken.utils.torch_utils import torch_dtype

_DT = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}

config = EngineConfig(
    model_path=folder,
    tp_info=try_get_tp_info(),
    dtype=torch.bfloat16,
    moe_strategy="offload",
    mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
)
object.__setattr__(config.model_config, "moe_strategy", "offload")
object.__setattr__(config.model_config, "decode_target", _decode_target(config))
rotary.set_rope_device(torch.device("cpu"))
rotary.get_rope.cache_clear()
with torch.device("meta"), torch_dtype(torch.bfloat16):
    built = {
        k: (tuple(v.shape), v.dtype)
        for k, v in create_model(config.model_config).state_dict().items()
        if ".mlp.experts." not in k
    }


class _MetaOpen:
    def __init__(self, path, framework="pt", device="cpu"):
        header, _ = W._safetensors_header(path)
        self.entries = {k: v for k, v in header.items() if k != "__metadata__"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return list(self.entries)

    def get_tensor(self, name):
        e = self.entries[name]
        return torch.empty(e["shape"], dtype=_DT[e["dtype"]], device="meta")


W.safetensors.safe_open = _MetaOpen
emitted = {}
for name, t in W.iter_weights(
    folder,
    torch.device("meta"),
    include_moe_experts=False,
    include_non_moe=True,
    include_vision=False,
):
    assert name not in emitted, f"emitted twice: {name}"
    emitted[name] = (tuple(t.shape), t.dtype)

missing = sorted(set(built) - set(emitted))
unexpected = sorted(set(emitted) - set(built))
mismatch = sorted(k for k in set(built) & set(emitted) if built[k] != emitted[k])
# buffers the model owns but never loads (rope tables, counters) show up as "missing"; list them
print(
    f"rank {rank}/{world}: built {len(built)} emitted {len(emitted)} "
    f"missing {len(missing)} unexpected {len(unexpected)} shape/dtype mismatch {len(mismatch)}"
)
for k in missing[:12]:
    print("  MISSING   ", k, built[k])
for k in unexpected[:12]:
    print("  UNEXPECTED", k, emitted[k])
for k in mismatch[:12]:
    print("  MISMATCH  ", k, "built", built[k], "emitted", emitted[k])
ok = not unexpected and not mismatch
print("KEYCHECK2", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
