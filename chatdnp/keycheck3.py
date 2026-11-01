"""Does qwen4_exp's iter_weights emit exactly the state dict the engine builds? (names, shapes, dtypes)

CPU only, no tensor data: the model is built on the meta device the way EngineConfig builds it,
and the REAL loader runs over the checkpoint with safe_open patched to hand out meta tensors
shaped and typed from each shard's safetensors header. So the check follows the loader's own
renames, fusions, TP sharding and load-time FP8 splits; nothing is re-derived here.

usage: FREETOKEN_FP8_DENSE=0|1 FREETOKEN_FP8_LMHEAD=0|1 python keycheck3.py <snapshot> <rank> <world> [vision]
Routed experts are checked through the real piece reader for one expert: the sliced pieces must
fit the bank layout of the kernel that the build selected (one slicer, nvfp4_banks._tp_shard).
Shapes only: a TP value bug such as a row-parallel bias summed per rank is not visible here.
"""

import sys

import torch

folder, rank, world = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
VISION = len(sys.argv) > 4 and sys.argv[4] == "vision"

from freetoken.distributed import set_tp_info  # noqa: E402

set_tp_info(rank=rank, size=world)

from freetoken.engine.config import EngineConfig  # noqa: E402
from freetoken.engine.engine import _decode_target  # noqa: E402
from freetoken.layers import rotary  # noqa: E402
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig  # noqa: E402
from freetoken.models import create_model  # noqa: E402
from freetoken.models.qwen4_exp import weight as W  # noqa: E402
from freetoken.utils.torch_utils import torch_dtype  # noqa: E402
from freetoken.distributed import try_get_tp_info  # noqa: E402

_DT = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "U8": torch.uint8,
    "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool,
}

config = EngineConfig(
    model_path=folder, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy="offload",
    mm=MultimodalConfig() if VISION else MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
)
object.__setattr__(config.model_config, "moe_strategy", "offload")
object.__setattr__(config.model_config, "decode_target", _decode_target(config))
rotary.set_rope_device(torch.device("cpu"))
rotary.get_rope.cache_clear()
with torch.device("meta"), torch_dtype(torch.bfloat16):
    model = create_model(config.model_config)
    built = {
        k: (tuple(v.shape), v.dtype)
        for k, v in model.state_dict().items()
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
    folder, torch.device("meta"), include_moe_experts=False, include_non_moe=True,
    include_vision=VISION,
):
    assert name not in emitted, f"emitted twice: {name}"
    emitted[name] = (tuple(t.shape), t.dtype)

missing = sorted(set(built) - set(emitted))
unexpected = sorted(set(emitted) - set(built))
_FLOATS = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def _bad(b, e):
    # shapes must agree; a float->float difference is a cast in the materialize step (A_log and
    # dt_bias are fp32 params by design), but fp8/int payloads must arrive in their own dtype
    return b[0] != e[0] or (b[1] != e[1] and not (b[1] in _FLOATS and e[1] in _FLOATS))


mismatch = sorted(k for k in set(built) & set(emitted) if _bad(built[k], emitted[k]))
# buffers the model owns but never loads (rope tables, counters) show up as "missing"; list them
print(f"rank {rank}/{world}: built {len(built)} emitted {len(emitted)} "
      f"missing {len(missing)} unexpected {len(unexpected)} shape/dtype mismatch {len(mismatch)}")
for k in missing[:12]:
    print("  MISSING   ", k, built[k])
for k in unexpected[:12]:
    print("  UNEXPECTED", k, emitted[k])
for k in mismatch[:12]:
    print("  MISMATCH  ", k, "built", built[k], "emitted", emitted[k])
# tower tensors must be present when the tower is built, and only then
vision_keys = [k for k in built if k.startswith("visual.")]
print(f"vision: built {len(vision_keys)} emitted {sum(k.startswith('visual.') for k in emitted)}")
assert bool(vision_keys) == VISION, f"vision mode {VISION} but {len(vision_keys)} tower tensors built"

# the real piece reader, under the kernel the build selected
experts = model.model.layers.op_list[0].mlp.experts
method = experts.quant_method
print(f"expert kernel selected on this host: {method.kernel.name} (tp {method.cfg.tp_size}, local I {method.cfg.local_intermediate})")
from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces  # noqa: E402

pieces = iter_nvfp4_expert_pieces(
    folder, config.model_config, W.nvfp4_expert_spec(folder, config.model_config), layers=[0],
    drop_page_cache=lambda path: None, primary=False,
)
_layer, e0, e1, piece = next(iter(pieces))
out = {
    role: torch.empty((e1 - e0, *spec.shape), dtype=spec.dtype, device="meta")
    for role, spec in method.layout().items()
    if not spec.resident
}
method.pack(piece, out)  # copy_ refuses a shape that does not fit the bank
print(f"expert pieces fit the banks: {sorted((r, tuple(t.shape)) for r, t in out.items())}")

ok = not unexpected and not mismatch and not missing
print("KEYCHECK3", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
