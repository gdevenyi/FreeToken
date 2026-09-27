from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


ENV_PREFIX = "FREETOKEN_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)
EnvStr = partial(EnvVar[str], fn=str)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    # None = unset -> resolved from the model's generation_config.json sampling defaults
    # (sglang's sampling_defaults='model'); set the env var to override.
    SHELL_TOP_K = EnvInt(None)
    SHELL_TOP_P = EnvFloat(None)
    SHELL_TEMPERATURE = EnvFloat(None)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    # GatedDeltaNet recurrent (SSM) state dtype: float32 (default) | bfloat16 | float16.
    # fp32 matches the Qwen3.x configs (mamba_ssm_dtype); fp16/bf16 halves the GDN state
    # pool at some precision cost on the long recurrence (mirrors SGLang's mamba_ssm_dtype).
    MAMBA_SSM_DTYPE = EnvStr("float32")
    # cap the longest prefill warmup forward; 0 = the engine's own chunk cap
    WARMUP_MAX_LEN = EnvInt(0)
    # GPU-decode MoE: run a layer's expert miss copy on a side stream under the shared expert
    MOE_COPY_OVERLAP = EnvBool(False)
    # GPU-decode MoE cross-layer prefetch: off | measure (predict and count, no copies) | on (predict and copy)
    MOE_PREFETCH = EnvStr("off")
    # router-lookahead candidates per layer, and the non-resident ones kept per target layer
    # (0 = 3 before GDN layers, 4 before full-attention layers)
    MOE_PREFETCH_K = EnvInt(16)
    MOE_PREFETCH_BUDGET = EnvInt(0)
    # measure mode: also log the per-layer breakdown with every summary
    MOE_PREFETCH_DEBUG = EnvBool(False)
    # debug: check every GPU decode layer call's slot maps (meta) and slot bytes before/after the GEMM (1)
    MOE_PREFETCH_VERIFY = EnvStr("0")
    # routed experts a call byte-checks (scratch rows, ~2.8 MB each on Qwen3.8-Flash)
    MOE_PREFETCH_VERIFY_ROWS = EnvInt(20)
    # debug: record every expert slot writer and audit every held slot each N decode steps (0 = off)
    MOE_SLOT_AUDIT = EnvInt(0)

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
