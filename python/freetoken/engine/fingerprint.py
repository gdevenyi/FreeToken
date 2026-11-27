"""Deterministic ``system_fingerprint`` for OpenAI Reproducible Outputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from freetoken.version import __version__


@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def _resolved_path_id(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.exists(path):
        try:
            real = os.path.realpath(path)
            # Keep the last 3 path segments so snapshots (models--org--name/snapshots/<rev>)
            # stay identifiable without leaking unrelated mount prefixes.
            parts = Path(real).parts
            return "/".join(parts[-3:]) if len(parts) > 3 else real
        except OSError:
            pass
    return str(path)


def _normalized_quant_backend(quant_backend: Any) -> list[list[str | None]] | None:
    if quant_backend is None:
        try:
            from freetoken.layers.quantization.quant_backend import get_quant_backend

            quant_backend = get_quant_backend()
        except Exception:  # noqa: BLE001
            return None
    items = getattr(quant_backend, "items", None)
    if not items:
        return None
    out: list[list[str | None]] = []
    for key, name in items:
        layer, kind = key
        out.append(
            [
                getattr(layer, "value", str(layer)),
                None if kind is None else getattr(kind, "value", str(kind)),
                str(name),
            ]
        )
    return out


def compute_system_fingerprint(
    config: Any,
    *,
    quant_backend: Any = None,
    git_commit: str | None = ...,  # type: ignore[assignment]
) -> str:
    """Compute a compact, deterministic ``fp_<12-hex>`` backend fingerprint.

    Captures the model checkpoint, engine build, precision/quantization flags,
    parallelism, speculative configuration, and batch-invariance mode so clients
    using ``seed`` can tell when a backend change explains an output difference.
    Never includes ``tp_info.rank`` so every TP rank derives the exact same value.
    """
    from freetoken.models.config import (
        fp8_dense_enabled,
        fp8_dense_scale,
        fp8_lmhead_enabled,
        fp8_lmhead_scale,
        fp32_router_enabled,
        hc_int8_enabled,
        int8_dense_enabled,
        verify_invariant_enabled,
    )

    commit = _git_commit() if git_commit is ... else git_commit
    tp_info = getattr(config, "tp_info", None)
    tp_size = int(getattr(tp_info, "size", 1) or 1)
    online_mods = getattr(config, "online_quant_modules", None)
    payload = {
        "version": __version__,
        "git_commit": commit,
        "model": _resolved_path_id(getattr(config, "model_path", None)),
        "hf_overrides": getattr(config, "hf_overrides", None),
        "use_dummy_weight": bool(getattr(config, "use_dummy_weight", False)),
        "dtype": str(getattr(config, "dtype", None)),
        "quant_backend_cfg": getattr(config, "quant_backend", None),
        "quant_backend_resolved": _normalized_quant_backend(quant_backend),
        "online_quant": getattr(config, "online_quant", None),
        "online_quant_modules": (
            sorted(str(m) for m in online_mods) if online_mods else None
        ),
        "kv_cache_dtype": getattr(config, "kv_cache_dtype", None),
        "attention_backend": getattr(config, "attention_backend", None),
        "moe_strategy": getattr(config, "moe_strategy", None),
        "moe_ep": bool(getattr(config, "moe_ep", False)),
        "ple_backend": getattr(config, "ple_backend", None),
        "ple_table": _resolved_path_id(getattr(config, "ple_table", None)),
        "fp8_dense": fp8_dense_enabled(),
        "fp8_dense_scale": fp8_dense_scale() if fp8_dense_enabled() else None,
        "fp8_lmhead": fp8_lmhead_enabled(),
        "fp8_lmhead_scale": fp8_lmhead_scale() if fp8_lmhead_enabled() else None,
        "int8_dense": int8_dense_enabled(),
        "hc_int8": hc_int8_enabled(),
        "fp32_router": fp32_router_enabled(),
        "verify_invariant": verify_invariant_enabled(),
        "tp_size": tp_size,
        "speculative_algorithm": getattr(config, "speculative_algorithm", None),
        "speculative_draft_model": _resolved_path_id(
            getattr(config, "speculative_draft_model_path", None)
        ),
        "speculative_num_steps": getattr(config, "speculative_num_steps", None),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"fp_{digest}"
