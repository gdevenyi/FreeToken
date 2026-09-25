"""``ft bench decode``: end-to-end decode throughput, and A/B across engine flags.

``ft bench bw`` measures one kernel in isolation over a small synthetic bank. That is
the right shape for calibrating a bandwidth ratio and the wrong shape for answering
"is this configuration faster", because a serving step is the kernel *plus* the PCIe
gather it contends with, the KV traffic, the GPU<->CPU handshake and the scheduler --
on the real model, at the real cache size. The two can disagree completely: on one
2-socket box a change measured +30% on the microbenchmark and -6.7% on tokens/s.

So this loads the actual model and generates.

    ft bench decode --model DIR
    ft bench decode --model DIR --compare moe-strategy=hybrid,offload
    ft bench decode --model DIR --compare moe-cache-rate=0.1,0.25,0.5 --cycles 3
    ft bench decode --model DIR --context-tokens 131072 --text-model-only \
        --set max-seq-len-override=132000 --set kv-reserve-tokens=132000

Two things it does that a hand-rolled loop usually does not:

* **Decode is isolated from prefill** by timing the same prompt twice, once with
  ``max_tokens=1`` and once with ``max_tokens=n``, and taking ``(n-1)/(t_n - t_1)``.
  Prefill cost cancels, so a long prompt does not quietly flatter the result.
* **Variants alternate**, one full pass per cycle rather than all runs of A then all
  of B. Thermal drift, page-cache state and whatever else the box is doing move over
  minutes; blocked runs attribute that drift to the variant.

Each measurement runs in a fresh subprocess. Expert banks are pinned and cannot be
unregistered, so tearing an engine down in-process does not reliably give the memory
back -- the second variant would be measuring the first one's leftovers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

DEFAULT_PROMPT = (
    "Write a detailed technical explanation of how a mixture-of-experts transformer "
    "routes tokens to experts, and why that makes memory bandwidth the bottleneck."
)


# SchedulerConfig fields LLM() sets itself, or that a FLAG=VALUE string cannot build.
_RESERVED_FIELDS = {
    "model_path": "use --model",
    "tp_info": "the bench runs one GPU",
    "dtype": "LLM() serves bf16",
    "offline_mode": "LLM() sets it",
    "mm": "use --text-model-only",
    "_unique_suffix": "internal",
}


def _field_types() -> dict[str, str]:
    """``{field: annotation}`` for every SchedulerConfig field ``--set``/``--compare`` may name."""
    from dataclasses import fields

    from freetoken.scheduler.config import SchedulerConfig

    return {f.name: str(f.type) for f in fields(SchedulerConfig) if f.init}


def _coerce(v: str, typ: str, key: str = "value"):
    """A CLI string to the type the SchedulerConfig field annotation ``typ`` declares.

    Typed by the field, not guessed from the value: ``moe_cpu_layers`` is a string spec,
    and ``8`` handed to it as an int crashes the engine's ``.strip()``.
    """
    parts = {t.strip() for t in typ.split("|")}
    raw = v.strip()
    if "None" in parts and raw.lower() in ("none", "null", ""):
        return None
    base = parts - {"None"}
    if base == {"str"}:
        return raw
    try:
        if base == {"bool"}:
            low = raw.lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError(raw)
        if base == {"int"}:
            return int(raw)
        if base == {"float"}:
            return float(raw)
    except ValueError:
        raise SystemExit(f"{key}={v!r}: expected {typ}") from None
    raise SystemExit(f"{key} ({typ}) cannot be set from the command line")


def _flag_key(flag: str, types: dict[str, str]) -> str:
    key = flag.strip().lstrip("-").replace("-", "_")
    if key in _RESERVED_FIELDS:
        raise SystemExit(f"{flag}: not settable here ({_RESERVED_FIELDS[key]})")
    if key not in types:
        raise SystemExit(f"{flag}: not an engine config field")
    return key


def _decode_rate(concurrency: int, tokens: int, t_prefill: float, t_total: float) -> float | None:
    """Aggregate decode tokens/s from a max_tokens=1 run and a max_tokens=``tokens`` run.

    Prefill (and the first token) is in both, so the difference is ``tokens - 1`` decode
    steps per stream; None when timing noise leaves no positive difference.
    """
    decode_s = t_total - t_prefill
    if decode_s <= 0:
        return None
    return concurrency * (tokens - 1) / decode_s


def _token_prompts(tokenizer, text: str, context_tokens: int, concurrency: int) -> list[list[int]]:
    """``concurrency`` token-id prompts of exactly ``context_tokens`` tokens each.

    ``text`` is tiled to length; each stream ends in its own " (variant i)" tail, as the
    text prompts do, so the streams are separate requests over a shared prefix.
    """
    body = tokenizer.encode(text, add_special_tokens=False)
    if not body:
        raise ValueError("the prompt tokenizes to nothing")
    prompts = []
    for i in range(concurrency):
        tail = tokenizer.encode(f" (variant {i})", add_special_tokens=False)
        n = context_tokens - len(tail)
        if n < 1:
            raise ValueError(f"--context-tokens {context_tokens} is shorter than the prompt tail")
        prompts.append((body * (n // len(body) + 1))[:n] + tail)
    return prompts


def measure_decode_tps(model_path: str, engine_kwargs: dict, prompt: str,
                       tokens: int, samples: int, concurrency: int = 1,
                       context_tokens: int | None = None,
                       text_model_only: bool = False) -> dict:
    """Decode tokens/s for one engine configuration, median over ``samples``.

    ``concurrency`` generates that many streams at once, so decode steps carry that
    many tokens. Anything whose cost depends on the decode batch -- expert dedup on
    the CPU MoE path, CUDA-graph batch selection, the scheduler itself -- is invisible
    at concurrency 1, which is the single-stream latency case, not the serving case.
    Reported tokens/s is aggregate across the streams. ``context_tokens`` tiles the
    prompt to that many tokens, so decode is measured at that context length.
    """
    import torch

    from freetoken.core import SamplingParams
    from freetoken.llm import LLM
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig

    # `ft serve` defaults the offload-family backends to --moe-cache-auto when no
    # cache-sizing flag is given (prepare_server_args); the offline LLM path does not,
    # so a bare `--compare moe-strategy=hybrid,offload` would die on moe_cache_size=0.
    if not any(k in engine_kwargs
               for k in ("moe_cache_size", "moe_cache_rate", "moe_cache_auto")):
        engine_kwargs = {**engine_kwargs, "moe_cache_auto": True}
    if concurrency > 1:
        # Without room for the streams the scheduler just serializes them and the
        # decode batch never grows, which silently measures concurrency 1.
        engine_kwargs.setdefault("max_running_req", concurrency)
        engine_kwargs.setdefault("cuda_graph_max_bs", concurrency)

    if text_model_only:
        engine_kwargs = {**engine_kwargs,
                         "mm": MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS))}

    llm = LLM(model_path, dtype=torch.bfloat16, **engine_kwargs)
    # `ignore_eos` so every run generates exactly `tokens` -- otherwise an early stop
    # silently shortens the measurement and inflates the rate.
    one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    many = SamplingParams(max_tokens=tokens, temperature=0.0, ignore_eos=True)

    # Distinct prompts: the radix cache would share the KV of identical prefixes, so
    # N copies of one prompt measures one stream plus N-1 cache hits.
    if context_tokens:
        prompts = _token_prompts(llm.tokenizer, prompt, context_tokens, concurrency)
    else:
        prompts = [f"{prompt} (variant {i})" for i in range(concurrency)]
    llm.generate(prompts, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True))

    rates = []
    for _ in range(samples):
        t0 = time.perf_counter()
        llm.generate(prompts, one)
        t_prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        llm.generate(prompts, many)
        t_total = time.perf_counter() - t0
        rate = _decode_rate(concurrency, tokens, t_prefill, t_total)
        if rate is not None:
            rates.append(rate)
    if not rates:
        raise RuntimeError("no usable timing samples")
    return {
        "tps": statistics.median(rates),
        "samples": [round(r, 2) for r in rates],
        "prefill_s": round(t_prefill, 3),
    }


def _run_worker() -> int:
    """Hidden per-measurement subprocess: read the spec from stdin, emit one JSON line."""
    try:
        spec = json.loads(sys.stdin.read())
        out = measure_decode_tps(spec["model"], spec["kwargs"], spec["prompt"],
                                 spec["tokens"], spec["samples"],
                                 spec.get("concurrency", 1),
                                 spec.get("context_tokens"),
                                 spec.get("text_model_only", False))
    except Exception as e:  # noqa: BLE001 - reported to the parent, not swallowed
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1
    print(json.dumps(out), flush=True)
    return 0


def _measure_in_subprocess(spec: dict, quiet: bool, timeout: float | None = None) -> dict:
    # The spec goes over stdin: a long prompt in argv hits the 128 KiB per-argument limit.
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "freetoken.moe.bench_decode", "--_worker"],
            input=json.dumps(spec), capture_output=True, text=True, check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"worker timed out after {timeout:.0f}s"}
    except OSError as e:
        return {"error": f"could not start the worker: {e}"}
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
    if line is None:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return {"error": "no result from worker:\n    " + "\n    ".join(tail)}
    out = json.loads(line)
    if "error" in out and not quiet:
        print(f"      worker: {out['error']}", file=sys.stderr)
    return out


def _variants(compare: str | None, extra: list[str],
              types: dict[str, str] | None = None) -> list[tuple[str, dict]]:
    types = _field_types() if types is None else types
    base = {}
    for kv in extra:
        k, sep, v = kv.partition("=")
        if not sep:
            raise SystemExit(f"--set wants FLAG=VALUE (got {kv!r})")
        key = _flag_key(k, types)
        base[key] = _coerce(v, types[key], key)
    if not compare:
        return [("baseline", base)]
    flag, _, values = compare.partition("=")
    if not values:
        raise SystemExit("--compare wants FLAG=value1,value2")
    key = _flag_key(flag, types)
    return [(f"{flag}={v}", {**base, key: _coerce(v, types[key], key)})
            for v in values.split(",")]


def main(argv: list[str] | None = None, prog: str = "ft bench decode") -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        return _run_worker()

    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="local checkpoint directory")
    p.add_argument("--compare", default=None, metavar="FLAG=A,B",
                   help="engine flag to vary, e.g. 'moe-strategy=hybrid,offload'")
    p.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                   help="engine flag held fixed across variants (repeatable)")
    p.add_argument("--cycles", type=int, default=2,
                   help="alternating passes over the variants (default 2)")
    p.add_argument("--samples", type=int, default=3,
                   help="timed generations per load (default 3)")
    p.add_argument("--tokens", type=int, default=128, help="tokens per generation")
    p.add_argument("--concurrency", type=int, default=1, metavar="N",
                   help="generate N streams at once, so decode steps carry N tokens "
                        "(default 1). Reported tokens/s is aggregate. Anything whose "
                        "cost depends on the decode batch is invisible at 1")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file", default=None, metavar="PATH",
                   help="read the prompt text from PATH instead of --prompt")
    p.add_argument("--context-tokens", type=int, default=None, metavar="N",
                   help="tile the prompt to exactly N tokens, so decode is measured at that "
                        "context length (size the KV with --set as for ft serve)")
    p.add_argument("--text-model-only", action="store_true",
                   help="skip building the encoder towers, as ft serve --text-model-only")
    p.add_argument("--timeout", type=float, default=None, metavar="SEC",
                   help="give up on a measurement after SEC seconds (default: no limit)")
    p.add_argument("-o", "--out", default=None, help="write results as JSON")
    p.add_argument("-q", "--quiet", action="store_true")
    ns = p.parse_args(argv)

    if not os.path.isdir(ns.model):
        raise SystemExit(f"--model must be a local directory (got {ns.model!r})")

    if ns.context_tokens is not None and ns.context_tokens < 2:
        raise SystemExit("--context-tokens must be at least 2")
    prompt = ns.prompt
    if ns.prompt_file:
        with open(ns.prompt_file) as f:
            prompt = f.read()
    variants = _variants(ns.compare, ns.set)
    results: dict[str, list[float]] = {name: [] for name, _ in variants}
    print(f"  {len(variants)} variant(s) x {ns.cycles} cycle(s), "
          f"{ns.samples} timed generations of {ns.tokens} tokens each"
          + (f", {ns.concurrency} streams at once" if ns.concurrency > 1 else ""))
    print("  each measurement reloads the model in a fresh process\n")

    for cycle in range(1, ns.cycles + 1):
        for name, kwargs in variants:
            spec = {"model": ns.model, "kwargs": kwargs, "prompt": prompt,
                    "tokens": ns.tokens, "samples": ns.samples,
                    "concurrency": ns.concurrency, "context_tokens": ns.context_tokens,
                    "text_model_only": ns.text_model_only}
            t0 = time.perf_counter()
            out = _measure_in_subprocess(spec, ns.quiet, ns.timeout)
            dt = time.perf_counter() - t0
            if "error" in out:
                print(f"  cycle {cycle}  {name:<28} FAILED ({dt:.0f}s)")
                continue
            results[name].append(out["tps"])
            print(f"  cycle {cycle}  {name:<28} {out['tps']:7.2f} tok/s  ({dt:.0f}s)")

    print()
    rows = [(n, v) for n, v in results.items() if v]
    if not rows:
        print("  no successful measurements")
        return 1
    best = max(statistics.median(v) for _, v in rows)
    print(f"  {'variant':<28} {'median':>9} {'spread':>17}   vs best")
    for name, vals in rows:
        med = statistics.median(vals)
        spread = f"{min(vals):.1f}-{max(vals):.1f}" if len(vals) > 1 else "-"
        rel = "best" if med == best else f"{(med / best - 1) * 100:+.1f}%"
        print(f"  {name:<28} {med:8.2f}  {spread:>17}   {rel}")
    if any(len(v) < 2 for _, v in rows):
        print("\n  Only one cycle per variant: nothing separates a real difference from "
              "drift.\n  Use --cycles 2 or more before believing a small gap.")

    if ns.out:
        with open(ns.out, "w") as f:
            json.dump({"model": ns.model, "tokens": ns.tokens,
                       "concurrency": ns.concurrency, "context_tokens": ns.context_tokens,
                       "results": {n: v for n, v in results.items()}}, f, indent=2)
            f.write("\n")
        print(f"\n  saved: {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
