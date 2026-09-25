"""`ft bench decode` argument handling and arithmetic, without a GPU or a model."""

from __future__ import annotations

import json
import subprocess

import pytest

from freetoken.moe import bench_decode
from freetoken.moe.bench_decode import _coerce, _decode_rate, _token_prompts, _variants


@pytest.fixture(scope="module")
def types():
    return bench_decode._field_types()


def test_string_fields_stay_strings(types):
    """moe_cpu_layers is a str spec; an int there crashes the engine's .strip()."""
    [(_, kw)] = _variants(None, ["moe-cpu-layers=8"], types)
    assert kw == {"moe_cpu_layers": "8"}
    runs = _variants("moe-cpu-layers=4,0.5", [], types)
    assert [kw["moe_cpu_layers"] for _, kw in runs] == ["4", "0.5"]


def test_typed_fields_are_coerced(types):
    [(_, kw)] = _variants(None, ["moe-cache-rate=0.25", "moe-cache-auto=false",
                                 "max-seq-len-override=none", "--moe-cpu-threads=20"], types)
    assert kw == {"moe_cache_rate": 0.25, "moe_cache_auto": False,
                  "max_seq_len_override": None, "moe_cpu_threads": 20}


def test_compare_names_each_variant(types):
    runs = _variants("moe-strategy=hybrid,offload", ["ple-backend=disk"], types)
    assert runs == [("moe-strategy=hybrid", {"ple_backend": "disk", "moe_strategy": "hybrid"}),
                    ("moe-strategy=offload", {"ple_backend": "disk", "moe_strategy": "offload"})]


@pytest.mark.parametrize("argv", [["no-such-flag=1"], ["dtype=float16"], ["mm=x"], ["moe-cache-size"]])
def test_bad_set_flags_fail_before_any_worker_runs(types, argv):
    with pytest.raises(SystemExit):
        _variants(None, argv, types)


def test_a_value_of_the_wrong_type_is_refused():
    with pytest.raises(SystemExit):
        _coerce("lots", "int", "moe_cache_size")
    with pytest.raises(SystemExit):
        _coerce("maybe", "bool", "moe_cache_auto")
    with pytest.raises(SystemExit):
        _coerce("1", "List[int] | None", "cuda_graph_bs")


def test_decode_rate_removes_prefill():
    # 2 streams, 128 tokens: 127 decode steps each over the 2.54 s the long run adds.
    assert _decode_rate(2, 128, 1.0, 3.54) == pytest.approx(2 * 127 / 2.54)
    assert _decode_rate(1, 128, 1.0, 1.0) is None


class _CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def test_token_prompts_hit_the_requested_length():
    prompts = _token_prompts(_CharTokenizer(), "abc", 50, 3)
    assert [len(p) for p in prompts] == [50, 50, 50]
    assert len({tuple(p) for p in prompts}) == 3
    assert prompts[0][:6] == [ord(c) for c in "abcabc"]


def test_spec_reaches_the_worker_over_stdin(monkeypatch):
    """A 200 KB prompt cannot ride in argv (128 KiB per-argument limit)."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["input"] = cmd, kw.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout='{"tps": 12.5}\n', stderr="")

    monkeypatch.setattr(bench_decode.subprocess, "run", fake_run)
    spec = {"model": "m", "kwargs": {}, "prompt": "x" * 200_000, "tokens": 8, "samples": 1}
    assert bench_decode._measure_in_subprocess(spec, quiet=True) == {"tps": 12.5}
    assert all(len(a) < 1000 for a in seen["cmd"])
    assert json.loads(seen["input"]) == spec


def test_a_worker_that_cannot_start_is_reported(monkeypatch):
    def fake_run(cmd, **kw):
        raise OSError(7, "Argument list too long")

    monkeypatch.setattr(bench_decode.subprocess, "run", fake_run)
    out = bench_decode._measure_in_subprocess({"prompt": ""}, quiet=True)
    assert "error" in out
