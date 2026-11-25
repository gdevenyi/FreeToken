"""/v1/internal/kld against a running server (GPU window only).

Start the server in its production form plus ``--enable-kld-endpoint``, then run this on the
server host (the base file must be readable here and by the server at the same path).

Env knobs:
  FREETOKEN_KLD_URL      server URL, e.g. http://127.0.0.1:1919 (required)
  FREETOKEN_KLD_BASE     llama.cpp --kl-divergence-base file (required)
  FREETOKEN_KLD_OFF_URL  optional: a server started WITHOUT the flag; its endpoint must be 404
  FREETOKEN_API_KEY      optional bearer token
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch
from freetoken.benchmark import kld

pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU window only"),
    pytest.mark.skipif(
        not (
            os.environ.get("FREETOKEN_KLD_URL") and os.environ.get("FREETOKEN_KLD_BASE")
        ),
        reason="set FREETOKEN_KLD_URL and FREETOKEN_KLD_BASE",
    ),
]

URL = os.environ.get("FREETOKEN_KLD_URL", "").rstrip("/")
BASE = os.environ.get("FREETOKEN_KLD_BASE", "")


def _post(url: str, body: dict, timeout: float = 900.0) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("FREETOKEN_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _score(chunks: list[int]) -> dict:
    status, out = _post(
        f"{URL}/v1/internal/kld",
        {"base_path": BASE, "chunks": chunks, "slice_rows": 16},
    )
    assert status == 200, out
    return out


def test_rows_line_up_with_the_base_file_and_repeat_bit_exact():
    base = kld.KldBase.open(BASE)
    first = _score([0, 1])
    for c in first["chunks"]:
        assert c["error"] is None and len(c["kld"]) == base.n_rows
    # the base NLL the server read for the target token equals the file's own: parser and targets agree
    np.testing.assert_allclose(
        first["chunks"][0]["nll_base"], kld.base_only_nll(base, [0]), rtol=0, atol=1e-5
    )
    # a cold isolated prefill (GPU experts only) repeats exactly
    again = _score([0])["chunks"][0]
    for name in kld.ROW_FIELDS:
        assert again[name] == first["chunks"][0][name], name


def test_model_is_near_its_bf16_reference():
    s = _score(list(range(4)))["summary"]
    # a misaligned row or a wrong vocab map gives a KLD of several nats and a top-1 near 0
    assert s["failed_chunks"] == [] and s["mean_kld"] < 0.2 and s["same_top"] > 0.8, s


def test_kld_chunks_leave_the_prefix_cache_alone():
    body = {
        "prompt": "The capital of France is",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 5,
    }
    _post(f"{URL}/v1/completions", body)  # cold: may insert the prompt
    status, warm = _post(f"{URL}/v1/completions", body)
    assert status == 200, warm
    _score([2, 3])
    status, after = _post(f"{URL}/v1/completions", body)
    assert after["choices"][0]["logprobs"] == warm["choices"][0]["logprobs"]


@pytest.mark.skipif(
    not os.environ.get("FREETOKEN_KLD_OFF_URL"), reason="set FREETOKEN_KLD_OFF_URL"
)
def test_endpoint_is_absent_without_the_flag():
    status, _ = _post(
        os.environ["FREETOKEN_KLD_OFF_URL"].rstrip("/") + "/v1/internal/kld",
        {"base_path": BASE, "chunks": []},
    )
    assert status == 404
