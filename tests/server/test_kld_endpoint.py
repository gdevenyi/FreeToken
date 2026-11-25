"""POST /v1/internal/kld: refused unless --enable-kld-endpoint; each chunk goes out as a max_tokens=1
token-id request carrying its KLD spec, and the terminal replies' rows come back per chunk."""

from __future__ import annotations

import asyncio
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from freetoken.message import UserReply
from freetoken.server import api_server
from freetoken.server.args import parse_args

N_CTX, VOCAB, N_CHUNK = 16, 40, 3


def _base_file(path):
    rng = np.random.default_rng(1)
    n_rows, nv = N_CTX - 1 - N_CTX // 2, 2 * ((VOCAB + 1) // 2) + 4
    tokens = rng.integers(0, VOCAB, size=(N_CHUNK, N_CTX)).astype("<i4")
    rows = np.zeros((N_CHUNK * n_rows, nv), "<u2")
    with open(path, "wb") as f:
        f.write(
            b"_logits_"
            + struct.pack("<Iii", N_CTX, VOCAB, N_CHUNK)
            + tokens.tobytes()
            + rows.tobytes()
        )
    return tokens


def _rows(chunk, n=N_CTX - 1 - N_CTX // 2):
    return {
        "chunk": chunk,
        "error": None,
        "kld": [0.01] * n,
        "nll": [2.0] * n,
        "nll_base": [1.9] * n,
        "same_top": [1] * n,
        "p_diff": [0.0] * n,
    }


class FakeState:
    def __init__(self, error_for=()):
        self.sent, self.error_for, self.uid = [], set(error_for), 0

    def new_user(self):
        self.uid += 1
        return self.uid

    async def send_one(self, msg):
        self.sent.append(msg)

    async def wait_for_ack(self, uid):
        chunk = self.sent[-1].sampling_params.kld["chunk"]
        yield UserReply(
            uid=uid, incremental_output="", finished=False, prompt_tokens_delta=N_CTX
        )
        if chunk in self.error_for:
            yield UserReply(
                uid=uid, incremental_output="", finished=True, error="kld: boom"
            )
        else:
            yield UserReply(
                uid=uid, incremental_output="", finished=True, kld=_rows(chunk)
            )


def test_flag_defaults_off_and_parses():
    common = [
        "--model",
        "/models/anon",
        "--dtype",
        "bfloat16",
        "--tool-call-parser",
        "llama3",
        "--reasoning-parser",
        "off",
        "--sampling-defaults",
        "none",
    ]
    assert parse_args(common)[0].kld_endpoint is False
    assert parse_args([*common, "--enable-kld-endpoint"])[0].kld_endpoint is True


@pytest.mark.parametrize("enabled", [False, True])
def test_endpoint_is_refused_unless_enabled(tmp_path, monkeypatch, enabled):
    _base_file(tmp_path / "b.kld")
    state = SimpleNamespace(
        config=SimpleNamespace(kld_endpoint=enabled), maintenance_state="serving"
    )
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    monkeypatch.setattr(
        api_server,
        "run_kld",
        lambda st, req: asyncio.sleep(0, result={"ran": req.base_path}),
    )
    r = TestClient(api_server.app).post(
        "/v1/internal/kld", json={"base_path": str(tmp_path / "b.kld"), "chunks": []}
    )
    if enabled:
        assert r.status_code == 200 and r.json() == {"ran": str(tmp_path / "b.kld")}
    else:
        assert r.status_code == 404 and "--enable-kld-endpoint" in r.json()["error"]


def test_run_kld_sends_each_chunk_as_a_token_id_read(tmp_path):
    tokens = _base_file(tmp_path / "b.kld")
    state = FakeState(error_for={2})
    req = api_server.KldRequest(
        base_path=str(tmp_path / "b.kld"), chunks=[0, 2], slice_rows=5
    )
    out = json.loads(asyncio.run(api_server.run_kld(state, req)).body)
    assert [m.input_ids for m in state.sent] == [tokens[0].tolist(), tokens[2].tolist()]
    for m, chunk in zip(state.sent, (0, 2)):
        sp = m.sampling_params
        assert (sp.max_tokens, sp.ignore_eos, sp.temperature, sp.logprobs) == (
            1,
            True,
            0.0,
            False,
        )
        assert sp.kld == {
            "path": str(tmp_path / "b.kld"),
            "chunk": chunk,
            "slice_rows": 5,
            "bos_token_id": None,
        }
    assert out["base"]["n_chunk"] == N_CHUNK
    assert [c["chunk"] for c in out["chunks"]] == [0, 2] and out["chunks"][1][
        "error"
    ] == "kld: boom"
    assert (
        out["summary"]["failed_chunks"] == [2]
        and out["summary"]["count"] == N_CTX - 1 - N_CTX // 2
    )


def test_run_kld_bos_and_bad_requests(tmp_path):
    tokens = _base_file(tmp_path / "b.kld")
    state = FakeState()
    asyncio.run(
        api_server.run_kld(
            state,
            api_server.KldRequest(
                base_path=str(tmp_path / "b.kld"), chunks=[1], bos_token_id=7
            ),
        )
    )
    assert state.sent[0].input_ids == [7] + tokens[1, 1:].tolist()
    r = asyncio.run(
        api_server.run_kld(
            state, api_server.KldRequest(base_path=str(tmp_path / "b.kld"), chunks=[3])
        )
    )
    assert r.status_code == 400 and "outside" in json.loads(r.body)["error"]
    r = asyncio.run(
        api_server.run_kld(
            state, api_server.KldRequest(base_path=str(tmp_path / "none.kld"))
        )
    )
    assert r.status_code == 400
    header = json.loads(
        asyncio.run(
            api_server.run_kld(
                state,
                api_server.KldRequest(base_path=str(tmp_path / "b.kld"), chunks=[]),
            )
        ).body
    )
    assert header["chunks"] == [] and header["base"]["n_ctx"] == N_CTX
