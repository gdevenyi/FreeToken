"""Unit tests for native Structured Decisions (`/v1/decisions`) and Jev System One (`/v1/systemone`, `/v1/decide`).

Covers full wire-schema parity with OpenAI Decisions (`client.decisions.create`), LiteLLM 1.108.9
(`mode: decisions`), Jev SDKs (`gzip` compression, `noul`/`choice`/`score` + `legend`), and AnyJev L0
content-free prior caching + permutation marginalization + `logprob_token_ids` diagnostics.
"""

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import freetoken.server.decisions_api as decisions_mod
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.api_server import install_api_key
from freetoken.server.decisions_api import register_decisions_routes
from freetoken.tokenizer.tokenize import TokenizeManager


class _DecisionStubTokenizer:
    eos_token_id = 0

    def __len__(self) -> int:
        return 512

    def encode(self, text: str, return_tensors=None, add_special_tokens: bool = True):
        stripped = text.strip()
        # Single-token IDs for Yes, No, digits 1..9, and letters A..Z (plus leading-space variants)
        lead = 100 if text.startswith(" ") else 0
        if stripped == "Yes":
            return [300 + lead]
        if stripped == "No":
            return [301 + lead]
        if len(stripped) == 1 and "1" <= stripped <= "9":
            return [310 + (ord(stripped) - ord("1")) + lead]
        if len(stripped) == 1 and "A" <= stripped <= "Z":
            return [100 + (ord(stripped) - ord("A")) + lead]
        return [10 + (ord(c) % 40) for c in text[:32]] or [10]

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)


def _is_content_free_probe(rendered: str) -> bool:
    for probe in ("State:\nN/A\n", "State:\n(empty)\n", "State:\n[MASK]\n"):
        if probe in rendered:
            return True
    return False


class _BiasedAnyJevBackend:
    """Simulates a biased LLM where position 0 (A / 1 / first word) has a +1.2 prior bias,
    while real states strongly boost 'billing', 'Paris', 'Yes', 'True', and 'urgent'."""

    def __init__(self, *, fail: bool = False) -> None:
        self.config = SimpleNamespace(
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            model_path="/models/Qwen3.8-Flash-Next",
            served_model_name="Qwen3.8-Flash-Next",
            maintenance_state="serving",
        )
        self._tok = _DecisionStubTokenizer()
        self._mgr = TokenizeManager(self._tok)
        self._uid = 0
        self.fail = fail
        self.sent: list[TokenizeMsg] = []
        self._by_uid: dict[int, TokenizeMsg] = {}

    def frontend_tokenizer(self):
        return self._mgr

    def new_user(self) -> int:
        self._uid += 1
        return self._uid

    async def send_one(self, msg: TokenizeMsg):
        if self.fail:
            raise RuntimeError("connection refused")
        self.sent.append(msg)
        self._by_uid[msg.uid] = msg

    async def abort_user(self, uid: int):
        pass

    async def wait_for_ack(self, uid: int):
        msg = self._by_uid[uid]
        rendered = "\n".join(
            m.get("content", "") for m in msg.text if isinstance(m, dict)
        )
        is_cf = _is_content_free_probe(rendered)

        scores_by_tid: dict[int, float] = {}
        # Binary noul ("Answer Yes or No." / "Answer No or Yes.")
        if "Answer Yes or No." in rendered or "Answer No or Yes." in rendered:
            first_yes = "Answer Yes or No." in rendered
            scores_by_tid[300] = (1.2 if first_yes else 0.0) + (0.0 if is_cf else 4.0)
            scores_by_tid[301] = 0.0 if first_yes else 1.2

        # Lettered options ("A. ...") and numbered score levels ("1. ...")
        for idx, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            tid = 100 + idx
            sc = 1.2 if idx == 0 else 0.0
            marker = f"\n{ch}. "
            pos = rendered.find(marker)
            if pos >= 0 and not is_cf:
                end = rendered.find("\n", pos + 1)
                line = rendered[pos + 1 : end if end > 0 else len(rendered)]
                opt_head = line.split(". ", 1)[1].split(":", 1)[0].strip()
                if opt_head in ("billing", "Paris", "par", "True", "urgent"):
                    sc += 4.5
            scores_by_tid[tid] = sc

        for idx in range(9):
            tid = 310 + idx
            sc = 1.2 if idx == 0 else 0.0
            marker = f"\n{idx + 1}. "
            pos = rendered.find(marker)
            if pos >= 0 and not is_cf:
                end = rendered.find("\n", pos + 1)
                line = rendered[pos + 1 : end if end > 0 else len(rendered)]
                opt_text = line.split(". ", 1)[1].split(":", 1)[0].strip()
                if opt_text in ("urgent", "4"):
                    sc += 4.5
            scores_by_tid[tid] = sc

        top = []
        best_tid = 100
        best_lp = -99.0
        for tid in msg.sampling_params.logprob_token_ids or []:
            lp = scores_by_tid.get(tid, -15.0) - 5.0
            top.append({"token_id": tid, "token": "x", "bytes": [120], "logprob": lp})
            if lp > best_lp:
                best_lp = lp
                best_tid = tid

        yield UserReply(
            uid=uid,
            incremental_output="",
            finished=False,
            prompt_tokens_delta=12,
            cached_tokens=4,
        )
        yield UserReply(
            uid=uid,
            incremental_output="A",
            finished=True,
            completion_tokens_delta=1,
            finish_reason="length",
            logprobs={
                "token_id": best_tid,
                "token": "A",
                "bytes": [65],
                "logprob": best_lp,
                "top": top,
            },
        )


def _make_client(*, api_key: str | None = None, fail: bool = False):
    decisions_mod._CF_PRIOR_CACHE.clear()
    state = _BiasedAnyJevBackend(fail=fail)
    app = FastAPI()
    register_decisions_routes(app, lambda: state)
    if api_key:
        install_api_key(app, api_key)
    return TestClient(app), state


OPENAI_PAYLOAD = {
    "model": "gpt-6-luna",
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "I was charged twice for my order."}
            ],
        }
    ],
    "questions": [
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which team?",
            "choices": [
                {"value": "billing", "description": "charges"},
                {"value": "technical"},
                {"value": "shipping"},
            ],
        },
        {"type": "predicate", "name": "urgent", "instructions": "Is this urgent?"},
        {
            "type": "score",
            "name": "severity",
            "instructions": "How severe?",
            "levels": [{"label": "minor"}, {"label": "urgent"}, {"label": "blocked"}],
        },
        {
            "type": "choice",
            "name": "refund",
            "instructions": "Refund?",
            "choices": [{"value": True}, {"value": False}],
        },
    ],
}


def test_openai_answers_keep_question_order_types_and_values():
    client, state = _make_client()
    r = client.post("/v1/decisions", json=OPENAI_PAYLOAD)
    assert r.status_code == 200, r.text
    d = r.json()
    assert [a["type"] for a in d["answers"]] == [
        "choice",
        "predicate",
        "score",
        "choice",
    ]
    assert [a["name"] for a in d["answers"]] == [
        "department",
        "urgent",
        "severity",
        "refund",
    ]
    dep, urg, sev, ref = d["answers"]
    assert dep["choice"] == "billing"
    assert [p["value"] for p in dep["probabilities"]] == [
        "billing",
        "technical",
        "shipping",
    ]
    assert abs(sum(p["probability"] for p in dep["probabilities"]) - 1.0) < 1e-6
    assert 0.5 < urg["probability"] <= 1.0
    assert [(p["value"], p["label"]) for p in sev["probabilities"]] == [
        (0, "minor"),
        (1, "urgent"),
        (2, "blocked"),
    ]
    assert 0.5 < sev["score"] < 1.5
    assert ref["choice"] is True and ref["probabilities"][0]["value"] is True
    assert d["model"] == "gpt-6-luna"
    assert d["usage"]["total_tokens"] == d["usage"]["input_tokens"] > 0
    assert d["usage"]["output_tokens"] == 0
    assert d["usage"]["input_tokens_details"]["cached_tokens"] > 0
    assert r.headers.get("X-AnyJev-Levels") == "L0,L0,L0,L0"

    # Verify all reads used logprob_token_ids (full-vocab reads, not restricted allowed_token_ids!)
    first_call_reads = len(state.sent)
    assert all(
        m.sampling_params.logprob_token_ids
        and m.sampling_params.allowed_token_ids is None
        for m in state.sent
    )

    # Second call hits the in-memory content-free prior cache and issues strictly fewer reads!
    r2 = client.post("/v1/decisions", json=OPENAI_PAYLOAD)
    assert r2.status_code == 200
    second_call_reads = len(state.sent) - first_call_reads
    assert 0 < second_call_reads < first_call_reads


def test_jev_answers_are_keyed_by_name_with_diagnostics():
    client, _ = _make_client()
    req = {
        "model": "jev-latest",
        "state": "Customer: I was charged twice.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Where?",
                "criteria": {"billing": "money", "bug": "broken"},
            },
            "escalate": {"type": "noul", "instructions": "Escalate?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["routine", "urgent", "critical"],
            },
        },
    }
    for path in ("/v1/decide", "/v1/systemone"):
        r = client.post(path, json=req)
        assert r.status_code == 200, r.text
        d = r.json()
        a = d["answers"]
        assert a["route"]["choice"] == "billing" and set(
            a["route"]["probabilities"]
        ) == {
            "billing",
            "bug",
        }
        assert a["escalate"]["type"] == "noul" and 0.5 < a["escalate"]["noul"] <= 1.0
        assert set(a["urgency"]["probabilities"]) == {"0", "1", "2"}
        assert a["urgency"]["legend"] == {
            "0": "routine",
            "1": "urgent",
            "2": "critical",
        }
        assert d["usage"]["input_tokens"] > 0
        assert d["usage"]["cost_usd"] == 0.0
        assert 0.0 < d["diagnostics"]["route"]["label_mass"] <= 1.0
        assert d["diagnostics"]["route"]["argmax_is_label"] is True


@pytest.mark.parametrize(
    "change, status, param",
    [
        ({"questions": []}, 400, "questions"),
        ({"stream": True}, 400, "stream"),
        (
            {
                "questions": [
                    {
                        "type": "choice",
                        "instructions": "x",
                        "choices": [{"value": f"o{i}"} for i in range(27)],
                    }
                ]
            },
            400,
            "questions[0].choices",
        ),
        (
            {
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,AA",
                            }
                        ],
                    }
                ]
            },
            400,
            "input[0].content[0]",
        ),
        (
            {"questions": [{"type": "box", "instructions": "x"}]},
            400,
            "questions[0].type",
        ),
    ],
)
def test_openai_refusals_name_the_field(change, status, param):
    client, _ = _make_client()
    r = client.post("/v1/decisions", json={**OPENAI_PAYLOAD, **change})
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["param"] == param
    assert err["type"] == "invalid_request_error"


def test_gzip_body_and_unique_ids_and_backend_502(monkeypatch):
    client, _ = _make_client()
    raw_gz = gzip.compress(
        json.dumps(
            {
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "y?"}},
            }
        ).encode()
    )
    r = client.post(
        "/v1/decide",
        content=raw_gz,
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["q"]["type"] == "noul"

    ids = {
        client.post("/v1/decisions", json=OPENAI_PAYLOAD).json()["id"] for _ in range(3)
    }
    assert len(ids) == 3

    down_client, _ = _make_client(fail=True)
    r_down = down_client.post("/v1/decisions", json=OPENAI_PAYLOAD)
    assert r_down.status_code == 502
    assert r_down.json()["error"]["type"] == "upstream_error"

    monkeypatch.setattr(decisions_mod, "MAX_QUESTIONS", 2)
    r_cap = client.post("/v1/decisions", json=OPENAI_PAYLOAD)
    assert r_cap.status_code == 400
    assert r_cap.json()["error"]["param"] == "questions"


def test_api_key_enforcement_on_decisions():
    client, _ = _make_client(api_key="s3cret")
    assert client.post("/v1/decisions", json=OPENAI_PAYLOAD).status_code == 401
    assert (
        client.post(
            "/v1/decisions",
            json=OPENAI_PAYLOAD,
            headers={"Authorization": "Bearer s3cret"},
        ).status_code
        == 200
    )
