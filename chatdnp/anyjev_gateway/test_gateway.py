"""Unit tests for the decision gateway with AnyJev's FakeBackend (no network, no GPU).

Run: python -m pytest chatdnp/anyjev_gateway/test_gateway.py -q   (needs anyjev and pytest)
"""

from __future__ import annotations

import gzip
import json
import threading
import urllib.error
import urllib.request

import gateway as gw
import pytest
from anyjev.backends.fake import FakeBackend


def _content(state: str, option: str) -> float:
    # billing / Yes / "urgent" win clearly, so the expected answers are known
    return 4.0 if option.split(":")[0].strip() in ("billing", "Yes", "urgent", "True") else 0.0


@pytest.fixture()
def server():
    def start(key=None, backend=None):
        engine = gw.Engine(backend or FakeBackend(_content), "test-model", 0.42, level="L0", prior="content_free")
        srv = gw.make_server(engine, "127.0.0.1", 0, key)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    started = []
    yield start
    for srv in started:
        srv.shutdown()


def post(base, path, body, headers=None, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


OPENAI = {
    "model": "gpt-6-luna",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "I was charged twice for my order."}]}],
    "questions": [
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which team?",
            "choices": [{"value": "billing", "description": "charges"}, {"value": "technical"}, {"value": "shipping"}],
        },
        {"type": "predicate", "name": "urgent", "instructions": "Is this urgent?"},
        {
            "type": "score",
            "name": "severity",
            "instructions": "How severe?",
            "levels": [{"label": "minor"}, {"label": "urgent"}, {"label": "blocked"}],
        },
        {"type": "choice", "name": "refund", "instructions": "Refund?", "choices": [{"value": True}, {"value": False}]},
    ],
}


def test_openai_answers_keep_question_order_types_and_values(server):
    # red if answers lose the question order, drop names, or turn a boolean choice value into a string
    s, d, h = post(server(), "/v1/decisions", OPENAI)
    assert s == 200, d
    assert [a["type"] for a in d["answers"]] == ["choice", "predicate", "score", "choice"]
    assert [a["name"] for a in d["answers"]] == ["department", "urgent", "severity", "refund"]
    dep, urg, sev, ref = d["answers"]
    assert dep["choice"] == "billing" and [p["value"] for p in dep["probabilities"]] == [
        "billing",
        "technical",
        "shipping",
    ]
    assert abs(sum(p["probability"] for p in dep["probabilities"]) - 1) < 1e-6
    assert 0.5 < urg["probability"] <= 1
    assert [(p["value"], p["label"]) for p in sev["probabilities"]] == [(0, "minor"), (1, "urgent"), (2, "blocked")]
    assert 0.5 < sev["score"] < 1.5  # the weighted level index, near "urgent"
    assert ref["choice"] is True and ref["probabilities"][0]["value"] is True
    assert d["model"] == "test-model" and d["usage"]["total_tokens"] == d["usage"]["input_tokens"]
    assert h.get("X-AnyJev-Levels") == "L0,L0,L0,L0"


def test_jev_answers_are_keyed_by_name(server):
    # red if the Jev path answers in OpenAI's array shape, or keys score levels by label instead of index
    req = {
        "model": "jev-latest",
        "state": "Customer: I was charged twice.",
        "questions": {
            "route": {"type": "choice", "instructions": "Where?", "criteria": {"billing": "money", "bug": "broken"}},
            "escalate": {"type": "noul", "instructions": "Escalate?"},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["routine", "urgent", "critical"]},
        },
    }
    base = server()
    for path in ("/v1/decide", "/v1/systemone"):
        s, d, _ = post(base, path, req)
        assert s == 200, d
        a = d["answers"]
        assert a["route"]["choice"] == "billing" and set(a["route"]["probabilities"]) == {"billing", "bug"}
        assert a["escalate"]["type"] == "noul" and 0.5 < a["escalate"]["noul"] <= 1
        assert set(a["urgency"]["probabilities"]) == {"0", "1", "2"} and a["urgency"]["legend"] == [
            "routine",
            "urgent",
            "critical",
        ]
        assert d["usage"]["cost_usd"] == pytest.approx(d["usage"]["input_tokens"] * 0.42 / 1e6)


@pytest.mark.parametrize(
    "change, status, param",
    [
        ({"questions": []}, 400, "questions"),
        ({"stream": True}, 400, "stream"),
        (
            {
                "questions": [
                    {"type": "choice", "instructions": "x", "choices": [{"value": f"o{i}"} for i in range(27)]}
                ]
            },
            400,
            "questions[0].choices",
        ),
        (
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}
                ]
            },
            400,
            "input[0].content[0]",
        ),
        ({"questions": [{"type": "box", "instructions": "x"}]}, 400, "questions[0].type"),
    ],
)
def test_openai_refusals_name_the_field(server, change, status, param):
    # red if an unsupported request reaches the model (or crashes) instead of a 400 that names the field
    s, d, _ = post(server(), "/v1/decisions", {**OPENAI, **change})
    assert s == status and d["error"]["param"] == param and d["error"]["type"] == "invalid_request_error"


def test_gateway_key(server):
    # red if a configured secret is not enforced, or LiteLLM's header forms are not accepted
    base = server(key="s3cret")
    assert post(base, "/v1/decisions", OPENAI)[0] == 401
    assert post(base, "/v1/decisions", OPENAI, {"X-Gateway-Key": "s3cret"})[0] == 200
    assert (
        post(
            base,
            "/v1/decide",
            {"state": "x", "questions": {"q": {"type": "noul", "instructions": "y?"}}},
            {"Authorization": "Bearer s3cret"},
        )[0]
        == 200
    )
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        assert r.status == 200


def test_backend_failure_is_a_502_and_ids_are_unique(server):
    # red if a FreeToken outage crashes the handler, or two responses share an id (LiteLLM drops repeated ids)
    class Down(FakeBackend):
        def next_token_logprobs(self, prompts, token_ids):
            raise RuntimeError("connection refused")

    s, d, _ = post(server(backend=Down(_content)), "/v1/decisions", OPENAI)
    assert s == 502 and d["error"]["type"] == "upstream_error"
    base = server()
    ids = {post(base, "/v1/decisions", OPENAI)[1]["id"] for _ in range(3)}
    assert len(ids) == 3


def test_gzip_body(server):
    # red if a gzip-compressed body (the Jev SDKs compress by default) is refused or misread
    body = gzip.compress(
        json.dumps({"state": "x", "questions": {"q": {"type": "noul", "instructions": "y?"}}}).encode()
    )
    s, d, _ = post(server(), "/v1/decide", None, {"Content-Encoding": "gzip"}, raw=body)
    assert s == 200 and d["answers"]["q"]["type"] == "noul"
