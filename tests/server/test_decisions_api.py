"""Unit tests for native Structured Decisions (`/v1/decisions`) and System One (`/v1/systemone`, `/v1/decide`)."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.decisions_api import register_decisions_routes
from freetoken.tokenizer.tokenize import TokenizeManager


class _DecisionStubTokenizer:
    eos_token_id = 0

    def __len__(self) -> int:
        return 512

    def encode(self, text: str, return_tensors=None, add_special_tokens: bool = True):
        # Map single letters 'A'..'Z' and ' A'..' Z' to deterministic distinct token IDs!
        stripped = text.strip()
        if len(stripped) == 1 and "A" <= stripped <= "Z":
            offset = 100 if not text.startswith(" ") else 200
            tid = offset + (ord(stripped) - ord("A"))
            return [tid]
        return [10 + (ord(c) % 40) for c in text[:32]] or [10]

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)


class _BiasedLabelBackend:
    """Simulates a model that strongly prefers whichever letter slot holds 'Paris' or 'Yes / True' or 'Score 4',
    with an artificial +1.0 slot-A bias so cyclic rotation + prior calibration can be verified."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            model_path="/models/qwen3.8",
            served_model_name="qwen3.8",
            maintenance_state="serving",
        )
        self._tok = _DecisionStubTokenizer()
        self._mgr = TokenizeManager(self._tok)
        self._uid = 0
        self.sent: list[TokenizeMsg] = []
        self._by_uid: dict[int, TokenizeMsg] = {}

    def frontend_tokenizer(self):
        return self._mgr

    def new_user(self) -> int:
        self._uid += 1
        return self._uid

    async def send_one(self, msg: TokenizeMsg):
        self.sent.append(msg)
        self._by_uid[msg.uid] = msg

    async def abort_user(self, uid: int):
        pass

    async def wait_for_ack(self, uid: int):
        msg = self._by_uid[uid]
        rendered = "\n".join(
            m.get("content", "") for m in msg.text if isinstance(m, dict)
        )
        # Base logprobs per letter A..Z: slot A gets +1.0 prior bias; matching semantic option gets +4.0!
        letter_scores: dict[str, float] = {}
        for idx, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            score = 1.0 if ch == "A" else 0.0
            marker = f"{ch}) "
            pos = rendered.find(marker)
            if pos >= 0:
                line = rendered[pos : rendered.find("\n", pos)]
                if any(w in line for w in ("Paris", "Yes / True", "Score 4 ")):
                    score += 4.0
            letter_scores[ch] = score

        top = []
        best_tid = 100
        best_lp = -99.0
        for tid in msg.sampling_params.logprob_token_ids or []:
            if 100 <= tid < 126:
                ch = chr(ord("A") + (tid - 100))
                lp = letter_scores[ch] - 5.0
            elif 200 <= tid < 226:
                ch = chr(ord("A") + (tid - 200))
                lp = letter_scores[ch] - 8.0
            else:
                lp = -20.0
            top.append({"token_id": tid, "token": "x", "bytes": [120], "logprob": lp})
            if lp > best_lp:
                best_lp = lp
                best_tid = tid

        yield UserReply(
            uid=uid, incremental_output="", finished=False, prompt_tokens_delta=12
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


def _make_client() -> tuple[TestClient, _BiasedLabelBackend]:
    state = _BiasedLabelBackend()
    app = FastAPI()
    register_decisions_routes(app, lambda: state)
    return TestClient(app), state


def test_openai_decisions_predicate_choice_and_score():
    client, state = _make_client()
    r = client.post(
        "/v1/decisions",
        json={
            "model": "qwen3.8",
            "input": "France is in Western Europe and its capital is Paris.",
            "questions": [
                {
                    "id": "q_pred",
                    "type": "predicate",
                    "question": "Is Paris the capital?",
                },
                {
                    "id": "q_choice",
                    "type": "choice",
                    "question": "Which city is the capital?",
                    "options": ["Berlin", "Madrid", "Paris"],
                },
                {
                    "id": "q_score",
                    "type": "score",
                    "question": "Rate confidence 1..5",
                    "min": 1,
                    "max": 5,
                },
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "decision"
    assert len(body["answers"]) == 3

    pred, choice, score = body["answers"]
    assert pred["id"] == "q_pred" and pred["expected"] > 0.9
    assert choice["id"] == "q_choice" and choice["choice"] == "Paris"
    assert choice["distribution"]["Paris"] > 0.9
    assert score["id"] == "q_score" and 3.5 < score["expected"] < 4.5
    assert body["usage"]["prompt_tokens"] > 0
    # Verify all reads used logprob_token_ids (full-vocab reads, not restricted allowed_token_ids!)
    assert all(
        m.sampling_params.logprob_token_ids
        and m.sampling_params.allowed_token_ids is None
        for m in state.sent
    )


def test_systemone_and_decide_with_diagnostics_and_noul():
    client, _ = _make_client()
    for path in ("/v1/systemone", "/v1/decide"):
        r = client.post(
            path,
            json={
                "state": [{"role": "user", "content": "Capital city lookup: Paris."}],
                "diagnostics": True,
                "questions": {
                    "city": {
                        "type": "choice",
                        "instructions": "Pick the mentioned city",
                        "options": {"lon": "London", "par": "Paris"},
                    },
                    "fallback": {
                        "type": "noul",
                        "instructions": "Pick the country if explicitly stated",
                        "options": {"jp": "Japan", "br": "Brazil"},
                    },
                },
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answers"]["city"]["choice"] == "par"
        assert "none_of_the_above" in body["answers"]["fallback"]["distribution"]
        diag = body["diagnostics"]["city"]
        assert 0.0 < diag["label_mass"] <= 1.0
        assert diag["argmax_is_label"] is True
