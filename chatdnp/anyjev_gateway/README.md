# AnyJev decision gateway

`gateway.py` serves two decision APIs, answered by AnyJev's `Decider` (level L0: answer-order averaging plus a label-free prior correction). The decisions read label logprobs from FreeToken.

| Route | Format |
|---|---|
| `POST /v1/decisions` | [OpenAI Decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create) (`client.decisions.create`): `input` + `questions[]` (predicate, choice, score) → `answers[]` in question order |
| `POST /v1/decide`, `POST /v1/systemone` | Jev: `state` + `questions{name: {type: choice \| score \| noul, instructions, criteria}}` → `answers{name: ...}` |
| `GET /v1/models`, `GET /health` | – |

LiteLLM fronts the gateway with authenticated pass-through routes:

client (own LiteLLM key) → LiteLLM → gateway → FreeToken `127.0.0.1:8000` `/v1/completions`

FreeToken must run the AnyJev logprobs port (production since 2026-11-15, `688d49d8`).

## Limits and behaviour

- **Choices:** at most 26 per question (AnyJev reads one letter per option). More is a 400 that names the field; OpenAI allows 255.
- **Score:** 2–10 levels. `score` is the probability-weighted level index, as in OpenAI's definition.
- **Not supported:** images (`input_image`, Jev `assets`) and streaming, which give a 400.
- **Load:** decisions run one at a time, and each one sends at most `--workers` (4) reads at once.
  - A new question first computes a content-free prior: about 24 reads, 3–4 s for 3 questions.
  - A repeated question takes about 6 reads, under 1 s.
- **Usage:** `input_tokens` counts the prompt tokens sent to FreeToken. `cost_usd` (Jev) uses `--price-per-mtok` (default 0).
- **Errors:** response ids are unique (`dec_<uuid>`). The header `X-AnyJev-Levels` reports each answer's level.
- **Shared secret:** the key in `~/anyjev-gw/key` (mode 600) is required on every route except `/health`, as `X-Gateway-Key: <key>` or `Authorization: Bearer <key>`.

## Files on chatdnp

- `~/anyjev-gw/.venv`: uv venv with anyjev 0.3.0, transformers 5.18 (tokenizer only), jinja2 and numpy.
- `~/anyjev-gw/run_gateway.sh`: the start script (a copy is here). It reads the key and logs to `~/logs/anyjev_gateway.log`.
- `anyjev-gateway.service`: a systemd user unit that binds the docker0 address (172.17.0.1). Not enabled yet.

## Production steps (root-owned LiteLLM files, done by the operator)

1. Add the key to `/opt/litellm/.env` (the value is in `~/anyjev-gw/key`):
   `ANYJEV_GW_KEY=<contents of ~/anyjev-gw/key>`
2. Add the routes to `general_settings` in `/opt/litellm/config.yaml`:

   ```yaml
   general_settings:
     pass_through_endpoints:
       - path: "/v1/decisions"
         target: "http://host.docker.internal:8100/v1/decisions"
         auth: true
         methods: ["POST"]
         headers:
           X-Gateway-Key: "os.environ/ANYJEV_GW_KEY"
       - path: "/v1/decide"
         target: "http://host.docker.internal:8100/v1/decide"
         auth: true
         methods: ["POST"]
         headers:
           X-Gateway-Key: "os.environ/ANYJEV_GW_KEY"
       - path: "/v1/systemone"
         target: "http://host.docker.internal:8100/v1/systemone"
         auth: true
         methods: ["POST"]
         headers:
           X-Gateway-Key: "os.environ/ANYJEV_GW_KEY"
   ```

3. Start the gateway on docker0: `cp anyjev-gateway.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now anyjev-gateway`. Stop the localhost test copy first: `tmux kill-session -t =jevgw`.
4. Restart LiteLLM when it is idle (`docker compose -f /opt/litellm/docker-compose.yml --project-directory /opt/litellm up -d litellm`).
5. Test with a user key: `curl -s https://<litellm>/v1/decisions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{"model":"anyjev","input":"The package arrived with a broken screen.","questions":[{"type":"predicate","name":"damaged","instructions":"Does the customer report a damaged item?"}]}'`

**Tested on 2026-11-15:** a throwaway LiteLLM 1.108.9 with exactly these blocks. Without a key it returned 401. With a key, all three routes returned the gateway's answers, and the injected `X-Gateway-Key` passed the gateway's check.

## Tests

`python -m pytest test_gateway.py -q`, in a venv with `anyjev` and `pytest`. The tests use AnyJev's `FakeBackend` and need no network.
