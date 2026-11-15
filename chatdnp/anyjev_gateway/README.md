# AnyJev decision gateway

`gateway.py` serves two decision APIs, answered by AnyJev's `Decider` (level L0: answer-order averaging plus a label-free prior correction). The decisions read label logprobs from FreeToken.

| Route | Format |
|---|---|
| `POST /v1/decisions` | [OpenAI Decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create) (`client.decisions.create`): `input` + `questions[]` (predicate, choice, score) → `answers[]` in question order |
| `POST /v1/decide`, `POST /v1/systemone` | Jev: `state` + `questions{name: {type: choice \| score \| noul, instructions, criteria}}` → `answers{name: ...}` |
| `GET /v1/models`, `GET /health` | – |

LiteLLM 1.108.9 serves decisions natively (`/v1/decisions` in OpenAI's format, `/v1/systemone` in Jev's System One format) and routes them by model. One `model_list` entry with provider `openai` points at this gateway; LiteLLM converts either client format and calls the gateway's `/v1/decisions`:

client (own LiteLLM key, `model: anyjev-decide`) → Caddy (`/v1/*`) → LiteLLM → gateway `/v1/decisions` → FreeToken `127.0.0.1:8000` `/v1/completions`

FreeToken must run the AnyJev logprobs port (production since 2026-11-15, `688d49d8`). The gateway's own `/v1/decide` and `/v1/systemone` stay available for direct, local use.

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

The full LiteLLM change plan is `LITELLM_CHANGES_PLAN.md` in the chatdnp-audit notes. The steps for the gateway:

1. Add the key to `/opt/litellm/.env` (the value is in `~/anyjev-gw/key`):
   `ANYJEV_GW_KEY=<contents of ~/anyjev-gw/key>`
2. Add these entries **under the existing `model_list:`** in `/opt/litellm/config.yaml` (do not add a second `model_list:` or `general_settings:` key):

   ```yaml
     # under the existing model_list:, written out in full (YAML anchors would share one dict between deployments)
     - model_name: anyjev-decide
       litellm_params:
         model: openai/anyjev-qwen3.8-flash-next
         api_base: http://host.docker.internal:8100
         api_key: os.environ/ANYJEV_GW_KEY
         timeout: 300
         max_parallel_requests: 4
       model_info:
         mode: decisions
         input_cost_per_token: 0.0
         output_cost_per_token: 0.0
     - model_name: jev-latest          # the Jev SDKs' default model name
       litellm_params:
         model: openai/anyjev-qwen3.8-flash-next
         api_base: http://host.docker.internal:8100
         api_key: os.environ/ANYJEV_GW_KEY
         timeout: 300
         max_parallel_requests: 4
       model_info:
         mode: decisions
         input_cost_per_token: 0.0
         output_cost_per_token: 0.0
     - model_name: gpt-6-luna          # the OpenAI Decisions model name
       litellm_params:
         model: openai/anyjev-qwen3.8-flash-next
         api_base: http://host.docker.internal:8100
         api_key: os.environ/ANYJEV_GW_KEY
         timeout: 300
         max_parallel_requests: 4
       model_info:
         mode: decisions
         input_cost_per_token: 0.0
         output_cost_per_token: 0.0
   `````````

3. Start the gateway on docker0: `tmux kill-session -t =jevgw` (the localhost test copy), then `cp anyjev-gateway.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now anyjev-gateway`.
4. Restart LiteLLM when it is idle: `docker compose -f /opt/litellm/docker-compose.yml --project-directory /opt/litellm up -d litellm`.
5. Test with a user key:

   ```python
   from openai import OpenAI  # openai >= 3.51 has client.decisions
   c = OpenAI(api_key=KEY, base_url="https://<litellm>/v1")
   c.decisions.create(model="anyjev-decide", input="The package arrived with a broken screen.",
                      questions=[{"type": "predicate", "name": "damaged", "instructions": "Does the customer report a damaged item?"}])
   ```

   For Jev, `POST /v1/systemone` with `{"model": "anyjev-decide", "state": ..., "questions": {...}}`.

**Tested on 2026-11-15** on throwaway LiteLLM 1.108.9 runs with production's own `config.yaml` plus these entries, their own Postgres, and virtual keys from `/key/generate`:
- the SDK `decisions.create` returned 200 with `anyjev-decide` and `gpt-6-luna`, and Jev `/v1/systemone` with `anyjev-decide` and `jev-latest`;
- each call wrote a spend row (`adecisions` or `asystemone`, with its prompt tokens);
- with `max_parallel_requests: 1` on the key, two parallel calls gave 200 and 429;
- a wrong key gave 401.

A Jev request without `model` gives 400: clients must send one of the three names.

**Known limit:** LiteLLM's decision endpoints do not decode gzip request bodies (400), so clients must send plain JSON. The official Jev SDKs compress by default.

## Tests

`python -m pytest test_gateway.py -q`, in a venv with `anyjev` and `pytest`. The tests use AnyJev's `FakeBackend` and need no network.
