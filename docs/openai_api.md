# OpenAI-compatible API: endpoints and parameters

What `ft serve` accepts on its OpenAI-shaped routes, what it ignores, and what it refuses,
next to the same surface in vLLM, SGLang and llama.cpp's server. "honoured" means the value
changes the generation; "accepted" means the request is not rejected but the value has no
effect; "400" means the server refuses the request with a clear message.

## Endpoints

| route | FreeToken | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| `POST /v1/chat/completions` | yes | yes | yes | yes |
| `POST /v1/completions` | yes | yes | yes | yes |
| `GET /v1/models` | yes | yes | yes | yes |
| `POST /v1/responses` (+ get, cancel) | yes | yes | yes | yes |
| `POST /v1/messages` (Anthropic), `count_tokens` | yes | yes | no | yes |
| `POST /tokenize`, `POST /detokenize` (also under `/v1/`) | yes | yes | yes | yes (`/tokenize`, `/detokenize`) |
| `GET /metrics` (Prometheus) | yes | yes | yes | yes |
| `GET /version` | yes | yes | yes | no (`/props`) |
| `GET /health` | yes | yes | yes | yes |
| `GET /v1/stats`, `/v1/cache/*`, `/v1/requests`, `/v1/admin/*` | yes (FreeToken control plane) | `/metrics` only | `/get_server_info`, `/flush_cache` | `/slots`, `/props`, `/metrics` |
| `POST /v1/embeddings`, `/rerank`, `/score`, `/pooling` | no (generative models only) | embedding / reranker models | embedding / reranker models | yes, any model with pooling |
| `POST /v1/audio/*`, `/v1/files`, `/v1/batches` | no | transcriptions, batch runner (offline) | transcriptions, files, batches | no |
| `POST /infill`, `/apply-template`, `/completion` (non-OpenAI) | `suffix` on `/v1/completions`; `/tokenize` with `messages` renders the template | no | no | yes |

## Chat completions: request

| parameter | FreeToken | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| `model`, `messages` (text + `image_url` parts) | honoured | honoured | honoured | honoured |
| `max_tokens`, `max_completion_tokens` | honoured (default 32k) | honoured | honoured | `max_tokens` |
| `temperature`, `top_p`, `top_k` | honoured | honoured | honoured | honoured |
| `min_p` | honoured (prob. below `min_p` x top prob dropped) | honoured | honoured | honoured |
| `presence_penalty`, `frequency_penalty` | honoured, over generated tokens | honoured | honoured | honoured |
| `repetition_penalty` | honoured, over prompt + generated tokens (HF semantics) | honoured | honoured | `repeat_penalty` |
| `logit_bias` | honoured (token id -> bias, clamped to [-100, 100]) | honoured | honoured | honoured |
| `min_tokens` | honoured: no EOS / stop token before this many generated tokens | honoured | `min_new_tokens` | no |
| `stop` (strings), `stop_token_ids` | honoured | honoured | honoured | `stop` only |
| `include_stop_str_in_output` | honoured | honoured | `no_stop_trim` | no |
| `skip_special_tokens` | honoured; default off here, the reasoning and tool parsers read the tags | honoured (default on) | honoured (default on) | n/a |
| `n` | 400 unless 1 | honoured | honoured | `n_cmpl` |
| `stream`, `stream_options.include_usage`, `stream_options.continuous_usage_stats` | honoured | honoured | honoured | honoured |
| `timings_per_token` | honoured on streams when `--enable-timings-report` is set | no | no | honoured |
| `logprobs`, `top_logprobs` | accepted, ignored on this route (`/v1/completions` honours `logprobs`) | honoured | honoured | `n_probs` |
| `tools`, `tool_choice` (`auto` / `none` / `required` / named), `parallel_tool_calls` | honoured | honoured | honoured | honoured |
| `reasoning_effort`, `thinking`, `chat_template_kwargs` | honoured | `chat_template_kwargs` | honoured | honoured |
| `continue_final_message` | honoured (the last assistant message is continued, no generation prompt) | honoured | honoured | no |
| `response_format` `json_object` / `json_schema`, `guided_*`, `structured_outputs` | honoured via grammar-constrained decoding (`pip install 'freetoken[grammar]'`: xgrammar primary, llguidance fallback); validates strict JSON Schema (`additionalProperties: false`, all properties `required`, valid `$ref`) with HTTP 400 `invalid_json_schema`; preserves KV/GDN prefix cache; defers masking across `<think>...</think>` on reasoning models; other types 400 | honoured (xgrammar / guidance) | honoured (xgrammar / outlines / llguidance) | `grammar`, `json_schema` |
| `seed` | accepted, not honoured (one batched sampling kernel, no per-request generator) | honoured | `sampling_seed` | honoured |
| `user`, `metadata`, `store`, `service_tier` | accepted, ignored | accepted, ignored | accepted, ignored | accepted, ignored |
| `request_id` | honoured: becomes the response id | honoured | `rid` | `id_slot` (different meaning) |
| `function_call` (legacy) | 400 | 400 | 400 | no |
| `best_of`, `use_beam_search`, `length_penalty`, `prompt_logprobs`, `echo` (chat), `truncate_prompt_tokens`, `priority`, `cache_salt`, `lora` | no | honoured | partly | partly |
| `ignore_eos` | honoured (benchmarking) | honoured | honoured | honoured |

## Legacy completions: request

`prompt` as a string, a list of strings, a list of token ids or a list of token-id lists
(each id in `[0, len(tokenizer))`; not with `echo` or `suffix`; a stream takes one prompt only); the same sampling parameters as above; plus:

| parameter | FreeToken | vLLM | SGLang | llama.cpp |
|---|---|---|---|---|
| `echo` | honoured (the prompt leads the text; not with `logprobs`) | honoured | honoured | no |
| `suffix` | honoured on models with fill-in-the-middle tokens (`<|fim_prefix|>` ... `<|fim_middle|>`): the prompt becomes prefix-suffix-middle | 400 | 400 | `/infill` |
| `n` | 400 unless 1 | honoured | honoured | no |
| `logprobs` | honoured, 0..32: raw pre-temperature logprobs of the sampled token and its top alternatives; -inf is sent as -9999. With speculative decoding (`--spec-algo`) only with `max_tokens=1` (400 otherwise) | honoured (`--max-logprobs`, default 20) | honoured | `n_probs` |
| `allowed_token_ids` | honoured, 1..1024 ids below `len(tokenizer)`: sampling is masked to them at every position. With `logprobs` the values are renormalized over the set (vLLM's `processed_logprobs` at temperature 0; `temperature` defaults to `0` when omitted), so `temperature > 0`, `min_p > 0`, penalties, `min_tokens` and `logit_bias` are a 400 there; entries past the set are left out | honoured (raw logprobs unless `--logprobs-mode processed_logprobs`) | no | no |
| `return_tokens_as_token_ids` | honoured: `tokens` and the `top_logprobs` keys are `token_id:N`, so ids that decode to the same text stay apart | honoured | n/a | no |

Set values of the vLLM fields this server does not implement (`prompt_logprobs`,
`bad_words`) are a 400 rather than ignored; `null` / `false` / empty pass.
Empty prompts and more than 256 prompts are a 400. Ids in `allowed_token_ids`, `logit_bias`,
`stop_token_ids` and token-id prompts must be below `len(tokenizer)` (a 400 otherwise).

## Responses

| field | FreeToken | notes |
|---|---|---|
| `id` | `chatcmpl-<uid>` / `cmpl-<uuid>`, or `<prefix>-<request_id>` when the client sent one | |
| `created`, `model`, `object` | yes | |
| `system_fingerprint` | `null` | vLLM and SGLang also send `null` |
| `choices[].message.reasoning_content` | yes, when the model reasoned | vLLM `reasoning_content`, llama.cpp `reasoning_content` |
| `choices[].message.tool_calls` | yes | |
| `choices[].finish_reason` | `stop`, `length`, `tool_calls` | |
| `usage.prompt_tokens`, `completion_tokens`, `total_tokens` | yes | |
| `usage.prompt_tokens_details.cached_tokens` | with `--enable-cache-report` | vLLM, SGLang: always |
| `usage.completion_tokens_details.reasoning_tokens` | yes, when the model emitted a reasoning end tag | vLLM, SGLang |
| streaming: one `[DONE]`, an optional final usage chunk | yes | |
| llama.cpp `timings` | with `--enable-timings-report` (and per-chunk on streams with `timings_per_token: true`) | `/v1/stats` and `/metrics` carry the server-side rates |

## Tokenize / detokenize

```
POST /tokenize      {"prompt": "text", "add_special_tokens": true, "return_token_strs": false}
POST /tokenize      {"messages": [...], "add_generation_prompt": true, "chat_template_kwargs": {}}
                    -> {"count": N, "max_model_len": L, "tokens": [...], "token_strs": [...]?}
POST /detokenize    {"tokens": [...], "skip_special_tokens": false} -> {"prompt": "text"}
```

`messages` are rendered through the model's chat template exactly as a generation would
render them, so `count` equals the `prompt_tokens` that generation reports.

## Structured Outputs and Strict Tool Calling

Install the optional grammar extra to enable per-step vocabulary bitmasking:

```bash
pip install 'freetoken[grammar]'
```

Supported entry points (all routed through one engine-level `StructuredOutputSpec` without mutating `messages[0]`, keeping KV and GDN prefix caches intact):

- **OpenAI `/v1/chat/completions` & `/v1/completions`**: `response_format` (`text`, `json_object`, `json_schema` with optional `strict: true`), `tools[].function.strict: true`, forced `tool_choice` (`required` / named function), and vLLM/SGLang guided fields (`guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `structured_outputs`).
- **OpenAI `/v1/responses`**: `text.format` (`text`, `json_object`, `json_schema` with optional `strict: true`, echoed on `Response.text`) and `tools[].strict: true`.
- **Anthropic `/v1/messages`**: `output_config.format`, beta top-level `output_format` (`json_schema` with required `schema`), `tools[].strict: true`, and forced `tool_choice` (`any` / `tool`).
- **Strict JSON Schema subset (`strict: true`)**: root must be an object, every object must set `"additionalProperties": false`, every key in `properties` must appear in `required`, and `$ref` must resolve locally (`#/$defs/...` or `#/definitions/...`). Invalid strict schemas return HTTP 400 with `code: "invalid_json_schema"`.
- **Reasoning & Speculative Decoding**: On forced-reasoning models (`qwen3`, `deepseekv32`, `glm`, `gemma4`, `minimax_m3` enabled), logits stay unconstrained until the closing reasoning tag (`</think>`, `</mm:think>`, `<channel|>`) commits; on optional-think models (`flint`, `ricursive`, `minimax_m3` adaptive), an optional token-bounded think tag wraps the grammar. Under overlap scheduling, constrained batches drain before preparing the next bitmask (zero overhead on unconstrained traffic). Under speculative decoding (MTP & DFlash2), verify batches build exact per-slot chain bitmasks and roll back rejected drafts cleanly, and DFlash2 filters its top-K draft candidates against the active grammar.

## Not implemented, on purpose

- Prompt logprobs and `echo` with `logprobs`: need the prefill logits.
- Per-request `seed`: the batched sampling kernel draws from one generator.
- Embeddings, reranking, scoring, audio: not generation.
