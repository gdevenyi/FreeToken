# chatdnp tools

Benchmark and probe tools for the FreeToken server on chatdnp. The server has 2x RTX 6000 Ada, TP=2,
and serves `Qwen3.8-Flash-Next` on `http://127.0.0.1:8000`. In a GPU window, `window.sh` serves the
candidate on `CAND_PORT` (default 8100) and sets `FT_BASE`, the default base URL of every probe, so
port 8000 stays closed and production clients never reach a test server.

Clone on the server, and run the tools from `chatdnp/`:

    git clone -b utils --single-branch https://github.com/gdevenyi/FreeToken.git ~/ft-utils
    cd ~/ft-utils/chatdnp
    PY=~/ft-chatdnpdeploy/.venv/bin/python

All tools need `requests` or only the standard library. `vision_probe.py` and `image_hit_probe.py`
need Pillow 10.1 or later. `keycheck2.py` needs the venv of a FreeToken tree. The system python3 has
no Pillow. Probes exit with status 0 when they pass, except `bench_client.py`, `stall_probe.py` and
`revisit_probe.py`, which only measure.

The sizes in the table are for a GPU window. Use the small sizes below on a server with users.

## Tools

| Tool | What it does | Run | Expected last line |
|---|---|---|---|
| `window.sh` | **Stops production.** Serves a candidate tree, runs tests, starts production again. | See below | `WINDOWDONE <label>` |
| `bench_client.py` | Single-stream rate (256 tokens), aggregate rate (512 tokens x C), TTFT on 1000 tokens, geometry. The same `--seed` gives the same prompts. Raw samples go to `--out`. | `$PY bench_client.py --base-url http://127.0.0.1:8000 --repeats 3 --concurrent 8 --seed 1 --out b.json` | JSON: `out single_median_tps agg_tps residency kv_tokens ttft_1k_s ttft_1k_prompt_tokens mamba_slots` |
| `gsm8k_probe.py` | First N GSM8K test questions, greedy. Downloads the data once to `~/.cache/freetoken-tools/`. | `$PY gsm8k_probe.py --n 300 --concurrency 8 --max-tokens 1536 --label L --out g.json [--thinking]` | JSON: `label n accuracy correct truncated mean_tokens wall_s thinking temperature errors` |
| `flipcheck.py` | Questions where two GSM8K runs do not agree. | `$PY flipcheck.py a.json b.json` | JSON: `a b common only_in_one both_correct a_only b_only both_wrong` |
| `vision_probe.py` | Two synthetic PNGs (text and circle colour) and a text-only question. | `$PY vision_probe.py` | `VISION_PROBE PASS` |
| `image_hit_probe.py` | Prefix cache with images: cold, turn 2, other image, repeat, long turn. Needs `--enable-cache-report`. | `$PY image_hit_probe.py [--long-tokens 23000]` | `IMAGE_HIT_PROBE PASS` |
| `disconnect_probe.py` | Closes the socket during a streamed and a non-streaming request, then checks `/v1/stats` and the log. | `FT_LOG=~/logs/ft.log $PY disconnect_probe.py [--max-tokens 2000]` | `DISCONNECT_PROBE PASS` |
| `revisit_probe.py` | Latency and cached tokens of a returning agent turn (cached prefix + about 20 new tokens). | `$PY revisit_probe.py --sessions 16 --tokens 24000 --out r.json` | `REVISIT_PROBE {... turn2_latency_median_s ...}` |
| `stall_probe.py` | Largest gap in a live stream during a long prefill of another request. | `$PY stall_probe.py --tokens 200000 --out s.json` | `STALL_PROBE {... max_gap_during_prefill_s ...}` |
| `sampling_probe.py` | Sampling options that the server must accept or refuse. | `$PY sampling_probe.py` | `SAMPLING_PROBE PASS` |
| `client_paths_probe.py` | Anthropic stream with thinking, and a tool call with `$ref` in its schema. | `$PY client_paths_probe.py` | `CLIENT_PATHS_PROBE PASS` |
| `keycheck2.py` | CPU check: the real loader must emit the state dict that the engine builds. No GPU, no server. | `CUDA_VISIBLE_DEVICES= FREETOKEN_FP8_DENSE=1 FREETOKEN_FP8_LMHEAD=1 ~/<tree>/.venv/bin/python keycheck2.py <snapshot> <rank> 2` | `KEYCHECK2 PASS` |
| `proxy_scope.py` | Option scan of the LiteLLM proxy. It reads the URL and the key from the environment only. | `CHATDNP_BASE=<url> CHATDNP_API_KEY=<key> python3 proxy_scope.py [name-filter]` | `<n>/<m> passed` |

## Small runs for a server with users

    $PY bench_client.py --repeats 1 --concurrent 2 --ttft-repeats 3 --seed $RANDOM --out /var/tmp/b.json
    $PY gsm8k_probe.py --n 3 --concurrency 2 --max-tokens 1536 --label tiny --out /var/tmp/g.json
    $PY revisit_probe.py --sessions 1 --tokens 2000 --out /var/tmp/r.json
    $PY stall_probe.py --tokens 2000 --stream-tokens 1200 --out /var/tmp/s.json
    FT_LOG=~/logs/ft.log $PY disconnect_probe.py --max-tokens 1500

`vision_probe.py`, `sampling_probe.py` and `client_paths_probe.py` are short at their defaults.

## Notes on the results

- **Seeds.** The same `--seed` on a running server can hit the prefix cache: TTFT changed from 0.60 s
  to 0.13 s, and `bench_client.py` prints a warning. Use a new seed for each run. `window.sh` does this.
- **Rate.** `single_median_tps` is completion tokens / (last token time - first token time). It counts
  all 256 tokens over 255 intervals, so it is 0.4% above the true interval rate.
- **Other clients.** `disconnect_probe.py` compares server counters to the values from before the
  request. Traffic from other clients can cause a false FAIL.
- **Cache report.** The server omits `prompt_tokens_details` for 0 cached tokens, so a missing object
  has the value 0.
- **Replaced text.** A prompt that replaces the last text of an earlier prompt can get 0 cached tokens
  with a long shared head (seen at 25k tokens, with and without an image). `image_hit_probe.py` tests
  turns that extend the earlier prompt.

## window.sh

**It stops production.** Do not run it on a server with users unless you plan the outage.

    nohup setsid env LABEL=w1 TREE=~/ft-candidate CAND_FLAGS="--moe-strategy offload ..." \
      TESTS="smoke settle bench vision" ~/ft-utils/chatdnp/window.sh > /dev/null 2>&1 < /dev/null &

Do not start a window before the previous one has written `WINDOWDONE` to its log.

| Variable | Default | Meaning |
|---|---|---|
| `LABEL`, `TREE`, `CAND_FLAGS` | required | Name of the window, candidate checkout with `.venv`, flags for `ft serve` |
| `CAND_ENV` | empty | Environment of the candidate |
| `TESTS` | `smoke vision bench` | Tests, in order |
| `PYC` | `$TREE/.venv/bin/python` | Python for the tools |
| `OUT` | `~/logs/windows` | Window log and `--out` files |
| `MODEL` | NVFP4 snapshot | Checkpoint for the candidate only |
| `READY_MIN`, `SETTLE_S` | 20, 480 | Load limit in minutes, settle time in seconds |
| `GPUTESTS`, `GPUCMD` | empty | Run after the tests, with the candidate stopped |

Tests: `smoke`, `geometry`, `listen`, `settle`, `bench`, `vision` (also runs `image_hit_probe.py`),
`gsm8k`, `sampling`, `clientpaths`, `disconnect`, `stall`, `revisit`, `revisitheavy`, or the name of
an executable file in this directory.

Output: `$OUT/<label>_window.log` has all output of all tests. The candidate server log is
`~/logs/ft_<label>.log`. To follow a window: `tail -f ~/logs/windows/<label>_window.log`.

`restore()` runs `env -u MODEL -u TREE -u CAND_ENV -u CAND_FLAGS ~/run_qwen38.sh start`. It reports
the production tree from the `FT_HOME` of `run_qwen38.sh`. Other knobs of `run_qwen38.sh` that you
export for the window (for example `FT_HOME` or `NUM_TOKENS`) also go to production: do not export them.
