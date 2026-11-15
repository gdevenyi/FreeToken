#!/usr/bin/env bash
# Start the AnyJev decision gateway (code: ~/ft-utils/chatdnp/anyjev_gateway/gateway.py, venv: ~/anyjev-gw/.venv).
# HOST=127.0.0.1 for local tests; HOST=<docker0 address> so the LiteLLM container can reach it (the key is then required).
set -eu
export ANYJEV_GW_KEY="$(cat "$HOME/anyjev-gw/key")" OMP_NUM_THREADS=1
TOK=/scratch/localadmin/hf-cache/hub/models--RedHatAI--Qwen3.8-Flash-Next-NVFP4/snapshots/949136021b05d3e1844f58cc4e01bd646b36d1a9
exec nice -n 5 "$HOME/anyjev-gw/.venv/bin/python" "$HOME/ft-utils/chatdnp/anyjev_gateway/gateway.py" \
  --tokenizer "$TOK" --upstream http://127.0.0.1:8000 --host "${HOST:-127.0.0.1}" --port "${PORT:-8100}" --workers 4 \
  >> "$HOME/logs/anyjev_gateway.log" 2>&1
