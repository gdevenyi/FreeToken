#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (nvidia NVFP4) at its native 262144-token context on one GTX 1080 Ti.
#
#   ./freetoken-qwen38-1080.sh start     start it in the background (returns when it is ready)
#   ./freetoken-qwen38-1080.sh run       run it in the foreground (what the systemd unit calls)
#   ./freetoken-qwen38-1080.sh stop      stop it cleanly (add --force to escalate)
#   ./freetoken-qwen38-1080.sh kill      SIGKILL every process of the service
#   ./freetoken-qwen38-1080.sh status    is it up, and what is it holding
#   ./freetoken-qwen38-1080.sh logs      follow the log
#   ./freetoken-qwen38-1080.sh test      send one request
#
# Override any setting from the environment, e.g.
#   PORT=8081 MEMORY_RATIO=0.78 ./freetoken-qwen38-1080.sh start

set -uo pipefail

FT_DIR="${FT_DIR:-$HOME/FreeToken}"
MODEL="${MODEL:-$HOME/models/nvidia/Qwen3.8-Flash-Next-NVFP4}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
# Budget: MXFP8 dense + lm_head + hyper-connections and the host-RAM embedding leave room for
# ~1150 GPU expert slots with prefill overlap at 0.80; above ~0.82 a 2048-token prefill chunk's
# activations no longer fit next to them.
MEMORY_RATIO="${MEMORY_RATIO:-0.80}"
MAX_PREFILL_LENGTH="${MAX_PREFILL_LENGTH:-2048}"
# 262144 logical KV tokens: 32768 in fp8 on the GPU, the rest in the pinned host mirror.
KV_GPU_TOKENS="${KV_GPU_TOKENS:-32768}"
KV_HOST_PAGES="${KV_HOST_PAGES:-3584}"
# Decode computes expert-cache misses on the 20 CPU cores and fetches at most this many per layer
# over PCIe: 1 measured 19.1 tok/s here (steady-state decode), 0 gave 18.0.
HYBRID_MAX_FETCH="${HYBRID_MAX_FETCH:-1}"
HC_QUANT="${HC_QUANT:-mxfp8}"
# Default budget for requests that set no max_tokens; the default thinking effort easily runs
# past the engine's 32k default on agentic tasks.
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-65536}"
# VISION=1 builds the image encoder (weights streamed from host RAM at run time).
VISION="${VISION:-0}"
LOG="${LOG:-$HOME/.cache/freetoken/qwen38-1080.log}"
PIDFILE="${PIDFILE:-$HOME/.cache/freetoken/qwen38-1080.pid}"
START_TIMEOUT="${START_TIMEOUT:-900}"

# GDN prefix snapshots evicted from the GPU land in host RAM and come back on a hit, which
# is what keeps an agent's long conversation from re-prefilling after side requests.
export FT_GDN_HOST_TIER="${FT_GDN_HOST_TIER:-1}"
export FT_GDN_HOST_SLOTS="${FT_GDN_HOST_SLOTS:-32}"
# An agent turn's short extension loads only its routed experts instead of streaming all 48
# layers (~6 s): 40 tokens start in 1.9 s, 128 in ~3.2 s; past ~256 streaming wins again.
export FREETOKEN_MOE_SMALL_PREFILL_TOKENS="${FREETOKEN_MOE_SMALL_PREFILL_TOKENS:-256}"
# /tmp is a RAM disk on this host: keep triton/torch scratch files off it.
export TMPDIR="${TMPDIR_OVERRIDE:-$HOME/.cache/freetoken/tmp}"
# nvcc 12.9 rejects the system gcc; the JIT kernels need gcc 14.
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++-14}"

FT="$FT_DIR/.venv/bin/ft"

die() { echo "error: $*" >&2; exit 1; }

running_pid() {
    [ -f "$PIDFILE" ] || return 1
    local pid; pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    echo "$pid"
}

# Every process of this service. The scheduler/tokenizer workers are multiprocessing.spawn
# children with a generic command line and outlive a killed frontend, so find them by the
# venv interpreter they run from and by the listening port as well.
service_pids() {
    {
        pgrep -u "$(id -u)" -f "ft serve --model $MODEL" 2>/dev/null
        pgrep -u "$(id -u)" -f "^$FT_DIR/.venv/bin/python3 -c from multiprocessing" 2>/dev/null
        ss -ltnp 2>/dev/null | grep -E ":$PORT " | grep -oE 'pid=[0-9]+' | cut -d= -f2
    } | sort -un
}

serve_args() {
    local args=(
        --model "$MODEL" --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
        --max-running-requests 1 --cuda-graph-max-bs 1
        --kv-cache-dtype fp8 --kv-reserve-tokens "$KV_GPU_TOKENS" --kv-host-pages "$KV_HOST_PAGES"
        --max-prefill-length "$MAX_PREFILL_LENGTH" --memory-ratio "$MEMORY_RATIO"
        --moe-strategy hybrid --moe-hybrid-max-fetch "$HYBRID_MAX_FETCH" --expert-load parallel
        --dense-quant mxfp8 --lm-head-quant mxfp8 --hc-quant "$HC_QUANT" --embed-weights host
        --max-output-tokens "$MAX_OUTPUT_TOKENS"
    )
    [ "$VISION" = "1" ] || args+=(--text-model-only)
    [ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")
    printf '%s\n' "${args[@]}"
}

preflight() {
    [ -x "$FT" ] || die "no ft binary at $FT"
    [ -d "$MODEL" ] || die "no model directory at $MODEL"
    command -v numactl >/dev/null || die "numactl is required (the expert banks span both NUMA nodes)"
    mkdir -p "$TMPDIR" "$(dirname "$LOG")"
    if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
        die "port $PORT is already in use"
    fi
}

cmd_run() {
    preflight
    mapfile -t args < <(serve_args)
    echo $$ > "$PIDFILE"
    # Interleaved pages let both sockets' memory controllers feed the CPU expert kernel and
    # the PCIe gather at once (13.1 + 11.6 GB/s overlapped vs 10.4 + 8.2 first-touch).
    exec numactl --interleave=all "$FT" serve "${args[@]}"
}

cmd_start() {
    if running_pid >/dev/null; then
        echo "already running (pid $(running_pid)) on $HOST:$PORT"; return 0
    fi
    preflight
    mapfile -t args < <(serve_args)
    : > "$LOG"
    setsid nohup numactl --interleave=all "$FT" serve "${args[@]}" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo -n "starting (pid $!, log $LOG)"
    local waited=0
    while [ "$waited" -lt "$START_TIMEOUT" ]; do
        if grep -q "API server is ready to serve" "$LOG" 2>/dev/null; then echo " ready on $HOST:$PORT"; return 0; fi
        if ! running_pid >/dev/null || grep -qE "Backend worker is gone|Traceback" "$LOG"; then
            echo; tail -20 "$LOG"; die "failed to start"
        fi
        sleep 5; waited=$((waited + 5)); printf '.'
    done
    echo; die "not ready after ${START_TIMEOUT}s (see $LOG)"
}

cmd_stop() {
    local force="${1:-}" pids
    pids=$(service_pids)
    [ -z "$pids" ] && { echo "not running"; rm -f "$PIDFILE"; return 0; }
    local pid; pid=$(running_pid || true)
    [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null
    kill -TERM $pids 2>/dev/null
    for _ in $(seq 1 30); do
        [ -z "$(service_pids)" ] && { echo "stopped"; rm -f "$PIDFILE"; return 0; }
        sleep 1
    done
    if [ "$force" = "--force" ]; then cmd_kill; else die "still running after 30s; use '$0 stop --force'"; fi
}

cmd_kill() {
    local pids; pids=$(service_pids)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    rm -f "$PIDFILE"; echo "killed: ${pids:-none}"
}

cmd_status() {
    local pid; pid=$(running_pid || true)
    if [ -n "$pid" ]; then echo "running (pid $pid) on $HOST:$PORT"; else echo "not running"; fi
    local pids; pids=$(service_pids)
    [ -n "$pids" ] && ps -o pid,rss,etime,args -p $(echo $pids | tr ' ' ',') | cut -c1-120
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400; echo
}

cmd_test() {
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
        -d "{\"model\":\"$SERVED_NAME\",\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colours.\"}]}"
    echo
}

case "${1:-}" in
    start) cmd_start ;;
    run) cmd_run ;;
    stop) cmd_stop "${2:-}" ;;
    kill) cmd_kill ;;
    status) cmd_status ;;
    logs) tail -f "$LOG" ;;
    test) cmd_test ;;
    *) sed -n '2,14p' "$0"; exit 1 ;;
esac
