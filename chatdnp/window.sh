#!/usr/bin/env bash
# GPU window on chatdnp: stop production, serve a candidate tree, probe it, ALWAYS restore production.
#   LABEL=w1 TREE=~/ft-candidate CAND_ENV="FREETOKEN_FP8_DENSE=1" CAND_FLAGS="..." TESTS="smoke vision bench" window.sh
# Start it detached, because the window continues after ssh closes:
#   nohup setsid env LABEL=... ./window.sh > /dev/null 2>&1 < /dev/null &
# The candidate binds 127.0.0.1:8000 only, so clients get connection errors during the window, never a test server.
# All output, and the full output of each test, goes to $OUT/${LABEL}_window.log. The script writes
# to that file directly and not through a pipe: a dead reader must not stop the script before restore().
# Knobs: PYC (python for the tools, default $TREE/.venv/bin/python; it needs requests and Pillow),
# OUT (default ~/logs/windows), READY_MIN, SETTLE_S, MODEL, GPUTESTS, GPUCMD.
set -u
export PATH=$HOME/.local/bin:/usr/local/cuda-13/bin:$PATH CUDA_HOME=/usr/local/cuda-13 HF_HOME=/scratch/localadmin/hf-cache
export PYTHONUNBUFFERED=1
D=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
LABEL=${LABEL:?}; TREE=${TREE:?}; CAND_ENV=${CAND_ENV:-}; CAND_FLAGS=${CAND_FLAGS:?}
TESTS=${TESTS:-smoke vision bench}; READY_MIN=${READY_MIN:-20}
PYC=${PYC:-$TREE/.venv/bin/python}
OUT=${OUT:-$HOME/logs/windows}
mkdir -p "$OUT" || exit 1
OUT=$(cd "$OUT" && pwd)
LOG=~/logs/ft_$LABEL.log
WLOG=$OUT/${LABEL}_window.log
echo "window log: $WLOG"
exec 3>&1 >> "$WLOG" 2>&1
NV=$(ls -d "$HF_HOME"/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/* | head -1)
NV=${MODEL:-$NV}   # MODEL=<snapshot dir> serves another checkpoint in the window
ts() { date "+%F %T"; }
say() { echo "[$(ts)] $*"; }

wait_free() {  # no server on :8000/:8001 and both GPUs nearly empty
  for i in $(seq 1 60); do
    busy=$(ss -ltn | grep -cE ":800[01] ")
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$busy" = 0 ] && [ "$mem" -lt 2000 ] && { say "GPUs free (max used ${mem} MiB)"; return 0; }
    # anchored on the python interpreter: the tmux server's argv keeps its first "... bin/ft serve ..."
    # command, and an unanchored pattern would kill every tmux session, production included
    [ "$i" = 20 ] && { say "still busy after 40 s: killing stray ft serve"; pkill -f "^[^ ]*/python[0-9.]* [^ ]*/bin/ft serv[e]"; }
    sleep 2
  done
  say "GPUs NOT free: busy=$busy mem=$mem"; return 1
}

ready() {  # a 200 from a 1-token chat completion; /v1/models answers while weights load
  local deadline=$(( $(date +%s) + 60 * $1 )) sess=$2
  while [ "$(date +%s)" -lt "$deadline" ]; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 http://127.0.0.1:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')
    [ "$code" = 200 ] && return 0
    tmux has-session -t "$sess" 2>/dev/null || { say "tmux session $sess is gone"; return 1; }
    sleep 10
  done
  return 1
}

restore() {
  say "=== RESTORE production"
  tmux kill-session -t ftc 2>/dev/null || true
  wait_free || true
  # run_qwen38.sh reads MODEL too: an inherited MODEL once kept production down
  local run=(env -u MODEL -u TREE -u CAND_ENV -u CAND_FLAGS "$HOME/run_qwen38.sh")
  local prod
  prod=$("${run[@]}" check | sed -n 's/^cd \([^ ]*\) .*/\1/p')
  say "production tree $prod at $(git -C "$prod" log --oneline -1)"
  t0=$(date +%s)
  if "${run[@]}" start; then
    say "production READY after $(( $(date +%s) - t0 )) s"
  else
    say "PRODUCTION DID NOT COME UP -- see ~/logs/ft.log"; tail -30 ~/logs/ft.log
  fi
  ss -ltn | grep -E ":800[01] "
}

say "=== WINDOW $LABEL: tree $TREE at $(git -C "$TREE" log --oneline -1)"
say "tools $D at $(git -C "$D" log --oneline -1), out $OUT"
say "stopping production"
tmux kill-session -t ft 2>/dev/null || true
wait_free || { restore; echo "WINDOWDONE $LABEL (candidate not started)" >&3; exit 1; }

cmd="cd $TREE && env $CAND_ENV $TREE/.venv/bin/ft serve --model $NV --served-model-name Qwen3.8-Flash-Next --host 127.0.0.1 --port 8000 --tp-size 2 --gpu 0,1 $CAND_FLAGS"
say "candidate: $cmd"
: > "$LOG"
t0=$(date +%s)
tmux new-session -d -s ftc "$cmd 2>&1 | tee -a $LOG"
if ! ready "$READY_MIN" ftc; then
  say "CANDIDATE FAILED to become ready after $(( $(date +%s) - t0 )) s"
  grep -aE "Error|error|Traceback|assert|Exception" "$LOG" | sed -E "s/\x1b\[[0-9;]*m//g" | tail -40
  tail -40 "$LOG" | sed -E "s/\x1b\[[0-9;]*m//g"
  restore; echo "WINDOWDONE $LABEL (candidate failed)" >&3; exit 1
fi
say "candidate READY after $(( $(date +%s) - t0 )) s"
grep -aE "Allocating .* tokens for KV cache|expert slots|residency|moe cache|FP8|fp8|encoder|vision" "$LOG" | sed -E "s/\x1b\[[0-9;]*m//g" | tail -15

cd "$D" || { restore; exit 1; }
for t in $TESTS; do
  case $t in
    smoke)
      S=$(curl -s -m 120 http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
        -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"What is 17 times 23? Reply with only the number."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' \
        | $PYC -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>&1)
      say "SMOKE 17*23 -> $S" ;;
    vision) say "--- vision_probe"; $PYC vision_probe.py
            say "--- image_hit_probe"; $PYC image_hit_probe.py ;;
    disconnect) say "--- disconnect_probe"; FT_LOG=$LOG $PYC disconnect_probe.py ;;
    bench)
      # each run has its own seed: the same seed again would hit the prefix cache and skew TTFT
      say "--- bench_client warm-up (concurrent 4)"
      $PYC bench_client.py --base-url http://127.0.0.1:8000 --repeats 1 --concurrent 4 --seed 100 --out "$OUT/${LABEL}_warm.json"
      for r in 1 2 3; do
        say "--- bench_client run $r (concurrent 8)"
        $PYC bench_client.py --base-url http://127.0.0.1:8000 --repeats 1 --concurrent 8 --seed $((100 + r)) --out "$OUT/${LABEL}_reb$r.json"
      done ;;
    settle) say "--- settle: idle ${SETTLE_S:-480} s (a new server benches lower for its first minutes)"; sleep "${SETTLE_S:-480}"; say "settle done" ;;
    gsm8k)  say "--- GSM8K-300 @1536"; $PYC gsm8k_probe.py --n 300 --concurrency 8 --max-tokens 1536 --label "${LABEL}-1536" --out "$OUT/${LABEL}_gsm8k_1536.json" ;;
    sampling) say "--- sampling_probe"; $PYC sampling_probe.py ;;
    clientpaths) say "--- client_paths_probe"; $PYC client_paths_probe.py ;;
    stall)  say "--- stall_probe (200k prefill vs a live stream)"; $PYC stall_probe.py --tokens 200000 --out "$OUT/${LABEL}_stall.json" ;;
    revisit) say "--- revisit_probe (16 sessions x 24k)"; $PYC revisit_probe.py --sessions 16 --tokens 24000 --out "$OUT/${LABEL}_revisit.json" ;;
    revisitheavy) say "--- revisit_probe (12 sessions x 48k)"; $PYC revisit_probe.py --sessions 12 --tokens 48000 --out "$OUT/${LABEL}_revisit_heavy.json" ;;
    geometry) say "--- geometry"; curl -s -m 10 http://127.0.0.1:8000/v1/cache/status | $PYC -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d.get(k) for k in ('state','geometry','pools','unit_bytes','budget') if k in d}))" 2>&1 ;;
    listen)
      say "--- listening sockets of the candidate's processes (no LAN address expected)"
      pids=$(pgrep -f "$(basename "$TREE")/.venv/bin/pytho[n]" | paste -sd"|")
      ss -ltnpH 2>/dev/null | grep -E "pid=(${pids:-none})," | awk '{print $4}' | sort | uniq -c ;;
    *) if [ -x "$D/$t" ]; then say "--- extra $t"; FT_LOG=$LOG "$D/$t"; else say "--- unknown test $t: skipped"; fi ;;
  esac
done
say "--- errors in the candidate log"
grep -aE "Traceback|Error|assert" "$LOG" | sed -E "s/\x1b\[[0-9;]*m//g" | tail -10
say "=== CANDIDATE DONE"
if [ -n "${GPUTESTS:-}" ]; then
  say "--- GPU tests on the candidate tree (candidate stopped first)"
  tmux kill-session -t ftc 2>/dev/null || true; wait_free || true
  (cd "$TREE" && timeout 1500 .venv/bin/python -m pytest -q -p no:cacheprovider --timeout 300 $GPUTESTS 2>&1)
fi
if [ -n "${GPUCMD:-}" ]; then
  say "--- GPU command (candidate stopped): $GPUCMD"
  tmux kill-session -t ftc 2>/dev/null || true; wait_free || true
  (timeout 1500 bash -c "$GPUCMD" 2>&1)
fi
restore
say "=== WINDOWDONE $LABEL"
echo "WINDOWDONE $LABEL" >&3
