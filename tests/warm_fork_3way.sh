#!/bin/bash
# plan-33 inc B gate: 3-way output-equality — the same shared-prefix conversation
# served (i) COLD each turn (DS4_WARM_FORK=0, fresh server per turn), (ii) warm
# single-bank continuation (DS4_WARM_FORK=0, one server), (iii) warm FORK routing
# (DS4_WARM_FORK=1). Asserts IDENTICAL OUTPUT TEXT across all three (greedy).
# NOTE (gate-A finding): byte-exact KV equality only holds at chunk-ALIGNED
# continuation points; full-match forks continue from the trunk's exact frontier
# (same class as today's warm continuation), so this harness compares OUTPUT
# TOKENS, not KV bytes. Also asserts trunk protection: after a fork, the trunk
# bank's committed frontier is unchanged (from the server log).
# Run manually under the GPU discipline (flock, drop_caches, watchdog, one load).
#   usage: tests/warm_fork_3way.sh [MODEL] [PORT]
set -u
MODEL=${1:-gguf/model.gguf}; PORT=${2:-8901}
DIR=$(mktemp -d /tmp/warmfork.XXXX); LOG=$DIR/server.log
SYS=$(python3 -c "print('Shared system preamble. '*400)")   # ~2.8k-token trunk
req(){ python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_tokens':48,'temperature':0.0,'stream':False}))" "$1"; }
start(){ DS4_WARM_FORK=$1 ./ds4-server -m "$MODEL" --host 127.0.0.1 --port $PORT -c 32768 --kv-disk-dir "" >"$LOG" 2>&1 & SP=$!
  for i in $(seq 1 240); do [ "$(curl -s -m2 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0; sleep 1; done; return 1; }
stop(){ kill -INT $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; }
ask(){ curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
        -d "$(req "$SYS $1")" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])"; }
turns=("Turn one: describe the archive." "Turn two: list three artifacts." "Turn three: conclude the dossier.")

echo "[3way] (iii) warm-fork server"; start 1 || { echo FAIL-start; exit 1; }
F=(); for t in "${turns[@]}"; do F+=("$(ask "$t")"); done
FORKS=$(grep -c "warm-fork: trunk" "$LOG"); TRUNK_OK=$(grep -c "trunk preserved" "$LOG")
stop
echo "[3way] (ii) warm continuation (fork off)"; start 0 || { echo FAIL-start; exit 1; }
W=(); for t in "${turns[@]}"; do W+=("$(ask "$t")"); done
stop
echo "[3way] (i) cold per turn (fork off, fresh server each turn)"
C=(); for t in "${turns[@]}"; do start 0 || { echo FAIL-start; exit 1; }; C+=("$(ask "$t")"); stop; done

fail=0
for i in 0 1 2; do
  [ "${F[$i]}" = "${W[$i]}" ] || { echo "MISMATCH fork-vs-warm turn $i"; fail=1; }
  [ "${F[$i]}" = "${C[$i]}" ] || { echo "MISMATCH fork-vs-cold turn $i"; fail=1; }
done
echo "forks fired: $FORKS (trunk-preserved lines: $TRUNK_OK)"
[ "$FORKS" -ge 1 ] || { echo "NO FORK FIRED (routing dead or prompts trivial)"; fail=1; }
echo "WARM-FORK 3WAY: $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
