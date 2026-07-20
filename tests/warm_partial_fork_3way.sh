#!/bin/bash
# plan-33 inc D gate: PARTIAL-prefix fork routing — the DOMINANT single-turn shape.
#
# Shape: a large SHARED system/few-shot TRUNK + a divergent user tail. This is the
# ~7x-TTFT payoff path: every request pays for the shared preamble once, then each
# divergent turn reuses it via a PARTIAL fork-cut instead of cold-prefilling it.
#
#   R1 : SYS + " Alpha: <question 1>"  -> bank A, generates, frontier = |SYS+Q1+gen1|
#   R2 : SYS + " Zeta:  <question 2>"  -> common == |SYS| (< frontier): PARTIAL match
#        -> ds4_session_bank_fork_partial cuts A at R=align_down(|SYS|) into bank B,
#           re-prefills [R..]; trunk A stays intact.
#
# SYS is ~300 tokens (>= DS4_WARM_PARTIAL_MIN=256) and deliberately NOT a 4096-chunk
# multiple, so the fork resumes from an UNALIGNED source length R — the production-
# realistic class where the resumed compressor carries the accepted warm-continuation
# last-ulp KV delta. Hence the oracle is the OUTPUT TOKEN STREAM (greedy/temp-0 text),
# NOT KV byte-identity (which only holds at chunk-aligned cuts, proven by C-P2).
#
# Three ways, assert IDENTICAL R2 output text:
#   (F) warm-partial-fork  DS4_WARM_FORK=1 -> partial cut fires, trunk preserved
#   (W) warm continuation  DS4_WARM_FORK=0 -> in-place rewind of A to |SYS|, prefill Q2
#   (C) cold               fresh server, R2 only -> full cold prefill, no reuse
# plus: >=1 "warm-fork-partial" fired with a cut logged, and a trunk-preserved line.
# Run manually under GPU discipline (flock, drop_caches, watchdog, one load).
#   usage: tests/warm_partial_fork_3way.sh [MODEL] [PORT]
set -u
MODEL=${1:-gguf/model.gguf}; PORT=${2:-8902}
DIR=$(mktemp -d /tmp/warmpartial.XXXX); LOG=$DIR/server.log
# ~300-token shared preamble (150 * "word "), unaligned to the 4096 prefill chunk.
SYS=$(python3 -c "print('You are a meticulous assistant. '+'context word '*150)")
Q1="$SYS
Alpha: Summarize the first policy in one sentence."
Q2="$SYS
Zeta: List two unrelated prime numbers."
jreq(){ python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_tokens':int(sys.argv[2]),'temperature':0.0,'stream':False}))" "$1" "$2"; }
start(){ DS4_WARM_FORK=$1 DS4_MSEQ_BANKS=3 DS4_WARM_PARTIAL_MIN=256 ./ds4-server -m "$MODEL" \
        --host 127.0.0.1 --port $PORT -c 32768 --kv-disk-dir "" >"$LOG" 2>&1 & SP=$!
  for i in $(seq 1 240); do [ "$(curl -s -m2 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0; sleep 1; done; return 1; }
stop(){ kill -INT $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; }
gen(){ curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
        -d "$(jreq "$1" "$2")" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])"; }

# (F) warm-partial-fork: R1 establishes the trunk, R2 partial-forks off it.
echo "[Dgate] (F) warm-partial-fork"; start 1 || { echo FAIL-start; exit 1; }
gen "$Q1" 24 >/dev/null                                    # R1: build trunk A
F2=$(gen "$Q2" 48)                                          # R2: partial fork A->B
PF=$(grep -c "warm-fork-partial: trunk" "$LOG")            # partial forks that fired
CUTLOG=$(grep -c "warm-fork-partial: .*resume" "$LOG")     # cut/resume logged
PRES=$(grep -c "trunk preserved" "$LOG")
stop

# (W) warm continuation (fork off): same two requests, in-place rewind on A.
echo "[Dgate] (W) warm continuation (fork off)"; start 0 || { echo FAIL-start; exit 1; }
gen "$Q1" 24 >/dev/null; W2=$(gen "$Q2" 48); stop

# (C) cold: fresh server, R2 only (no trunk in KV at all).
echo "[Dgate] (C) cold"; start 0 || { echo FAIL-start; exit 1; }; C2=$(gen "$Q2" 48); stop

fail=0
[ "$F2" = "$W2" ] || { echo "MISMATCH R2: partial-fork vs warm-continuation"; fail=1; }
[ "$F2" = "$C2" ] || { echo "MISMATCH R2: partial-fork vs cold"; fail=1; }
echo "partial forks fired: $PF (cut/resume logged: $CUTLOG, trunk-preserved: $PRES)"
[ "$PF"     -ge 1 ] || { echo "EXPECTED >=1 partial fork, got $PF"; fail=1; }
[ "$CUTLOG" -ge 1 ] || { echo "partial fork fired without a cut/resume log line"; fail=1; }
[ "$PRES"   -ge 1 ] || { echo "trunk never reported preserved"; fail=1; }
echo "WARM-PARTIAL-FORK 3WAY: $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
