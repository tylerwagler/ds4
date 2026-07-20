#!/bin/bash
# plan-33 inc D gate — THE server-fork proof: PARTIAL-prefix fork routing on the
# DOMINANT shape (a LARGE shared text prefix + divergent user tails) + the TTFT
# payoff number.
#
# CONSTRUCTION (settled with the coordinator): the partial fork only fires, and
# only matters, on a LARGE shared prefix. Appended detokenized GENERATION diverges
# at the trunk/gen junction (~47 tok in) — so the shared region must be literal
# TEXT, identical across branches, with NO generated text inside it. Then P
# tokenizes stably and common ~= |P| > warm_partial_min.
#
#   P            : a long shared system+few-shot TEXT block (~600 tokens), used
#                  verbatim as the literal start of every branch prompt.
#   R1 = P+tail0 : establishes the trunk (bank A), commits -> frontier > |P|.
#   R2 = P+tailA : common ~= |P| (diverges at the P/tail boundary),
#                  256 <= common < frontier -> PARTIAL fork A->B (A intact).
#   R3 = P+tailB : A still intact -> second PARTIAL fork A->C.
#
# Full-fork (common==frontier) is rare-by-design (exact continuation, hard to hit
# via text) and its engine primitive is already proven by the fork gate (A); this
# gate proves the SERVER partial path.
#
# ASSERTS:
#  (i)   OUTPUT-TOKEN identity fork-vs-warm-vs-cold for each branch (temp-0 text is
#        the token-stream oracle; NOT KV bytes — the partial cut resumes at R, the
#        unaligned last-ulp class, see fork-gate P5).
#  (ii)  >=2 warm-fork-partial fired (grep `warm-fork-partial:`), trunk preserved.
#  (iii) TTFT payoff: warm-forked branch vs cold branch (max_tokens=1 wall time).
# Run manually under GPU discipline (flock, drop_caches, watchdog, one load).
#   usage: tests/warm_partial_fork_3way.sh [MODEL] [PORT]
set -u
MODEL=${1:-gguf/model.gguf}; PORT=${2:-8902}
DIR=$(mktemp -d /tmp/warmpartial.XXXX); LOG=$DIR/server.log
# ~600-token shared system+few-shot TEXT block (literal, identical across branches).
P=$(python3 -c "
import sys
hdr='You are a precise assistant. Follow the format exactly.\n'
shot='Example %d: given input, respond with a single factual sentence and stop.\n'
print(hdr + ''.join(shot%i for i in range(1,120)))")
T0="$P
Task-0: name a color."
TA="$P
Question-Alpha: name two prime numbers."
TB="$P
Question-Bravo: name a European river."
TT="$P
Probe-Gamma: name a mountain."     # dedicated TTFT tail (fresh partial fork)
jreq(){ python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_tokens':int(sys.argv[2]),'temperature':0.0,'stream':False}))" "$1" "$2"; }
start(){ DS4_WARM_FORK=$1 DS4_MSEQ_BANKS=3 DS4_WARM_PARTIAL_MIN=256 ./ds4-server -m "$MODEL" \
        --host 127.0.0.1 --port $PORT -c 32768 --kv-disk-dir "" >"$LOG" 2>&1 & SP=$!
  for i in $(seq 1 240); do [ "$(curl -s -m2 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ] && return 0; sleep 1; done; return 1; }
stop(){ kill -INT $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; }
gen(){ curl -s -m 600 http://127.0.0.1:$PORT/v1/completions -H 'content-type: application/json' \
        -d "$(jreq "$1" "$2")" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['text'])"; }
# wall time (seconds) of a max_tokens=1 request == TTFT proxy (prefill + 1 token).
ttft(){ curl -s -m 600 -o /dev/null -w '%{time_total}' http://127.0.0.1:$PORT/v1/completions \
        -H 'content-type: application/json' -d "$(jreq "$1" 1)"; }

# (F) warm-partial-fork: trunk then two branches partial-forking off it.
echo "[Dgate] (F) warm-partial-fork (branching)"; start 1 || { echo FAIL-start; exit 1; }
gen "$T0" 24 >/dev/null                                    # R1: build trunk A
FA=$(gen "$TA" 48)                                          # R2: partial fork A->B
FB=$(gen "$TB" 48)                                          # R3: A intact -> partial fork A->C
T_WARM=$(ttft "$TT")                                        # TTFT: warm partial fork off A
PF=$(grep -c "warm-fork-partial:" "$LOG")
PRES=$(grep -c "trunk preserved" "$LOG")
stop

# (W) warm continuation (fork off): same requests, in-place rewind on A.
echo "[Dgate] (W) warm continuation (fork off)"; start 0 || { echo FAIL-start; exit 1; }
gen "$T0" 24 >/dev/null; WA=$(gen "$TA" 48); WB=$(gen "$TB" 48); stop

# (C) cold: fresh server per branch (no trunk in KV).
echo "[Dgate] (C) cold per branch"
start 0 || { echo FAIL-start; exit 1; }; CA=$(gen "$TA" 48); stop
start 0 || { echo FAIL-start; exit 1; }; CB=$(gen "$TB" 48); stop
start 0 || { echo FAIL-start; exit 1; }; T_COLD=$(ttft "$TT"); stop

fail=0
[ "$FA" = "$WA" ] || { echo "MISMATCH branch A: fork vs warm"; fail=1; }
[ "$FA" = "$CA" ] || { echo "MISMATCH branch A: fork vs cold"; fail=1; }
[ "$FB" = "$WB" ] || { echo "MISMATCH branch B: fork vs warm"; fail=1; }
[ "$FB" = "$CB" ] || { echo "MISMATCH branch B: fork vs cold"; fail=1; }
echo "partial forks fired: $PF (trunk-preserved: $PRES)"
[ "$PF"   -ge 2 ] || { echo "EXPECTED >=2 partial forks (both branches), got $PF"; fail=1; }
[ "$PRES" -ge 1 ] || { echo "trunk never reported preserved"; fail=1; }
SPEEDUP=$(python3 -c "w=$T_WARM;c=$T_COLD;print('%.2f'%(c/w) if w>0 else 'inf')")
echo "TTFT payoff: warm-fork ${T_WARM}s vs cold ${T_COLD}s -> ${SPEEDUP}x faster"
echo "WARM-PARTIAL-FORK GATE: $([ $fail = 0 ] && echo PASS || echo FAIL)"; exit $fail
