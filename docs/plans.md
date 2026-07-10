# ds4-gb10 — open plans and deferred levers

Living document. Each entry carries enough spec to start cold. Items move to
git history when shipped; measured non-wins are recorded in the memory notes,
not here.

## Adaptive conf-sched τ (deferred 2026-07-10)

Today τ is a static threshold (default 0.35; `DS4_DSPARK_CONF_SCHED`
overrides). The principled rule is the breakeven inequality: keep draft
position i while `p(accept_i) × (ms per committed token) > (marginal ms of one
more verify row)`. Both sides move with context and content — the 2026-07-09
sweep measured the optimum drifting (τ≈0.25 best at 32k, 0.35 > 0.45 at 128k)
— so any constant is only right at one operating point.

Design: in the fused loop, maintain EMAs of (a) step_ms deltas between
adjacent n_batch values → marginal row cost, (b) ms per committed token →
token value; `τ_t = clamp(marginal/value, 0.15, 0.60)`, recomputed per step
from stats the loop already collects (host-side arithmetic only).
`DS4_DSPARK_CONF_SCHED=<fixed>` pins static; unset = adaptive.
Output-preserving as ever (trim only bounds the verify budget).

Validation: A/B vs best static at 32k / 128k / 512k (baselines in
temp/tau_sweep2.log, temp/tau512f.log). Expected +2-5% vs best-static at any
fixed point; the real value is workload-proofing (no re-sweeps as ctx/corpus
change). Cannot fix head miscalibration (none observed — acceptance holds
3.79 tok/step at 446k ctx).

## Tree speculation ("DTree") — KILLED at Phase 0 (2026-07-10)

Measured non-win; do not build. Phase 0 measured p₂ = P(target correction ==
drafter #2 | #1 rejected) bucketed by the conf head (commit 6ded65f,
DS4_DTREE_STATS). p₂ is informative (0.44 short / 0.585 @446k) but the sibling
row's marginal yield (1−a(c))·p₂(c) peaks at 0.22 tok/row — below the bar at
every operating point — because a(c) > (1−a)·p₂ at every conf bucket (the
chain's #1 row always out-yields a #2 sibling). Long-ctx thesis refuted:
acceptance RISES with ctx (fewer splits) and the marginal verify row gets more
expensive (~26/34/51 ms at short/128k/446k), so the bar rises rather than
dropping to 0.27. No operating point clears +5% even for an oracle upper bound
(max +4.2% @446k). Full evidence + p₂ table in the ds4-dspark-drafter memory
note; design in docs/dtree-design.md (annotated with the Phase-0 verdict).

## Drafter retune (shelved dessert, ~+3-5%)

Full plan and validated pipeline in docs/drafter-retune-scope.md. Engine is
within ~6 pts of the torch teacher-forced ceiling; residual ≈ requant.

## Quality ceiling

On this box the ceiling is the 2-bit REAP requant itself. Every lossless
container win has been taken (indexer MXFP4, raw f16, attn-comp pack, sliced
work buffers — all bit-exact, all default-on). Recovering quality means
bigger weights (e.g. 3-bit experts, +~10 GB), not engine code.
