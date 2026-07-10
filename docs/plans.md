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

## Tree speculation ("DTree") — ACTIVE next

DSpark drafts a single chain; a tree verifies alternative branches at
low-confidence positions in one batch (typical +10-20% over chain spec).
Needs: branch-aware verify batching (positions share a prefix; attention masks
per branch), KV forking/rewind per branch (the spec frontier
snapshot/restore machinery generalizes), drafter multi-sample at split points
(markov head already yields a distribution), and accept logic that picks the
surviving branch. The only remaining double-digit decode lever — verify-row
marginal cost is bandwidth physics (19.7 ms/row), so the win must come from
higher accepted-tokens-per-row, which trees provide.

## Drafter retune (shelved dessert, ~+3-5%)

Full plan and validated pipeline in docs/drafter-retune-scope.md. Engine is
within ~6 pts of the torch teacher-forced ceiling; residual ≈ requant.

## Quality ceiling

On this box the ceiling is the 2-bit REAP requant itself. Every lossless
container win has been taken (indexer MXFP4, raw f16, attn-comp pack, sliced
work buffers — all bit-exact, all default-on). Recovering quality means
bigger weights (e.g. 3-bit experts, +~10 GB), not engine code.
