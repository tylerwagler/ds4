# DTree: tree speculation for the DSpark fused loop — design (2026-07-10)

Produced by design exploration against the code as of commit 5698afe; every
mechanism cited was verified at the referenced file:line. See docs/plans.md
for the one-paragraph scope; this is the working design.

## 1. Economics — where branching pays

Measured survival marginals at draft=5: S = [0.75, 0.53, 0.36, 0.22, 0.13];
E[tok/step] = 1 + sum(S). Marginal verify row ~= 19.7 ms; per-step fixed
overhead ~= 3.8 ms — the step is almost purely row cost, so TOKENS PER ROW is
the objective. Chain-5 = 2.99 tok / 6 rows = 0.498 tok/row, and the bar for
any added row at 24.5 t/s is ~0.48 tok/row.

KEY NEGATIVE RESULT (analytic): an UNCONDITIONAL top-2 split never clears the
bar — a position-1 sibling yields at most 0.25*p2 <= 0.25 tok/row even at
p2=1. The +10-20% literature number (SpecInfer/Sequoia) assumes near-free
verify rows; ours cost linearly. DTree is therefore a ROW REALLOCATION
problem: displace below-bar chain-tail rows (which conf-sched already trims)
with above-bar sibling rows, gated by the confidence head.

Where it pays: (a) confidence-gated splits at positions scoring p(accept)
0.3-0.5, where the chain tail collapses but P(reject)*p2' can clear the bar;
(b) LONG CONTEXT: at 446k (13.8 t/s, 3.79 tok/step) a token is worth ~72 ms
and the bar drops toward 0.27 tok/row, while acceptance falls (more sibling
opportunities). Realistic envelope: +3-6% short ctx, +8-15% at >=128k.
Everything hinges on p2 = P(target token == drafter #2 | #1 rejected),
bucketed by confidence — CURRENTLY UNMEASURED (Phase 0 measures it).

Note: in greedy mode the corrected token after a rejection is already known
(row_tops[commit] becomes next_base) — a sibling row buys early emission +
KV/state advance + frontier logits/anchor, i.e. one row's worth, not a
re-decode.

## 2. Verify batching — minimal-change design (no CUDA kernel changes)

Siblings can mostly coexist because raw-KV writes are position-addressed and
self-healing under rewind. Two hard exceptions and their fixes:

1. RAW-RING SLOT COLLISION: siblings at position p share slot p % raw_cap
   (ds4_cuda_norm_kv.cu:698). Fix: TWO-LAUNCH ordering per layer with an UNDO
   LOG — (i) A-branch stores + attention exactly as today (bit-identical);
   (ii) copy canonical slots p..p+dB-1 to per-layer undo scratch; (iii) store
   B rows AT canonical slots (store launcher with pos0=p on the B row view);
   (iv) B attention as a plain 1-2-row decode-mixed/indexed launch with
   pos0=p — the existing position-arithmetic mask is then exactly right and A
   never sees B (A's attention already ran). B rows get their own RoPE calls
   at logical positions (rope launcher already takes arbitrary pos0/views).
   Undo log is <= 2 rows x 43 layers x 512 floats, batched via the existing
   ds4_gpu_batched_copy tables. A-win: replay undo. B-win: nothing (B is
   canonical; A's deeper slots become stale-overwritten, as chain rewinds
   already rely on).

2. MID-BATCH COMP EMISSION: the sequential pool recurrence emits a comp row
   at (pos+1)%ratio==0 (gpu_prefill.c:1046) visible to later batch rows — a
   boundary inside [split, split+dB-1] bakes A tokens into a row B can see,
   contaminating gating logits (breaks exact-greedy). Phase 1-3 fix:
   SPLIT-PLACEMENT CONSTRAINT — only split where no ratio-4 boundary falls in
   the B span (3 of 4 positions legal at dB=1). B rows are EXCLUDED from the
   pool recurrence (no state update, no emission); their contribution is
   replayed at commit by the existing Stage-B rollforward. Phase 4 (only if
   skip-rate logs justify): per-branch pool scratch + staged emission via the
   attn_comp stage/commit machinery + the existing per-(t,c) comp_mask path.

Two-phase-commit support today: pools = snapshot + Stage-B redo log; comp
cache = stage/commit; drafter ring = seeded only post-selection. Raw ring is
write-in-place — hence the undo log (deferring B's raw write is impossible:
B's own attention must read it).

## 3. Drafter side

- Exact top-2 from the markov head: extend dspark_markov_step_kernel's
  per-block reduction to track best+second (~25 LOC) + host merge. Do NOT use
  indexer_topk (vocab-shaped single-thread fall-through).
- Branch continuations nearly free: draft forward is noise-token/non-causal,
  so base rows are chain-independent; a B continuation is just extra markov
  steps with prev = alt token. No second draft forward.
- Split selection is free: conf[] is already host-side each step (Step 5a);
  natural split = trim point k when conf[k] is in a mid band (~[0.25,0.65]),
  intersected with the comp-boundary constraint. B tokens conf-scoreable too.

## 4. Commit/rewind deltas (all small)

(1) tree acceptance walk: B alive iff A dies exactly at split AND
row_tops[parent] == alt token (same parent row gates both branches — no extra
row for B's own acceptance); (2) winner ROW-MAP threaded into the Stage-B
rollforward + drafter seeding loops (rows[17] parameter); (3) raw-ring
finalization per §2; (4) winner-row logits/anchor selection; (5) pending
descriptor grows {split_idx, alt_token[, alt_cont]} + staleness drops tree
shape. Output-preservation: emitted tokens are still target argmax given a
bit-exact committed prefix; the tree only changes which rows get COMPUTED —
byte-identical by construction, enforced by the §2 constraint.

## 5. Risk register (abridged; see task history for the full table)

Raw slot arithmetic, causal-mask position math, RoPE position math (fixed by
two-launch + per-view calls); comp-boundary contamination (constraint, then
Phase-4 forking); rollforward "comp rows already correct" (constraint keeps
it true); caps pend[16]/refined[17]/Stage-B 17-row clamp (enforce
1+K_A+K_B <= 16); RAW_F16/ATTN_PACK: undo copies are raw byte copies; DTree
stays greedy-only like the fused loop.

## 6. Phases and gates

- PHASE 0 (~1 day, no behavior change): markov top-2 + per-rejection logging
  of "was target == markov#2", bucketed by conf; suite + 128k + 446k runs.
  GATE: analytic projection >= +5% at >=1 operating point, else STOP and
  record the non-win.
- PHASE 1 (~1 week, DS4_DTREE=1 opt-in): depth-1 sibling at the conf-trim
  position; two-launch verify + undo log; row-map commit. GATE: byte-identical
  output with flag off AND vs off-run (hard gate); eff t/s >= +3% at 128k or
  >= +5% at 446k.
- PHASE 2: B depth 2 + alt continuations + B conf-scoring. GATE: +2% over P1.
- PHASE 3: unified expected-yield-per-row allocator (merges chain trim, split
  choice, and adaptive-tau from docs/plans.md). GATE: >= +10% vs 13.8 t/s at
  446k; short-ctx regression <= 1%.
- PHASE 4 (conditional): pool forking to lift the boundary constraint.

Key files: session.c 4965-5318/971-1127; gpu_prefill.c 393-1980/2390-2450/
3262-3333; ds4_cuda_dspark.cu 12-210; imatrix.c 1081-1217;
ds4_engine_internal.h 1378-1383.

## 7. Prior art: DDTree (arXiv 2604.12989, Ringel & Romano) — added after review

The actual DDTree paper targets exactly our drafter class (block/parallel
drafters a la DFlash/DSpark; "vanilla DFlash verifies only a single drafted
trajectory"). Method: best-first HEAP over products of per-position draft
probabilities selects the top-B tree nodes (budgets 64-512), verified in ONE
target pass with an ANCESTOR-ONLY ATTENTION MASK and LINEAR position ids
(tree encoded in the mask, not positions). Longest-accepted-path rule;
non-accepted KV discarded. Reported 2.3-2.8x vs chain, 1.8-2.2x vs EAGLE-3.
Reference impl: github.com/liranringel/ddtree.

RECONCILIATION WITH OUR ECONOMICS — their speedups do NOT transfer directly:
budgets of 64-512 rows assume near-free verify rows; ours cost 19.7 ms each
(64 rows = 1.3 s/step). Their own "when it doesn't pay" list (small budgets,
high acceptance) is our operating regime at short ctx. What DOES transfer:
1. TREE CONSTRUCTION: replace our single-split heuristic with their
   best-first heap over markov-head path-probability products, just at tiny
   budgets (B = 6-10 rows chosen by expected-yield-per-row against the live
   bar — merging naturally into the Phase-3 unified allocator). The conf
   head becomes a calibrated prior on top of the path products.
2. MASKED SINGLE-PASS VERIFY is the scalable alternative to our two-launch
   undo-log scheme; our kernels' causality is position-arithmetic, so the
   mask path needs kernel work (the per-(t,c) comp_mask additive path is the
   seed). Keep two-launch for Phase 1 (zero kernel changes, B<=2); adopt
   ancestor-mask if Phase 2+ budgets grow. Their linear-position-ids choice
   confirms sibling rows CAN share positions once masking is explicit.
3. Phase 0 unchanged and reinforced: it measures the calibration of exactly
   the path probabilities the heap ranks by (p2-by-confidence is the B=2
   special case).
