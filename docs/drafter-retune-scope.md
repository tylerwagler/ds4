# Scope: Retuning the DSpark Drafter Against Our Quantized Engine

**Goal.** Lift DSpark block acceptance from ~40% (draft=2, measured on
oracle-reap25-compact-mxfp8head via the ds4 engine) toward the reference's 50%+
by fine-tuning the drafter against **our deployed inference function** — REAP-pruned,
2-bit/MXFP8-requantized hidden states — instead of the bf16 model it shipped
trained on. Every on-box lever is exhausted (fused loop, conditioning fix,
verify-kernel work: spec step 134→101.5 ms); acceptance is the only remaining
sign-changer. The gap is understood: the drafter forward is implementation-correct
(bit-exact bookends vs a torch reference), but its inputs drifted — H2, the
requant residual.

**Payoff math.** At draft=2 and ~101 ms/step (short ctx), per-position acceptance
40%→55% moves tokens/step 1.73→~2.1 → **~17.3 → ~21 t/s (+20%)**, and turns the
ctx-2k parity into a clear win. Diminishing but real returns continue toward the
reference's 50.8%-block-accept regime.

---

## What exists already (assets)

| Asset | Where | Why it matters |
|---|---|---|
| Training recipe + reference trainer | [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT) | Ships DSpark/DFlash/Eagle3 trainers, data prep, eval. Losses: CE (0.1) + total-variation distribution match (0.9, direct acceptance proxy) + BCE on the confidence head vs analytic acceptance labels; target frozen. |
| Pre-trained V4-Flash drafter | HF checkpoint (`mtp.*`) + our `dspark-drafter.gguf` | **Warm start** — this is a retune/calibration, not from-scratch training. DeepSpec's 38 TB cache warning applies to their from-scratch corpus, not to us. |
| Validated torch drafter reference | `temp/dspark_ref.py` + `temp/dspark-ref-venv` | Bit-exact vs the engine on main_x and head bookends — the training forward can be built on it with high confidence that train-time == deploy-time semantics. |
| Hidden-state extraction | `dspark_dump_step` (session.c), `DS4_DSPARK_DUMP` | Already reads target_h[40/41/42], main_x, logits, per-position cur_hc off the engine. Needs a bulk prefill-time variant + top-k logits. |
| Converter | `converter/dspark_convert.py` | Retrained HF drafter → gguf → immediate on-box A/B. Closes the loop in hours. |
| Trainer hardware | This GB10 (128 GB unified) + Docker | NGC ≥25.10 / cu130 wheels run torch training on sm_121 (unofficial but community-proven). No flash-attn needed at 3-block scale (SDPA). 3.3 TB free on pve1-fast for caches. |

## Drafter size reality check

The drafter is 3 **full V4 blocks including MoE** (~20B params; the 10.26 GB gguf
at ~4 bits/param checks out). Full fine-tune in bf16 = ~160 GB optimizer state —
does **not** fit. The scoped plan freezes the expert FFNs (the requant mismatch
lives plausibly in attention/router/norms/heads, and experts carry the knowledge)
→ trainable ≈ 0.5–1B params → weights 40 GB frozen bf16 + optimizer ~8 GB +
activations fits the GB10 comfortably. LoRA-on-experts is the fallback if
frozen-expert results plateau.

Compute is not the constraint: ~0.7B active params/token forward → 50M tokens ×
3 epochs ≈ hours-per-epoch on GB10.

---

## Phases

**Phase 0 — confidence/markov calibration only (~1 day, near-free).**
Extend the dump to record (confidence-head inputs, accept outcome) at scale;
retrain just the confidence head (+ markov) — MLP-scale, minutes to train.
Current head AUC is 0.720 on our numerics; a calibrated head makes
conf-scheduled trimming actually selective. Worth a few % on its own and
exercises the export→convert→measure loop end to end.

**Phase 1 — data generation (on-box, ds4 engine).**
Add a prefill-time bulk dumper: per position, target_h[40/41/42] (3×4096 bf16 ≈
24 KB), token id, and top-64 target logits (~0.5 KB, for the TV loss). Corpus:
20–50M tokens of mixed text/code/chat — primary candidate is **NVIDIA
OpenCode-Instruct** (5M coding instruction samples; the Spark Auto-Round work
found it substantially outperforms pile-10k / github-code-clean as a
calibration corpus on this class of workload, and coding/agentic traffic is
what this box serves), blended with general text for coverage. At 428 t/s
prefill ≈ 13–32 h of dumping; 0.5–1.2 TB on pve1-fast (fits 3.3 TB free). This
data *must* come from our engine — it IS the point — so on-box generation is
mandatory regardless of where training runs.

**Phase 2 — training (on-box, NGC pytorch container).**
Port DeepSpec's DSpark objective onto a `dspark_ref.py`-based forward reading the
offline cache (SpecForge-style offline mode). Freeze experts + embeddings + LM
head; train attention/router/norms/main_proj/hc_head/markov/confidence. Losses
per DeepSpec: 0.1·CE + 0.9·TV + BCE(confidence, analytic accept labels).
**Pilot first**: 5M tokens, 1 epoch, CE-only → measure acceptance delta on-box
before committing to the full run. The pilot is the cheap test of H2's
tractability (~2 days including plumbing).

**Phase 3 — evaluation loop.**
Export → `dspark_convert.py` → gguf → server A/B (the exact harness from this
week: per-n_batch step_ms + acceptance + text). Gates: block acceptance (target
≥50%), tokens/step, t/s at ctx 250 and ctx 2k. Output quality needs no gate —
spec decode is output-lossless by construction; only speed changes.

## Risks

- **H2 partially wrong** — some of the 40% ceiling may be inherent block-drafter
  quality, not requant drift. The pilot bounds this for ~2 days of effort before
  the full spend.
- **TV loss needs the target distribution** — top-64 logits is an approximation;
  if insufficient, re-dump with larger k (storage scales linearly).
- **Unified-memory pressure during training** — 40 GB frozen weights + optimizer
  + activations is comfortable on paper; grad checkpointing and 8-bit optimizer
  are the fallbacks. Model-load discipline applies (never co-resident with the
  76 GB inference model; phase dump/train).
- **sm_121 torch is unofficial** — community-proven via NGC ≥25.10/cu130, but
  expect a day of container/wheel friction. Direct precedent that
  gradient-based optimization loops run fine on GB10 unified memory:
  **Spark Auto-Round** (a GB10-tuned AutoRound fork) runs 1000-iteration
  torch rounding-optimization at batch 8 / seqlen 2048 against 27–35B models
  entirely on-box.

## Alternatives considered

- **NVIDIA Model-Optimizer (modelopt)** — trains Medusa/EAGLE draft modules
  (HF + Megatron flows, frozen base). It does not implement the DSpark
  architecture, and documents neither offline hidden-state training nor
  quantized-target support, so it is not the primary tool here. It is the
  fallback reference if we ever abandon the DSpark drafter for an EAGLE3-style
  head (DeepSpec also ships an Eagle3 trainer, so even then DeepSpec likely
  remains the better starting point).
- **Rent an 8-GPU node + DeepSpec as-is** — faster wall-clock for the training
  phase, but Phase 1 (engine dumps) must happen on this box regardless, the
  drafter-scale compute genuinely fits the GB10, and adapting DeepSpec's
  target-model plumbing to read our offline cache is the same work either way.
  Revisit only if on-box training iteration proves too slow in the pilot.

## Estimate

~1.5–2 weeks calendar, all on-box, no external compute spend:
Phase 0 ≈ 1 day → pilot ≈ 2 days → full data gen 1–2 days (wall-clock, mostly
unattended) → training + 2–3 iteration cycles ≈ 3–5 days.

## Sources

- [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) — trainer, MIT, 8-GPU default configs, 38 TB from-scratch cache warning
- [MarkTechPost: DSpark release](https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/) — loss structure (CE + TV 0.9 + confidence BCE, frozen target)
- [VentureBeat: DSpark open-sourced](https://venturebeat.com/orchestration/deepseek-open-sources-dspark-a-new-framework-to-speed-up-llm-inference-by-up-to-85)
- [LMSYS SpecForge](https://www.lmsys.org/blog/2025-07-25-spec-forge/) — offline vs online hidden-state cache modes (we use offline)
- [NVIDIA forums: GB10 sm_121 training support](https://forums.developer.nvidia.com/t/roadblock-dgx-spark-pytorch-2-8-and-ngc-25-09-missing-support-for-gb10-gpus-with-sm-121-for-flux-training/351552) + [DGX Spark ML guide](https://github.com/martimramos/dgx-spark-ml-guide) — NGC ≥25.10 / cu130 wheels work on GB10
- [Spark Auto-Round + OpenCode-Instruct](https://forums.developer.nvidia.com/t/introducing-spark-auto-round-w-opencode-instruct-dataset/373475) — GB10 on-box optimization-loop precedent; OpenCode-Instruct as calibration corpus
- [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) + [spec-decode guide](https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html) — Medusa/EAGLE drafter training reference (no DSpark support)
