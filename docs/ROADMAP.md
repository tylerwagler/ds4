# ds4-spark Roadmap — best single-Spark DeepSeek-V4-Flash

**North star:** make DeepSeek-V4-Flash genuinely *good and fast* on one DGX Spark
(GB10/Blackwell, 128 GiB unified): high single-stream decode **and** real
concurrent serving. Quality + speed are the constraint; cost is not.

## Bottleneck map (decides which lever helps where)
- **Single-stream decode** = **memory-bandwidth bound** (read selected experts + KV/token).
  Levers: **MTP** (more useful tokens per memory sweep); smaller quant for the bulk.
  *Tensor cores DON'T help here* (batch=1, no compute to saturate).
- **Prefill** = **compute bound**. Levers: FP8 / FP4 tensor cores.
- **Concurrency (N streams)** = **compute bound** (batched grouped-GEMM). Lever: **MXFP4**.
  This is the serving play.

Measured (2026-06-29, oracle-fp8wh-99gb, FP8 workhorse path, no spec decode, un-clock-pinned):
decode ~25.5 t/s @depth0 / 24.5 @4k / 21.6 @8k (c1); prefill 431 / 399 / 389 t/s. That ~25.5
is ~74% of the ~32 t/s memory-BW ceiling (14B active) — the graph bug is fixed and we're now
near the bandwidth wall, so the remaining single-stream win is spec decode, not quant. (Old
smart-99gb baseline was ~12.7 — the 2x gap was the graph bug + greedy decode, now closed.)

**UNVERIFIED (2026-07-02): this ~25.5 t/s decode number does not currently reproduce.**
`oracle-fp8wh-99gb` doesn't exist on disk anymore. Fresh `ds4-bench` runs today against
`oracle-zeroq8-99gb.gguf` on this same GB10 show decode at ~13 t/s (12.98-13.14 across
2k-8k context) — roughly half this line's claim. Root cause not yet investigated: could be
clock/power state, could be an actual regression since 2026-06-29, could be this number was
never real under normal conditions. Don't treat "~25.5 t/s" or "~32 t/s memory-BW ceiling" as
current fact for any comparison (including the P1 target below) until re-verified. See the
P3 entry in `==COMPLETED==` for the head-to-head numbers that surfaced this.

## Task Tracker

| ID | Task | Phase | Status | Blocked By | Est. Effort |
|----|------|-------|--------|------------|-------------|
| P0 | HF→ds4 GGUF converter (repacker + quantizer) | P0 | **done** | — | ~1-2 weeks |
| P1.1 | DSpark: GGUF conversion | P1 | **done** | — | 200-300 lines |
| P1.2 | DSpark: weight binding | P1 | **done** | P1.1 | ~120 lines |
| P1.3 | DSpark: GPU graph allocation | P1 | **done** | P1.1 | ~200 lines |
| P1.4 | Non-causal SWA attention mode (= A4) | P1 | **done** | — | ~150 lines |
| P1.5 | DSpark: target hidden capture | P1 | **done** | P1.2 | ~80 lines |
| P1.6 | DSpark: Markov + confidence heads | P1 | **done** | P1.2 | ~120 lines |
| P1.7 | DSpark: block spec decode cycle | P1 | **done** | P1.3, P1.4, P1.5, P1.6 | ~700 lines |
| P1.8 | DSpark: CUDA graph capture | P1 | deferred (see note) | P1.7 | ~80 lines |
| P1.9 | DSpark: server integration (`--dspark`) | P1 | **done** | P1.7 | ~60 lines |
| P1.10 | DSpark: load-aware scheduler | P1 | deferred | P1.7 | ~200 lines |
| P2a | Real packed FP8 KV cache (replace fake-quant) | P2 | **done** (gated) | — | ~1 week |
| P2b | Profiling GATE (decode split analysis) | P2 | deferred | — | ~1 day |
| P2c | FP8 weights via cuBLASLt (attn_q/kv + shared experts MXFP8) | P2 | **done** | — | — |
| P2d.ao | attn_output_a/b MXFP8 | P2 | **done** | — | — |
| P2d.lm | lm_head MXFP8 + KL/perplexity gate | P2 | planned | P2b+P3 | ~3 days |
| P2e | MXFP8 for F16 weight groups (compressor, indexer, HC) | P2 | **done** (pending quant) | — | ~1 week |
| P3 | Wire MXFP4 CUTLASS TC path (or extend prefill batching) | P3 | **done** | — | ~2 weeks |
| P4 | Oracle-driven multi-format allocation | P4 | deferred | P2/P3 formats settled | ~1 week |
| P5 | Expert pruning (REAP/RIY) | P5 | deferred | — | ~1 week |
| A1 | Fused prefill attention kernel (Flash-style tiling) | — | planned | — | ~2 weeks |
| A3 | Multi-sequence KV paging (vLLM-style) | — | planned | — | ~1 week |

**Legenda:**
- Status: **NEXT** = immediate next item; **planned** = spec'd but not started; **in_progress** = active; **done** = complete; **deferred** = lower priority, not blocking.
- A1 and A3 are attention-infrastructure items tracked here but owned by the appendix.
- MTP (P1a) skipped — straight to DSpark.
- Full writeups for **done** items live in `==COMPLETED==` at the bottom of this file.
- **NEXT:** P1 — DSpark runtime (GGUF conversion → weight binding → GPU graph → block cycle).

**Dependency graph (simplified):**
```
P1.1 → P1.2 → P1.3 ───────────────────┐
P1.4 (no deps, parallel) ─────────────┤
P1.5 (no deps, parallel) ─────────────┤──→ P1.7 → P1.8 → P1.9
P1.6 (no deps, parallel) ─────────────┘         └──→ P1.10 (deferred)
P2b ──→ P2d.lm (profile before lm_head FP8)
```
Note: P1.4 and P1.6 are pure CUDA kernel work, no weight model needed — can be
built in parallel with P1.2. P1.5 (target capture) needs P1.2 only for the
main_proj/main_norm weight names, but the capture loop around layers 40-42 is
independent — partial parallelization possible.

## Phases

### P0 — HF→ds4 GGUF converter  [FOUNDATIONAL — done]
**Done** — full writeup (source-format decisions, target layout, staging) moved to
`==COMPLETED==` at the bottom of this file. P1.1 (DSpark GGUF conversion) builds on this tooling.

### P1 — Speculative decode → DSpark serving  [HIGHEST LEVERAGE, DO FIRST]
Hits the actual bottleneck (single-stream decode, memory-bound). **Measured ceiling:**
raw decode tops out at ~25.5 t/s (FP8 path, depth0) ≈ 74% of the ~32 t/s GB10 memory-BW
ceiling for 14B active. The *only* way past ~30 is more useful tokens per memory sweep →
spec decode. Reference: Qwen-class (≈10-14B active) with DFlash-grade drafts hits **60+ t/s**
on this hardware. Target: **60+ t/s decode** via DSpark.

"DSpark" = DeepSeek's 2026 spec-decode method (`github.com/deepseek-ai/DeepSpec`, also ships
DFlash + Eagle3), NOT DGX Spark (naming clash). MTP skipped — straight to DSpark.

- **DSpark:**
  **Reference:** vLLM PR #46995 (benchislett/NVIDIA, merged 2026-07-01 into mainline vLLM),
  SGLang PR #29538 (adityakamat24, open).

  **Source checkpoint:** `deepseek-ai/DeepSeek-V4-Flash-DSpark` on HF — same 284B V4-Flash
  weights plus a pre-trained DSpark draft module (~9 GB addl) bundled as 48 safetensors shards
  (167 GB total). Downloaded at `/mnt/pve1-fast/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/`.

  **Actual checkpoint structure (discovered 2026-07-02 from HF weight_map):**

  The DSpark draft module uses `mtp.N.*` namespace (not `model.dspark_model.*` as earlier
  docs assumed).  Its layers are interleaved in the same 48 shards as the main model.

  ```
  mtp.0.main_proj          → target hidden projection (FP8, weight + scale)
  mtp.0.main_norm          → RMS norm after projection
  mtp.0.{attn,ffn,etc}     → 1st draft decoder layer (full layer: attn + MoE + HC)
  mtp.1.{attn,ffn,etc}     → 2nd draft decoder layer
  mtp.2.{attn,ffn,etc}     → 3rd draft decoder layer
  mtp.2.markov_head        → markov_w1 (vocab→256) + markov_w2 (256→vocab)
  mtp.2.confidence_head    → proj.weight (4096+256→1, linear)
  mtp.2.hc_head_*          → HC collapse (base/fn/scale, same pattern as main model)
  mtp.2.norm               → final RMS norm
  ```

  Each of the 3 draft decoder layers has its own 256 routed experts (MXFP4, `.weight` + `.scale`)
  and shared experts (FP8), same architecture as a main model layer but with **compression
  ratio = 0** (no compressor, no indexer) and **non-causal sliding-window attention**.

   - **Flow**: target layers 40-42 (V4-Flash, 43 layers, last 3) HC states → concat → `main_proj` FP8 matmul → `main_norm` →
    [decoder layer mtp.0] → [decoder layer mtp.1] → [decoder layer mtp.2] →
    Markov refine (sequential over 5 positions) → confidence score → HC collapse → lm_head.
   - **Block size**: runtime configurable (`--dspark-draft N`, default 5, range 1-16).
     The Markov head autoregressively refines each of the N positions; each step is
     just an embedding lookup + bias add, so N up to 16 adds negligible compute.
  - **Draft KV**: 3-layer raw SWA cache (no compressor state, window=128); each committed
    position's row is `kv_norm(wkv(main_x))` — the draft layer's KV projection applied to the
    projected target hidden (see RESOLVED item 2 below).
  - vLLM implements non-causal SWA by reusing SparseMLA backends with expanded topk
    (queries all include each other independent of causality) — avoids custom kernels.

  **vLLM production numbers** (8×B300, DeepSeek-V4-Pro-DSpark): ~250 tok/s at BS1, ~5 avg
  acceptance length at draft depth 7, 12-42% higher acceptance than MTP across SPEED-Bench
  categories, ~14ms E2E step time (0.6ms backbone + 0.6ms sampling + 11-13ms verify).
  Full DFlash backbone + autoregressive sampling captured in one CUDA graph.

  **Implementation (10 sub-items, ~1800-2500 lines C/CUDA):**

  1. **GGUF conversion** — convert the 48-shard HF safetensors to GGUF format for ds4's
     mmap-backed weight loader. The draft weights live under the `mtp.0.*`, `mtp.1.*`,
     `mtp.2.*` namespace in the HF checkpoint. Two approaches: (a) combined single GGUF
     (simpler, but the ~9 GB draft is always mapped), or (b) separate draft GGUF loaded
     via `--dspark` flag (same pattern as `--mtp`). Reuse `gguf-tools` encoder infra.
     The main model tensors (`layers.*`, `embed.*`, `output.*`) are already handled by the
     existing quantizer; the DSpark module adds ~4700 new tensor names under `mtp.*`.
     *Effort: ~200-300 lines.*

  2. **Weight binding** (`weights.c`) — new `ds4_dspark_weights` struct + binder. Tensors
     in GGUF format using `dspark.` prefix:
     - `dspark.main_proj.weight` / `.scale` — target hidden projection (FP8)
     - `dspark.main_norm.weight` — projection output norm (F32)
     - `dspark.0.{attn,ffn,hc_*,norm,etc}.*` — 1st decoder layer weights
     - `dspark.1.*` — 2nd decoder layer weights
     - `dspark.2.*` — 3rd decoder layer weights
     - `dspark.2.hc_head_*` / `dspark.2.markov_head.*` / `dspark.2.confidence_head.*`
     - `dspark.2.norm.weight` — final norm
     The `deepseek4-quantize.c` MTP name mapping already handles `mtp.N.*`→`mtp.N.*`
     identity mapping; for GGUF we'll use `dspark.N.*` naming to avoid confusion with
     the legacy MTP `mtp.0.*` prefix.
     *Effort: ~120 lines.*

  3. **GPU graph allocation** (`gpu_graph_alloc.c`) — extend `ds4_gpu_graph` with draft
     model tensor views: 3 decoder layers' working buffers, 3-layer draft KV raw cache
     (128 SWA window each), hidden state intermediates for projection/Markov/confidence.
     *Effort: ~200 lines.*

   4. **Draft backbone forward** — `dspark_forward_backbone()`: runs 3 decoder layers via
      the existing `gpu_graph_encode_decode_layer` pipeline, but with a **non-causal**
      attention mask. The N draft positions (1-16, configurable via `--dspark-draft`)
      form a sliding-window "sentence" where every token attends to every other. This
      requires adding a 4th attention mode to the fused kernel (alongside causal, SWA,
      batched). *Effort: ~150 lines kernel mod + orchestration.*

   5. **Target hidden capture** — capture main model's hidden at layers 40, 41, 42
      (V4-Flash, 43 layers total — last 3 layers before output). Per the reference
      (`model.py` Transformer.forward): the captured value is `h.mean(dim=2)` — the MEAN
      over the hc_mult HC copies of the layer output, NOT the raw HC buffer. Then concat
      (3×4096=12288) → `dspark.main_proj` FP8 matmul → `dspark.main_norm` RMS norm = main_x.
      The graph reuses per-layer HC buffers across the 43-layer loop, so the captures must
      be copy/reduce nodes added INSIDE the forward right after layers 40/41/42 execute — a
      post-forward read would see only layer 42's data. *Effort: ~80 lines.*

  6. **Markov + confidence heads** — two small CUDA kernels:
     - `dspark_markov_refine()`: for each of 5 positions, gather `markov_w1[prev_token]`,
       project via `markov_w2`, add to base logits, argmax. Sequential across positions,
       but each step is just an embedding lookup + bias add.
     - `dspark_confidence_score()`: concat hidden + markov embed, linear, sigmoid.
     *Effort: ~120 lines.*

  7. **Block spec decode cycle** (`session.c`) — new `ds4_session_eval_speculative_block()`,
     parallel to the existing `ds4_session_eval_speculative_argmax`. Flow per step:
      (1) capture target hidden states from layers 40-42 (last 3 of 43 for V4-Flash),
     (2) `project_main_hidden()` → 3-layer backbone → block hidden,
     (3) shared norm + lm_head → base logits for all 5 positions,
     (4) Markov refine + confidence score,
     (5) submit draft block to target verify (`gpu_graph_verify_suffix_tops`),
     (6) rejection sampling: accept longest valid prefix,
     (7) write accepted positions' draft-KV ring rows as `kv_norm(wkv(main_x))` per layer,
     (8) frontier snapshot/restore for partial accept (reuse existing `spec_frontier_*` infra).
     *Effort: ~700 lines. Crib from vLLM's `speculator.py`/`dspark_worker_v2.py`
     and SGLang's `dspark_worker_v2.py` — the logic is well-documented Python, transliterate
     to C/CUDA.*

  8. **CUDA graph capture** — deferred.  The DSpark spec cycle depends on variable
     inputs (main_x, token IDs) that change each step; CUDA graphs would need the
     `cudaGraphExecUpdate` API or device-side indirection for parameter passing.
     On the GB10, decode is memory-bandwidth-bound (~74% MBU); eager execution
     already saturates the bus.  The CUDA graph infrastructure in ds4 is limited to
     `begin_commands`/`end_commands` (no-ops on CUDA); building full graph capture
     would require a separate stream, capture begin/end calls, instantiation, and
     parameter update logic — well beyond the ~80-line estimate.  This is consistent
     with the project's own assessment: "CUDA-graph decode capture — eager wins this
     (GPU-bound) workload; antirez ships no graphs. Parked."
     *Effort: deferred.*

  9. **Server integration** (`generate.c`, `cli_main.c`) — `--dspark` flag, gating rules:
     greedy-only (temperature <= 0.0), not with SSD-streaming, conflicts with `--mtp`.
     Wire to `ds4_session_eval_speculative_block()` in the server's generation loop.
     *Effort: ~60 lines.*

  10. **Load-aware scheduler** (deferrable) — confidence-threshold based verification
      budget trim, per P1b measured numbers. The confidence head outputs per-position
      survival probability; the scheduler uses these + GPU utilization to decide how
      many of the 5 draft tokens to actually submit for verification. Useful for
      concurrency; nearly irrelevant for single-stream. *Effort: ~200 lines.*

  **GPU memory budget for DSpark draft:**
  | Component | Size |
  |-----------|------|
  | 3 decoder layer weights (FP8 attn/dense/shared, MXFP4 experts) | ~9 GB |
  | 3-layer draft KV cache (128 SWA × 3) | ~7 MiB |
  | Hidden state buffers (backbone + projection intermediates) | ~10 MiB |
  | **Total** | **~9 GB** (vs 97 GB target → ~9% overhead) |

  **RESOLVED from DeepSeek's reference implementation (2026-07-03) — the checkpoint ships
  authoritative code at `<snapshot>/inference/model.py`; both former open questions are
  settled by it, not by the vLLM port:**
  1. **Draft-position inputs / noise token** (`model.py` `forward_embed`, ~line 851):
     `draft_input_ids = full([B, block_size], noise_token_id); draft_input_ids[:, 0] =
     input_ids` — position 0 gets the just-sampled token, positions 1..N-1 get the noise
     token (`dspark_noise_token_id: 128799`), all embedded through the SHARED main-model
     embedding table, then HC-expanded (`unsqueeze(2).repeat(1,1,hc_mult,1)`). The draft
     also shares the main lm_head. P1.1 must carry `dspark.noise_token_id` in GGUF metadata;
     P1.7 step (2) must build this input block.
  2. **Draft KV seeding** (`DSparkAttention.forward`, ~lines 752-795): the persistent draft
     KV row for each committed position is `kv_norm(wkv(main_x))` — the draft layer's own KV
     projection applied to the PROJECTED TARGET hidden (`main_x` from P1.5), one ring row
     written per step (`kv_cache[:, start_pos % win] = main_kv`). The N speculative
     positions' KV is computed fresh each cycle and only transiently concatenated
     (`cat([cache, kv])`) — never persisted. So: no row-copy of speculative KV (decision 6
     as originally written is wrong), and no separate `kv_from_hidden` module — just one
     small per-layer matvec per accepted token. During prefill (`start_pos == 0`) the draft
     layers do KV-fill only from main_x (no FFN work), populating the last `win` positions.
     Two engine notes from the same code: `attn_sink` IS used in the draft sparse attention,
     and the attention output applies an INVERSE rotary on its rope tail
     (`apply_rotary_emb(o[..., -rd:], freqs_cis, True)`) before the grouped wo_a/wo_b.

  **Resolved design decisions (2026-07-03):**
   1. **GGUF format**: separate file, loaded via `--dspark PATH`. Same pattern as `--mtp`
      (already implemented in P1.1).
   2. **Non-causal attention**: Add a `non_causal` flag to `attention_decode_mixed_kernel`.
      For draft layers (compression-ratio 0, raw KV only): change the hi bound
      from `hi = qpos < raw_last_pos ? qpos : raw_last_pos` to `hi = raw_last_pos` (all
      window positions visible). The compressed paths are never hit. NOTE: draft layers DO
      have `attn_sinks` (`dspark.N.attn_sinks.weight` is in the P1.2 binding list) — sink
      handling must be preserved in the non-causal mode. Orchestration requirement: all N
      draft positions' KV must be written to the raw cache BEFORE the attention pass so the
      block can attend to itself (two-phase: KV write for all N, then attend).
      *~150 lines, no new kernel needed.*
   3. **Confidence threshold**: Tunable runtime flag `--dspark-confidence FLOAT` (default 0.5)
      + `DS4_DSPARK_CONFIDENCE` env var override.
   4. **Target layer indices**: V4-Flash = 43 layers → layers 40, 41, 42 (last 3). Stored
      in GGUF metadata as `dspark.target_layer_ids` (int array of 3 values); at binding time,
      fall back to `[n_layer - 3, n_layer - 2, n_layer - 1]` if missing.
   5. **Draft block size**: Runtime configurable via `--dspark-draft N` (1-16, default 5)
      + `DS4_DSPARK_DRAFT_TOKENS` env var. Flash was trained at block size 5
      (`dspark_block_size: 5` in config.json) — store it in GGUF metadata
      (`dspark.block_size`) rather than hardcoding, default to it, and warn when N exceeds
      it (positions past the trained block are unvalidated; expect acceptance falloff).
      Graph capture and buffers must be sized for the max (16), not the current N.
   6. **Draft KV seeding** (REVISED per reference implementation, see RESOLVED item 2
      above): per committed token, compute `kv_norm(wkv(main_x))` in each draft layer and
      write one ring row at `pos % 128`. NOT a copy of speculative-run KV rows — those are
      transient. Cost: one small matvec per draft layer per accepted token, consuming the
      `main_x` P1.5 already produces.
   7. **A3 (multi-sequence KV paging) not required** for DSpark. The draft uses a dedicated
      3-layer raw KV cache (window=128), separate from the main model's KV. No paging needed
      because the draft cache is small (~256 KB per layer: 128 rows x 2048 B) and flushed each spec cycle.

  **P1 engine options** (additions to `ds4_engine_options` in `ds4.h`):
  | Field | Type | CLI flag | Default |
  |-------|------|----------|---------|
  | `dspark_path` | `const char *` | `--dspark PATH` | NULL |
  | `dspark_draft_tokens` | `int` | `--dspark-draft N` | 5 |
  | `dspark_confidence` | `float` | `--dspark-confidence FLOAT` | 0.5 |

  **Completion criteria per sub-item:**
  
  | ID | Completion Criteria | Verification |
  |----|-------------------|--------------|
  | P1.1 | 78 tensors, correct shapes/types; loadable via `--dspark` | `ds4 --dspark gguf/dspark.gguf` exits clean |
  | P1.2 | All `ds4_dspark_weights` fields non-NULL; layout validation passes type+shape | Startup logs: "DSpark support model loaded (draft=5)" |
  | P1.3 | 3 draft KV caches (window=128), projection/Markov/confidence buffers allocated; free symmetric | Engine logs allocation; valgrind/asan clean |
  | P1.4 | 5 draft positions attend to all window positions equally; existing causal path unchanged; compressed path never hit (ratio=0) | Black-box: 5 identical KV rows → 5 equal outputs; `--logprob-vectors` passes |
   | P1.5 | `after_ffn_hc` at layers 40/41/42 captured after decode; `main_proj`+`main_norm` produce finite output | GPU graph diagnostics report target HC norms |
  | P1.6 | 5 tokens refined from base logits; 5 survival probabilities ∈ [0,1]; both deterministic | CPU reference comparison vs GPU output |
  | P1.7 | End-to-end spec cycle: produce N drafts, accept prefix ≥1, commit. **Greedy parity**: 0 top-1 mismatches vs non-spec baseline over 1000 tokens. **Acceptance**: avg ≥ 1.5. **Ppl**: within 0.01 of baseline. KV save/restore works mid-generation. | `ds4-eval` greedy comparison; `--logprob-vectors`; checkpoint save/restore |
  | P1.8 | Full DSpark cycle captured in one graph; graph output matches non-graph | `--gpu-graph-test`/`--gpu-graph-full-test` pass |
  | P1.9 | `ds4 --dspark PATH` serves; gating rules enforced (no `--mtp`, no `--ssd-streaming`, greedy-only) | CLI error messages; end-to-end generation |
  | P1.10 | Not gating P1 ship | — |

  **Test plan:**

  | Test | Item | Prereq | Method |
  |------|------|--------|--------|
  | `tests/cuda_long_context_smoke` — dspark_attn | P1.4 | GPU only | Feed 5 identical KV rows, verify all 5 outputs equal |
  | `tests/cuda_long_context_smoke` — dspark_heads | P1.6 | GPU only | Fixed Markov/confidence input vs CPU reference output |
  | `ds4_test --dspark-self-test` | P1.7 | model + dspark GGUF | Same prompt with/without `--dspark`; report match rate, acceptance length, speedup |
  | `--logprob-vectors` (pinned env) | P1.7 | model + dspark GGUF | `DS4_CUDA_PREFILL_CHUNK=2048`; compare against official API vectors |
  | KV checkpoint integrity | P1.7 | model + dspark GGUF | Spec N steps, save checkpoint, restore, continue — verify token continuity |
  | CLI gating matrix | P1.9 | dspark GGUF | All flag combos: `--dspark+--mtp`, `--dspark+--ssd-streaming`, `--dspark+--temperature 0.1` — all rejected |
  | Memory leak | P1.3 | dspark GGUF | `ds4 --dspark PATH` in + out; valgrind or `DS4_ALLOC_GUARD` clean |

  **Build order (with tests integrated):**

  ```text
  Step 1: P1.4 (non-causal SWA) + P1.6 (Markov/confidence) — CUDA only, no model needed
          → add CUDA smoke tests to tests/cuda_long_context_smoke
  Step 2: P1.2 (weight binding) — enables loading dspark GGUF
          → verify with ds4 --dspark PATH
  Step 3: P1.5 (target capture) — needs P1.2 for weight names
          → instrument target HC norms in gpu_decode.c
  Step 4: P1.3 (graph allocation) — needs P1.2 for tensor sizes
          → verify alloc/free symmetric
  Step 5: P1.7 (block spec cycle) — needs P1.2-1.6
          → add --dspark-self-test to ds4_test
  Step 6: P1.8 (CUDA graph capture) — needs P1.7
          → verify graph passes --gpu-graph-test
  Step 7: P1.9 (server integration) — needs P1.7
          → CLI gating test matrix
  ```

### P2 — FP8 (the decode/quality track)  [MEDIUM]
FP8 is the *decode* play: everything read every token (KV, attn proj, always-on shared
experts, dense FFN) can go FP8 — lossless vs the FP8 source AND on the idle FP8 cores.
Dependency reality: **all NVIDIA first-party, no vLLM.** The *GEMM* is cuBLASLt (already
linked); the *storage encoder* is ours in quants.c (FP8 = a near-identity repack since the
source attn/dense is already E4M3).

- **2a FP8-KV — ALREADY IN antirez, enable + measure first.** `fp8_kv_quantize_kernel`,
  `ds4_gpu_dsv4_fp8_kv_quantize_tensor`, kvstore quant-aware checkpoints all exist. MLA
  already compresses KV to a latent; FP8-ing that latent ~halves KV bandwidth/memory →
  decode win at depth + more context room. Lowest effort, do right after the oracle quant.
  **CORRECTION (2026-07-02 audit):** `fp8_kv_quantize_kernel` (`ds4_cuda_norm_kv.cu:358-380`)
  only fake-quants — it rounds nope-dim KV values to the E4M3 grid, then writes the result
  back as a 32-bit float. Raw-window storage is `DS4_N_HEAD_DIM * sizeof(float)` = 2048
  bytes/token/layer, identical to a plain f32 store. We are paying the E4M3 rounding cost
  with ZERO bandwidth win banked today. TODO: actually repack to real ~1-byte FP8 storage
  (both raw-window and compressed-cache writes) to realize the memory win this item claims.
  Biggest payoff is at depth (compressed-KV reads scale with context); the fixed 128-token
  raw window itself is small (~11 MiB/decode-step across 43 layers) — prioritize the
  compressed-cache repack over the raw window.
- **2b Profiling GATE** — deferred (2026-07-04).  2c is already done; the profiling
  is only needed to gate 2d.lm (lm_head FP8).  The lm_head is 0.56 GB of the ~8 GB/token
  active-weight estimate (~7%), likely noise in the decode bandwidth budget.
  Run `ds4-bench` or nsys when a working model is available to confirm.
- **2c FP8 weights via cuBLASLt — done.** See `==COMPLETED==`.
- **2d Eliminate ALL remaining Q8 (end-state goal: zero Q8 in the model).** After 2c, the
  only Q8 left is **3.63 GB**: `attn_output_a`+`attn_output_b` (3.07 GB) and `output.weight`/
  lm_head (0.56 GB). These do NOT ride the generic FP8-aware matmul — each goes through a
  dedicated kernel that hard-asserts `w->type==8`:
    - `attn_output_a` → `matmul_q8_0_grouped_batch` (head-grouped)
    - `attn_output_b` → `matmul_q8_0_batch`
    - lm_head → `matvec_q8_0` (prefill) + `matvec_q8_0_decode_scratch` (decode)
  Path per tensor (same pattern as 2c): add an FP8 branch to each kernel (reuse
  `cuda_matmul_fp8_mx_tensor_labeled` / `mxfp8_mmvq_kernel`); relax `tensor_expect_layout`
  (ds4.c:3619-20/3689-90, and :3598 for output); add to the requant FP8 override; validate.
  CAVEAT — **lm_head is logit-sensitive** (token selection, same class as the F16 router/NSA
  indexer): take it to FP8 only behind a KL/perplexity gate, else keep it at the top tier.
  Payoff is **prefill tensor-core unification only** (8.5→8.25 bpw ≈ 0.45 GB; decode is
  memory-bound so no decode gain). SEQUENCING (user, 2026-06-29): **do P3 MXFP4-experts FIRST**
  (97% of the model = the real lever); 2d is the cleanup pass to reach a fully Q8-free model.
  **2d.ao (attn_output_a/b MXFP8) is done** — see `==COMPLETED==`. Remaining Q8/BF16 item is
  lm_head only. **The KL gate now exists (2026-07-02):** `ds4 --kl-file CALIB --kl-ref-dump
  REF.bin` / `--kl-score REF.bin` measures full-vocab mean KL (+ block-bootstrap stderr)
  between any two model configurations on calibration text — built for the P4 measured-KL
  allocator, directly usable as this gate. To gate lm_head: dump the reference with BF16
  lm_head, quantize an FP8-lm_head variant, `--kl-score`, and require the delta to be
  indistinguishable from zero (within ~2 stderr) before shipping it.
- **2e Extend MXFP8 to the remaining F16 GEMM weight groups (2026-07-02 audit, NEW).**
  The KV compressor (`attn_compressor_{ape,kv,gate}`, 41/43 layers), the DSA indexer
  (`indexer_attn_q_b`/`indexer_proj`/`indexer_compressor_*`, 21/43 ratio-4 layers), and
  hyper-connections (`hc_attn_fn`/`hc_ffn_fn`/`output_hc_fn`, all 43 layers + final) are still
  `DS4_TENSOR_F16`. Shape-derived (not measured) per-token decode bytes at F16 vs MXFP8:
  | group | F16 | MXFP8 | saved |
  |---|---:|---:|---:|
  | KV compressor | ~499 MiB | ~260 MiB | ~240 MiB |
  | DSA indexer | ~431 MiB | ~222 MiB | ~209 MiB |
  | Hyper-connections | ~65 MiB | ~33 MiB | ~31 MiB |
  Combined ~480 MiB/token saved, roughly 5-6% of the ~8 GB/token active-weight estimate
  implied by the 32 t/s bandwidth ceiling above. Same mechanical migration as 2c, no new
  kernel infra needed. CAVEAT: the indexer feeds DSA top-k selection — a quantization
  regression there means wrong tokens attended, not just numeric drift. Validate by comparing
  top-k selection sets, not just logit RMS, separately from the compressor/HC quality bar.

### P3 — MXFP4 experts on FP4 tensor cores  [the serving play]
KEY FACT: the source routed experts are **already MXFP4** — `dequant_fp4_weight` decodes
E2M1 values `{0,±0.5,±1,±1.5,±2,±3,±4,±6}` + `F8_E8M0` scale per **32-block**. So storing
experts as **MXFP4 (E2M1 + E8M0/32, ~4.25 bpw) is a byte-lossless match to the source** AND
runs on the 5th-gen FP4 cores (2x FP8 / 4x BF16). It is strictly **better than our current
Q4_K** rich-experts: Q4_K's uniform integer levels can't represent `{0.5,1.5,3,6}` exactly
(lossy), is bigger (4.5 bpw), and runs on int/mmq not FP4 cores.

Nearly free given P2's MXFP8 work: MXFP4 uses the **same `VEC32_UE8M0` scale mode and the
same verified 128x4 scale-swizzle** — only the data type changes (`CUDA_R_4F_E2M1` vs
`CUDA_R_8F_E4M3`). New pieces: a 2-bit-mantissa **E2M1 codec** (sibling of our E4M3 codec)
and the grouped-MoE FP4 GEMM.

**DECODE-NEUTRAL** (memory-bound; ~4.25 bpw read either way). Wins **prefill + concurrency**.

Dependency reality: **the math is NVIDIA; vLLM is optional and only for MoE fusion.**
- The hard part is the **grouped/batched MoE GEMM** — experts go through `routed_moe_launch`,
  not dense cuBLASLt. Options: (a) batched-dense MXFP4 at prefill via cuBLASLt; (b) CUTLASS
  grouped FP4 collectives; (c) lift vLLM's fused FP4-MoE kernel (a CUTLASS wrapper, Apache-2.0)
  via a C bridge. Do NOT hand-write tcgen05 FP4 MMA.
- Risk: CUTLASS/cuBLASLt grouped FP4 on sm_121 is bleeding edge (toolchain friction).
- **STATUS (2026-07-02 audit): option (b) is partially built and NOT wired in.**
  `ds4_cutlass_expert_ffn` (`ds4_mxfp4_cutlass.cu`) implements a CUTLASS `mx_float4_t` grouped
  GEMM, compiles and links into every binary — but has zero call sites outside its own
  `DS4_MXFP4_STANDALONE` test `main()`. It never runs. The real dispatch,
  `routed_moe_launch` (`ds4_cuda_moe.cu:2284`), routes MXFP4 weights to per-token `qwarp32`
  kernels (`moe_gate_up_mid_mxfp4_qwarp32_kernel`, `moe_down_mxfp4_sum6_qwarp32_kernel`) — no
  tensor cores, decode-shaped and prefill-shaped alike.
- **REGRESSION RISK:** `use_sorted_pairs = n_tokens > 1 && !mxfp4_path` (`ds4_cuda_moe.cu:2391`)
  explicitly disables the batched/tiled prefill optimization whenever expert weights are
  MXFP4. IQ2_XXS/Q2_K experts get that batching; MXFP4 experts don't. **MXFP4 prefill may
  currently be SLOWER than IQ2_XXS/Q2_K prefill** — measure before assuming MXFP4 is a strict
  upgrade at prefill/concurrency until the CUTLASS path (or equivalent batching) is wired in.
  TODO: either finish wiring `ds4_cutlass_expert_ffn` into `routed_moe_launch`, or extend
  `use_sorted_pairs`-style batching to the MXFP4 path before shipping it as the default
  expert format.
- **P3 is done (wired 2026-07-02)** — CUTLASS MXFP4 TC path is live, functionally correct, a
  clean prefill win, decode-neutral. Full writeup + measured numbers moved to `==COMPLETED==`.
  **OPEN QUESTION (bigger than this item, still unresolved):** both baseline and CUTLASS decode sit at ~13 t/s
  here, roughly half the ~25.5 t/s this file's earlier "Measured (2026-06-29, oracle-fp8wh-99gb...)"
  line up top claims. That number's source file doesn't exist on disk anymore and was never
  reproduced against the current build — don't trust it until re-verified (clock-pinning,
  `--power` state, or an actual regression since 2026-06-29 are all still open explanations).
  Chase this before optimizing further against it.

**NVFP4** (E2M1 + E4M3 per-16 microscale + FP32 global, ~4.5 bpw) stays in our pocket for
quantizing *from high precision* (bf16) — e.g. if we ever push attn/dense below FP8 — where
its finer two-level scaling earns its keep. For the experts it adds nothing: the source is
already E8M0/32, so NVFP4's finer scale can't recover detail that isn't there.

### P4 — Oracle-driven multi-format allocation across the full menu
Extend `--mse-probe` + prisma to allocate over ALL tensors (experts + dense/attn) and
the curated format menu below, picking by imatrix-weighted real error per (tensor,format).
**AURA-era update (2026-07-02, from the RobTand/prismaquant comparison):** for routed
experts, the allocator now prefers **measured end-to-end KL** over any proxy — upstream's
own finding is that gradient/imatrix proxies are structurally blind to router flips, and
their MoE ship path is empirical per-serving-unit KL. Built: `--expert-overlay` (swap one
layer's expert tensors from a donor GGUF at load), `--kl-file`/`--kl-ref-dump`/`--kl-score`
(full-vocab KL vs a reference logit dump, with block-bootstrap stderr),
`prisma/measure_layer_kl.py` (43-run harness → `kl.json`), exact 0/1 DP knapsack +
optional `--ucb-z` uncertainty charge in `prisma_alloc.py`, and the previously-missing
`--format-map` manifest consumer in the quantizer. When P4 extends to dense/attn tensors,
the same measured-KL machinery applies (dense tensors are smooth, so upstream's AURA
gradient probe would also work there — but it needs backprop we can't run at 284B locally;
measured KL per candidate is the hardware-honest equivalent).

### P5 — Expert pruning (REAP / RIY) — workload-specific capacity  [HIGH VALUE]
REAP = Router-weighted Expert Activation Pruning (Cerebras): drop underutilized MoE
experts. RIY ("Reap It Yourself", github.com/flash7777/vllm @riy, `riy_live.py` TUI +
`--riy-expert-profile`) profiles YOUR workload -> JSON of dead experts -> zeroes them at
load, reversible via HTTP API. Reported: 5-20% prune to make room/fit; ~70% stable on
some models, ~90% breaks. PRINCIPLE: profile your own workload, not C4/GSM8K.
ref: https://forums.developer.nvidia.com/t/why-you-should-reap-it-yourself-live-moe-expert-pruning-in-vllm/364098

WHY IT MATTERS: a SECOND memory lever, orthogonal to quantization, and it directly serves
the "preserve DeepSeek's FP4, don't crush to IQ2" goal. Pruning a dead expert frees its
FULL footprint at ~zero quality cost (it's rarely selected) -- STRICTLY better than
IQ2-crushing it. So pruning is effectively the **0-bit tier at the bottom of the
allocation ladder**: the cheapest option for the least-active experts, below IQ2. Prune
the cold experts -> reclaim memory -> keep the hot experts at native FP4/NVFP4.

UNIFYING INSIGHT (the big one): the router-weighted expert-activation profile is the SAME
signal we want for everything. One per-expert ranking ("how much does this matter for OUR
workload") drives all three memory levers at once:
  (a) PRUNE the dead experts (0-bit),
  (b) QUANTIZE the rest by rank (FP4/NVFP4 hot, Q-quant warm, IQ2 marginal),
  (c) SSD-STREAM priority (hot in RAM, cold on disk).
=> Fold expert-activation profiling into the Prisma oracle as the master sensitivity
signal, replacing/augmenting the imatrix proxy. Profile on the real coding/agentic
workload, not generic text.

PORT: ds4 (not vLLM), so we bring the CONCEPT, not the code -- a profiler (router-weight
histogram per expert) + a skip/zero mask on ds4's routed-MoE dispatch. antirez's SSD
streaming expert cache is the natural home (it already gates experts in/out of RAM by
activity). CAVEAT: pruning trades general capability for fit -- safe at 5-20%, aggressive
only for a narrow/fixed workload profile.

## Appendix: Attention architecture (2026-07-02 audit)

ds4 attention is **Multi-Query Attention (MQA)** — 64/128 query heads, **1 KV head**, head dim
512 — entirely custom CUDA, no FlashAttention or FlashInfer dependency. The computation is
standard scaled dot-product (`score = dot(Q,K) / sqrt(512)`), online softmax (running max/sum),
V accumulation, all **FP32**. Three-tier KV cache per layer: raw ring buffer (128 SWA window),
attention-compressed (ratio 4 or 128 via learned MLP), and indexer-compressed (ratio 4 only,
128-dim via top-K selection).

**Identified improvement opportunities, highest ROI first:**

1. **Fused prefill attention kernel (replace cuBLAS SGEMM path).** Today prefill does
   `QK^T → HBM → softmax → HBM → PV → HBM` — three HBM round trips via cuBLAS strided-batched
   SGEMM. A single fused kernel with FlashAttention-style tiling (partial accumulators in SRAM,
   one write) would cut prefill attention HBM traffic ~3×. This is the single biggest latency
   lever for long-prompt prefill (e.g. 32K tokens). *Reference: the decode path already does
   fused online softmax; extend the pattern to batched prefill.*

2. **Real packed FP8 KV cache (vs current fake-quant).** The raw cache stores `float[512]` per
   entry — 2 KB/token/layer. The existing `fp8_kv_quantize_kernel` rounds to E4M3 grid but
   writes back as f32 (zero bandwidth win). Repacking to ~1-byte FP8 halves KV bandwidth and
   memory — the biggest decode win at context depth. Already flagged in P2a as deferred work;
   prioritize the compressed-cache repack over the raw window (the 128-token window is only
   ~11 MiB across 43 layers). *Also: remove the hard-coded `DS4_CUDA_SCORE_BUF_CAP = 8192`
   compressed-row limit that truncates sessions above 8K compressed KV.*

3. **Multi-sequence KV paging for speculative decode.** The ring buffer is a single contiguous
   slab. A page-table-based KV cache (vLLM-style) would let multiple candidate sequences
   share memory efficiently during verification. **No longer a P1 prerequisite** (P1 resolved
   decision 7, 2026-07-03): DSpark's draft uses its own small dedicated 3-layer SWA cache
   (~256 KB/layer, flushed per spec cycle), and the target side reuses the existing
   `spec_frontier_*` snapshot/restore, same as MTP today. Paging remains a concurrency-era
   improvement, not a spec-decode blocker.

4. **Non-causal SWA mode (for DSpark draft backbone).** All current kernels assume causal
   masking. DSpark's 3-layer draft backbone needs all 5 draft positions to attend to each
   other in a single forward pass. Add a 4th attention mode alongside causal, SWA, and
   batched — the new mode uses the same kernel pipeline but with a non-causal mask (every
   position visible to every other, bounded by the SWA window). *P1b item 4, ~200 lines.*

5. **Blackwell TMA + WGMMA for decode attention.** The CUTLASS submodule ships Blackwell FMHA
   examples (`examples/77_blackwell_fmha/`) using TMA for async tensor loads and WGMMA for
   warp-group matmul. All ds4 attention kernels are hand-written warp-level — porting the hot
   decode kernel to TMA+WGMMA would overlap KV loads with compute and improve shared memory
   utilization. Lower ROI than items 1-4 due to current memory-bandwidth-bound decode (TMA
   helps compute-bound workloads more). *Defer until after P1b.*

**Why ds4 doesn't need FlashAttention or FlashInfer:**
- **MQA simplification:** With 1 KV head and SWA=128, the KV set is always bounded (256 raw
  max + 8192 compressed). Full-scan fits in registers with no HBM intermediate writes —
  FlashAttention's tiling overhead isn't justified.
- **Specialized beats general:** ds4's kernels are hard-coded for exactly one shape set
  (Flash/Pro), one cache topology, one dispatch pattern. No JIT, no template dispatch, no
  branching — every kernel call is a straight-line CUDA graph node for the GB10 Tensor Cores.
- **Trade-off:** This specialization blocks FlashInfer's features (paged KV, tree attention,
  ALiBi, MLA) and makes non-causal or multi-sequence attention harder to add. The P1b DSpark
  build pays this tax.

## Curated format menu (every format must earn its kernel)
| bpw  | format     | hw path                    | role |
|------|------------|----------------------------|------|
| 0    | PRUNE      | n/a (skip)                 | dead experts (P5, not yet built) |
| 2.06 | IQ2_XXS    | int (native)               | bulk experts: gate/up in IQ2 combo |
| 2.6  | Q2_K       | int (native)               | bulk experts: down in IQ2 combo |
| 4.25 | MXFP4      | warp Q8_K dot (no TC yet)  | rich experts (type-39 active; CUTLASS type-40 TC path is dead code) |
| ~8.25| FP8_E4M3   | FP8 TC / cuBLASLt MX       | attention/dense/shared experts (primary workhorse) |
| 16   | BF16       | cuBLASLt / custom          | lm_head only (precision tier; FP8 pending KL gate) |
Adopted: FP8+MXFP4 in P2/P3, expert pruning in P5. Rejected: INT4/AutoRound. Removed from table
(rejected at load): Q3_K (never implemented), Q4_K (lossy vs MXFP4 source, removed),
Q8_0 (replaced by MXFP8 for weights; Q8_K retained only as activation quant inside MoE kernels).
NVFP4 dropped — no source weights are BF16, so the format has no application here.

## Rejected / parked
- **PR#266 rows/block 8→16 occupancy opt** — REJECTED (2026-06-30). The upstream PR is *buggy* (verbatim it leaves half the rows unwritten + mixes the rest — proven in a standalone micro-test: 1024/2048 wrong). A *correct* reimplementation (16 threads/row + 16-wide segmented reduction; MoE rr<8/*256/keep-rr*32) is coherent but **perf-neutral on GB10**: decode 26.6 vs 26.2, prefill 437 vs 434 (within noise). Decode is bandwidth-bound (~74% MBU), not occupancy-bound, so the knob can't help. Don't revisit; the single-stream lever is spec decode (P1).
- **CUDA-graph decode capture** — eager wins this (GPU-bound) workload; antirez ships no graphs. Parked.
- **q5/q6 on EXPERTS** — FP4 source ceiling makes them wasteful. (q5/q6 encoders kept for a future >FP4 use.)
- **Entrpi mmq graft** — benchmark-gated vs antirez native Q4 MoE; likely unnecessary given P2/P3.

## ==COMPLETED==

- Clean fork off antirez/main (native GB10, batched Q4 MoE, mixed-precision serving, SSD streaming).
- Prisma quantizer toolchain merged (encoders iq2/q2/q3/q4/q5/q6 + dequants + `--mse-probe` + allocator).
- Oracle quant (oracle-99gb, Q8 non-experts) validated at parity with hand baseline.

### P0 — HF→ds4 GGUF converter
Today we re-quantize antirez's pre-made `template.gguf`, inheriting its conversion choices
(it **dropped the MTP head**; we can't change per-tensor precision at conversion time). Owning
the HF→ds4 conversion gives control of speed AND quality: set precision per tensor, include
MTP, emit expert quants matched to the production decode path. **Source formats** per tensor group:
- **Dense/attention:** native FP8 E4M3 (repack only, lossless → FP8_E4M3)
- **Routed experts:** native MXFP4 E2M1+E8M0/32 (must quantize down to IQ2_XXS/Q2_K for the
  primary decode path; can repack lossless → FP4_E2M1 for the non-TC MXFP4 path)
- **Compressor/indexer/HC:** F16 (can repack → MXFP8 per P2e)
- **lm_head:** BF16 or FP8 (repack only; FP8 pending KL gate)

So P0 is both a repacker (for FP8, MXFP4, F16 groups that stay at their source precision)
and a **quantizer** (for the MXFP4→IQ2_XXS/Q2_K expert path that powers decode today).
- **Target layout:** match `template.gguf`'s exact block layout (reverse-engineer by diffing
  one known tensor HF-vs-template); remap to `blk.N.*`/`mtp.0.*`, stack experts into
  `ffn_{gate,up,down}_exps`, write `deepseek4.*` metadata.
- **Stage:** (1) MTP draft layer only → validates the codec + yields an FP8/MXFP4 draft (higher
  accept → more spec-decode gain, pairs with P1); (2) full main model incl `mtp.0.*` → feeds P3
  (MXFP4 experts) and P2d (zero-Q8). Removes the antirez-`template.gguf` dependency entirely.

**(2026-07-02) P0 quantizer upgraded** — added MXFP4 quantize/dequant output, MTP tensor
name mapping + expert handling in `deepseek4-quantize.c`, stripped unused types. Can produce
IQ2_XXS/Q2_K/FP8_E4M3/MXFP4 GGUFs from HF safetensors.

**Stage 2 RESOLVED (2026-07-02) — genuinely, on the second attempt.** First attempt (same day)
was circular and got called out as such: it derived a zero-data template by stripping tensor
data out of `oracle-zeroq8-99gb.gguf`. That doesn't remove the antirez dependency, it just
launders it one hop — `oracle-zeroq8-99gb.gguf`'s own tensor list was itself originally produced
using antirez's template, so copying it forward reproduces his layout choices exactly, just from
a different file.

The real fix, `gguf-tools/build_main_template.py`, derives the *entire* tensor manifest and all
`deepseek4.*`/`general.*` metadata directly from the HF checkpoint itself, with zero reference to
any pre-existing ds4 GGUF (ours or antirez's):
- non-expert tensor shapes/dtypes: read from the real safetensors shard headers
- per-layer heterogeneity (which of the 43 layers have an indexer, a compressor, the
  `tid2eid` hash table): read from which HF tensors actually exist per layer — confirmed by
  direct inspection to exactly match `config.json`'s `compress_ratios` array (indexer only on
  the 21 layers where `compress_ratios[L]==4`; compressor on the 41 layers where it's 4 or 128;
  hash tables on the 3 layers found via `num_hash_layers`) — not hardcoded index lists
- routed-expert combined shapes: derived from `config.json` (`hidden_size`,
  `moe_intermediate_size`, `n_routed_experts`), same `[in,out,R]` stacking convention
  `build_dspark_template.py` already uses for the DSpark draft
- architecture metadata (rope params, hyper-connection config, sliding window, etc.): read
  directly from `config.json`; sampling defaults from `generation_config.json`
- the HF↔ds4 tensor-name mapping: lifted verbatim from `deepseek4-quantize.c`'s own
  `top_map[]`/`layer_map[]` tables — our own already-written, already-working mapping, not a
  copy of any GGUF's structure
- **scoped out, not silently skipped:** `tokenizer.ggml.*`/`chat_template` — that's DeepSeek's
  own published tokenizer data, not an antirez architectural choice, so it isn't part of this
  claim, but a correct raw-`tokenizer.json`→GGUF conversion (byte-level BPE vocab/merges/special
  tokens) is its own nontrivial task. Until that's built, `--splice-tokenizer-from` copies those
  specific KV entries from an existing valid GGUF as a stopgap.

**Verified end-to-end** against the real `DeepSeek-V4-Flash-DSpark` HF checkpoint (167 GB, 48
shards): `--dry-run` planned all 1328 tensors correctly. `--compare-tensor` byte-matched
`oracle-zeroq8-99gb.gguf` exactly (identical FNV1a64) across every non-expert category — F32
norms, FP8_E4M3 dense/attn, F16 compressor/indexer/HC/router, BF16 `output.weight`, the I32
hash-layer LUT — and, once the matching imatrix file was supplied, the lossy IQ2_XXS/Q2_K routed
experts too, across multiple layers.

**One caveat found and run to ground, not hidden:** 4 of the 43 layers (37, 39, 41, 42) use a
promoted MXFP4 format for at least `ffn_down_exps` in the reference file, instead of the uniform
IQ2_XXS/Q2_K everywhere else — a quantize-time *precision-allocation* choice (which layers get
richer experts), not a template/architecture-layout question. Once given the matching
`--tensor-type blk.N.ffn_down_exps.weight=mxfp4` override, sizes matched exactly, but the byte
*content* still didn't. Control test: regenerating the same tensor from the old, circular,
oracle-derived template produces the **identical** (still-mismatching) hash as the new
HF-derived template — proving the discrepancy predates and is independent of either template;
it's specific to whatever imatrix state or quantizer version originally produced that one
historical file. Not a stage-2 defect.

### P2c — FP8 weights via cuBLASLt
attn_q/kv + shared experts Q8_0→MXFP8 via cuBLASLt MX prefill + mmvq decode; validated
coherent, 25.5 t/s decode / 431 t/s prefill @d0.

### P2d.ao — attn_output_a/b MXFP8
`attn_output_a`/`attn_output_b` are now on MXFP8 — fused MXFP8×MXFP8 kernels for both decode
and prefill (`cuda_attention_output_a_mx_gemm`, `ds4_cuda_matmul.cu:655`), currently the
uncommitted rework being validated. Confirmed the `rank%128==0, group_dim%32==0` shape gate on
`attn_output_a`'s MX-GEMM path is satisfied by the real Flash/Pro shapes (`rank=1024,
group_dim=4096`) — the fast path is reachable, not a silent fallback.

### Prefill QK-norm+RoPE fusion (2026-07-02)
`gpu_prefill.c` now uses the same fused `ds4_gpu_head_rms_norm_rope_tail_tensor` kernel decode
already used. Verified bit-identical against reference via `--logprob-vectors`,
`--local-golden-vectors`, `--tensor-equivalence`.

### 2026-07-02 performance audit
Full active-path table audited against source. Found P3 CUTLASS MXFP4 path is dead code (never
called) and disables prefill batching for MXFP4 experts; KV "FP8" (2a) is fake-quant only with
zero bandwidth win banked; confirmed the `attn_output_a` MX-GEMM shape gate is reachable for
real model shapes. This audit is what produced the P3/2d.ao findings above and the new 2e item
and Attention-appendix analysis.

### P3 — MXFP4 CUTLASS TC path wired
The CUTLASS path is live, functionally correct, and a clean prefill win with no measured decode
cost. New tensor type `DS4_TENSOR_CUTLASS_MXFP4 = 40` (expert-major ColumnMajor E2M1 data +
swizzled E8M0 SF per expert, `cutlass_mxfp4_expert_layout()` in `gguf.c`); accepted through
`weights.c` validation as a third routed-expert combo; a new `routed_moe_launch_cutlass()` in
`ds4_cuda_moe.cu` sorts tokens by expert (reusing the existing count/prefix/scatter kernels),
gathers each active expert's rows, calls a rewritten `ds4_cutlass_expert_ffn_scratch()`
(caller-provided scratch via `cuda_tmp_alloc`, no more per-call
cudaMalloc/cudaFree/cudaDeviceSynchronize — that was 10 allocations per call in the original
standalone-test code), and scatters results back for the existing `moe_sum_kernel` to reduce.
Standalone CUTLASS kernel test still passes bit-exact (`max_rel=0.00000`) after the
scratch-buffer refactor. Loaded and served `oracle-cutlass-mxfp4-99gb.gguf` end-to-end;
`--logprob-vectors`, `--tensor-equivalence`, `--short-prefill-ratio4`, `--local-golden-vectors`
all pass, zero top-1 mismatches.

**Measured on this GB10** (`ds4-bench`, `speed-bench/promessi_sposi.txt`, ctx 2048-8192),
head-to-head against `oracle-zeroq8-99gb.gguf` on the identical build/hardware/methodology
(correcting an earlier draft of this entry that compared against the old, no-longer-present
`oracle-fp8wh-99gb` log line instead of measuring fresh — do not trust historical numbers from
a file that isn't on disk):

| | prefill t/s | decode t/s |
|---|---|---|
| baseline (zeroq8) | 167 / 163 / 161 / 161 | 12.98 / 13.14 / 13.10 / 13.04 |
| CUTLASS | 445 / 414 / 408 / 404 | 12.98 / 13.08 / 13.06 / 12.98 |

Prefill is a clean **~2.5-2.7x** win, holding across depth. Decode is identical within noise —
CUTLASS is decode-neutral exactly as predicted. `routed_moe_launch_cutlass` does read
per-expert token counts back to host with a blocking `cudaMemcpy` once per CUTLASS layer per
forward pass (real code behavior, still worth removing eventually for decode latency headroom),
but it is evidently not the bottleneck at current decode throughput.

*(The pre-wiring history for this item — why CUTLASS was dead code, the `use_sorted_pairs`
prefill-batching regression it fixed — is still in `## Phases` under P3, since it's design
rationale worth keeping next to the still-**open**, unresolved ~13 vs ~25.5 t/s decode-number
discrepancy tracked there.)*
