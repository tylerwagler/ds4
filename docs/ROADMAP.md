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
P3 status entry for the head-to-head numbers that surfaced this.

## Phases

### P0 — HF→ds4 GGUF converter  [FOUNDATIONAL — owns both levers]
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

  - **Flow**: target layers 40-42 HC states → concat → `main_proj` FP8 matmul → `main_norm` →
    [decoder layer mtp.0] → [decoder layer mtp.1] → [decoder layer mtp.2] →
    Markov refine (sequential over 5 positions) → confidence score → HC collapse → lm_head.
  - **Block size**: 5 draft tokens per speculative step (configurable).
  - **Draft KV**: 3-layer raw SWA cache (no compressor state, window=128), seeded from
    target hidden states via `kv_from_hidden`.
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
     *Effort: ~100 lines.*

  3. **GPU graph allocation** (`gpu_graph_alloc.c`) — extend `ds4_gpu_graph` with draft
     model tensor views: 3 decoder layers' working buffers, 3-layer draft KV raw cache
     (128 SWA window each), hidden state intermediates for projection/Markov/confidence.
     *Effort: ~200 lines.*

  4. **Draft backbone forward** — `dspark_forward_backbone()`: runs 3 decoder layers via
     the existing `gpu_graph_encode_decode_layer` pipeline, but with a **non-causal**
     attention mask. The 5 draft positions form a sliding-window "sentence" where every
     token attends to every other. This requires adding a 4th attention mode to the fused
     kernel (alongside causal, SWA, batched). *Effort: ~200 lines kernel mod + orchestration.*

  5. **Target hidden capture** — capture main model's HC state at layers 40, 41, 42 →
     concat → `dspark.main_proj` FP8 matmul → `dspark.main_norm` RMS norm.
     The target layers' HC tensors are live in the main graph after each decode step;
     this reads them before they're overwritten. *Effort: ~100 lines.*

  6. **Markov + confidence heads** — two small CUDA kernels:
     - `dspark_markov_refine()`: for each of 5 positions, gather `markov_w1[prev_token]`,
       project via `markov_w2`, add to base logits, argmax. Sequential across positions,
       but each step is just an embedding lookup + bias add.
     - `dspark_confidence_score()`: concat hidden + markov embed, linear, sigmoid.
     *Effort: ~150 lines.*

  7. **Block spec decode cycle** (`session.c`) — new `ds4_session_eval_speculative_block()`,
     parallel to the existing `ds4_session_eval_speculative_argmax`. Flow per step:
     (1) capture target hidden states from layers 40-42,
     (2) `project_main_hidden()` → 3-layer backbone → block hidden,
     (3) shared norm + lm_head → base logits for all 5 positions,
     (4) Markov refine + confidence score,
     (5) submit draft block to target verify (`gpu_graph_verify_suffix_tops`),
     (6) rejection sampling: accept longest valid prefix,
     (7) materialize accepted KV rows into draft cache via `kv_from_hidden`,
     (8) frontier snapshot/restore for partial accept (reuse existing `spec_frontier_*` infra).
     *Effort: ~600-1000 lines. Crib from vLLM's `speculator.py`/`dspark_worker_v2.py`
     and SGLang's `dspark_worker_v2.py` — the logic is well-documented Python, transliterate
     to C/CUDA.*

  8. **CUDA graph capture** — capture the full draft cycle (capture + backbone + markov +
     verify) in one CUDA graph per decode step, matching vLLM's approach. Reuse the existing
     CUDA graph infrastructure. *Effort: ~100 lines.*

  9. **Server integration** (`generate.c`, `cli_main.c`) — `--dspark` flag, gating rules:
     greedy-only (temperature <= 0.0), not with SSD-streaming, conflicts with `--mtp`.
     Wire to `ds4_session_eval_speculative_block()` in the server's generation loop.
     *Effort: ~50 lines.*

  10. **Load-aware scheduler** (deferrable) — confidence-threshold based verification
      budget trim, per P1b measured numbers. The confidence head outputs per-position
      survival probability; the scheduler uses these + GPU utilization to decide how
      many of the 5 draft tokens to actually submit for verification. Useful for
      concurrency; nearly irrelevant for single-stream. *Effort: ~200-300 lines.*

  **GPU memory budget for DSpark draft:**
  | Component | Size |
  |-----------|------|
  | 3 decoder layer weights (FP8 attn/dense/shared, MXFP4 experts) | ~9 GB |
  | 3-layer draft KV cache (128 SWA × 3) | ~7 MiB |
  | Hidden state buffers (backbone + projection intermediates) | ~10 MiB |
  | **Total** | **~9 GB** (vs 97 GB target → ~9% overhead) |

  **Open questions to resolve before P1b build:**
  1. **GGUF format**: Bundle draft weights in the main GGUF, or load as a separate file
     (`--dspark DRAFT.gguf`, same pattern as `--mtp`)?
  2. **Non-causal attention**: The existing fused attention kernels (`ds4_cuda_attn.cu`)
     assume causal masking. How much surgery to add a non-causal SWA mode for the 3 draft
     layers? vLLM reused SparseMLA backends; ds4 has different attention kernel architecture.
  3. **Confidence threshold**: Tunable runtime flag or hardcoded default (paper uses 0.5)?
  4. **Target layer selection**: DSpark Flash uses layers 40-42. Configurable, or bake it?

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
- **2b Profiling GATE** — profile the oracle quant's decode (attn vs shared vs MoE vs KV
  split) BEFORE building 2c. Earlier nsys put attention ~6.6ms and the real stall was the
  now-fixed graph bug → decode is likely MoE-memory-bound. If attn/shared is a thin slice,
  2c is a quality win more than a speed one; if fat, build it.
- **2c FP8 weights via cuBLASLt — DONE.** attn_q/kv + shared experts Q8_0→MXFP8 via cuBLASLt
  MX prefill + mmvq decode; validated coherent, 25.5 t/s decode / 431 t/s prefill @d0.
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
  **STATUS UPDATE (2026-07-02):** `attn_output_a`/`attn_output_b` are now on MXFP8 — fused
  MXFP8×MXFP8 kernels for both decode and prefill (`cuda_attention_output_a_mx_gemm`,
  `ds4_cuda_matmul.cu:655`), currently the uncommitted rework being validated. Confirmed the
  `rank%128==0, group_dim%32==0` shape gate on `attn_output_a`'s MX-GEMM path is satisfied by
  the real Flash/Pro shapes (`rank=1024, group_dim=4096`) — the fast path is reachable, not a
  silent fallback. Remaining Q8/BF16 item is lm_head only. TODO: **the KL/perplexity gate
  mentioned above does not exist in code** — no `kl_div`/`perplexity`-gated runtime logic
  found anywhere; format selection today is a raw `w->output->type == DS4_TENSOR_BF16` check
  (`weights.c:538-543`) on whatever the GGUF ships. Building that gate is a prerequisite, not
  a formality, before flipping lm_head to FP8.
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
- **WIRED (2026-07-02): the CUTLASS path is live, functionally correct, and a clean prefill
  win with no measured decode cost.** New tensor type `DS4_TENSOR_CUTLASS_MXFP4 = 40`
  (expert-major ColumnMajor E2M1 data + swizzled E8M0 SF per expert,
  `cutlass_mxfp4_expert_layout()` in `gguf.c`); accepted through `weights.c` validation as a
  third routed-expert combo; a new `routed_moe_launch_cutlass()` in `ds4_cuda_moe.cu` sorts
  tokens by expert (reusing the existing count/prefix/scatter kernels), gathers each active
  expert's rows, calls a rewritten `ds4_cutlass_expert_ffn_scratch()` (caller-provided scratch
  via `cuda_tmp_alloc`, no more per-call cudaMalloc/cudaFree/cudaDeviceSynchronize — that was
  10 allocations per call in the original standalone-test code), and scatters results back for
  the existing `moe_sum_kernel` to reduce. Standalone CUTLASS kernel test still passes
  bit-exact (`max_rel=0.00000`) after the scratch-buffer refactor. Loaded and served
  `oracle-cutlass-mxfp4-99gb.gguf` end-to-end; `--logprob-vectors`, `--tensor-equivalence`,
  `--short-prefill-ratio4`, `--local-golden-vectors` all pass, zero top-1 mismatches.
  **Measured on this GB10** (`ds4-bench`, `speed-bench/promessi_sposi.txt`, ctx 2048-8192),
  head-to-head against `oracle-zeroq8-99gb.gguf` on the identical build/hardware/methodology
  (correcting an earlier draft of this entry that compared against the old, no-longer-present
  `oracle-fp8wh-99gb` log line instead of measuring fresh — do not trust historical numbers
  from a file that isn't on disk):
  | | prefill t/s | decode t/s |
  |---|---|---|
  | baseline (zeroq8) | 167 / 163 / 161 / 161 | 12.98 / 13.14 / 13.10 / 13.04 |
  | CUTLASS | 445 / 414 / 408 / 404 | 12.98 / 13.08 / 13.06 / 12.98 |

  Prefill is a clean **~2.5-2.7x** win, holding across depth. Decode is identical within
  noise — CUTLASS is decode-neutral exactly as this section originally predicted.
  `routed_moe_launch_cutlass` does read per-expert token counts back to host with a blocking
  `cudaMemcpy` once per CUTLASS layer per forward pass (real code behavior, still worth
  removing eventually for decode latency headroom), but it is evidently not the bottleneck at
  current decode throughput.
  **OPEN QUESTION (bigger than this item):** both baseline and CUTLASS decode sit at ~13 t/s
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

## Tasks

| ID | Task | Phase | Status | Blocked By | Est. Effort |
|----|------|-------|--------|------------|-------------|
| P0 | HF→ds4 GGUF converter (repacker + quantizer) | P0 | **done** | — | ~1-2 weeks |
| P1.1 | DSpark: GGUF conversion | P1 | planned | — | 200-300 lines |
| P1.2 | DSpark: weight binding | P1 | planned | P1.1 | 100 lines |
| P1.3 | DSpark: GPU graph allocation | P1 | planned | P1.1 | 200 lines |
| P1.4 | Non-causal SWA attention mode (= A4) | P1 | planned | — | 200 lines |
| P1.5 | DSpark: target hidden capture | P1 | planned | P1.2 | 100 lines |
| P1.6 | DSpark: Markov + confidence heads | P1 | planned | P1.2 | 150 lines |
| P1.7 | DSpark: block spec decode cycle | P1 | planned | P1.3, P1.4, P1.5, P1.6, A3 | 600-1000 lines |
| P1.8 | DSpark: CUDA graph capture | P1 | planned | P1.7 | 100 lines |
| P1.9 | DSpark: server integration (`--dspark`) | P1 | planned | P1.7 | 50 lines |
| P1.10 | DSpark: load-aware scheduler | P1 | deferred | P1.7 | 200-300 lines |
| P2a | Real packed FP8 KV cache (replace fake-quant) | P2 | planned | — | ~1 week |
| P2b | Profiling GATE (decode split analysis) | P2 | planned | — | ~1 day |
| P2d.lm | lm_head MXFP8 + KL/perplexity gate | P2 | planned | P2b | ~3 days |
| P2e | MXFP8 for F16 weight groups (compressor, indexer, HC) | P2 | planned | — | ~1 week |
| P3 | Wire MXFP4 CUTLASS TC path (or extend prefill batching) | P3 | planned | P0 | ~2 weeks |
| P4 | Oracle-driven multi-format allocation | P4 | deferred | P2/P3 formats settled | ~1 week |
| P5 | Expert pruning (REAP/RIY) | P5 | deferred | — | ~1 week |
| A1 | Fused prefill attention kernel (Flash-style tiling) | — | planned | — | ~2 weeks |
| A3 | Multi-sequence KV paging (vLLM-style) | — | planned | — | ~1 week |

**Legenda:**
- Status: **NEXT** = immediate next item; **planned** = spec'd but not started; **in_progress** = active; **done** = complete; **deferred** = lower priority, not blocking.
- A1 and A3 are attention-infrastructure items tracked here but owned by the appendix.
- P2c and P2d.attn_output (attn_output_a/b MXFP8) are **done** — not listed here.
- MTP (P1a) skipped — straight to DSpark.

**Dependency graph (simplified):**
```
P0 ──┬──→ P3 (MXFP4 TC path)
     └──→ P2d.lm (lm_head format)
P1.1 → P1.2 → P1.3 ─┐
P1.4 ────────────────┤
P1.5 ────────────────┤──→ P1.7 → P1.8 → P1.9
P1.6 ────────────────┤         └──→ P1.10
A3 ──────────────────┘
P2b ──→ P2d.lm (profile before lm_head FP8)
```

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
   slab. Both MTP (existing) and DSpark (planned) need independent KV state for multiple
   candidate sequences during verification. A page-table-based KV cache (vLLM-style) would
   let draft and target models share memory efficiently, and enable the `kv_from_hidden` seed
   required by DSpark's draft KV initialization. *Prerequisite for P1b item 7 (block spec
   decode cycle).*

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

## Foundation status
- Clean fork off antirez/main (native GB10, batched Q4 MoE, mixed-precision serving, SSD streaming).
- Prisma quantizer toolchain merged (encoders iq2/q2/q3/q4/q5/q6 + dequants + `--mse-probe` + allocator).
- DONE: oracle quant (oracle-99gb, Q8 non-experts) validated at parity with hand baseline.
- DONE: P2 FP8 workhorse (MXFP8 attn/dense/shared via cuBLASLt MX prefill + mmvq decode) —
  validated coherent, measured 25.5 t/s decode / 431 t/s prefill @depth0.
- DONE: Prefill QK-norm+RoPE fusion (2026-07-02). `gpu_prefill.c` now uses the same fused
  `ds4_gpu_head_rms_norm_rope_tail_tensor` kernel decode already used. Verified bit-identical
  against reference via `--logprob-vectors`, `--local-golden-vectors`, `--tensor-equivalence`.
- DONE (2026-07-02): performance audit of full active-path table against source. Found P3
  CUTLASS MXFP4 path is dead code (never called) and disables prefill batching for MXFP4
  experts; KV "FP8" (2a) is fake-quant only with zero bandwidth win banked; confirmed the
  `attn_output_a` MX-GEMM shape gate is reachable for real model shapes. See P2/P3 status
  updates, new 2e item, and the Attention appendix for deeper analysis.
- DONE (2026-07-02): P0 quantizer upgraded — added MXFP4 quantize/dequant output, MTP tensor
  name mapping + expert handling in `deepseek4-quantize.c`, stripped unused types.
  Can produce IQ2_XXS/Q2_K/FP8_E4M3/MXFP4 GGUFs from HF safetensors.
- NEXT: P1 — DSpark runtime (GGUF conversion → weight binding → GPU graph → block cycle).
