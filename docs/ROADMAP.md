# ds4-spark Roadmap — best single-Spark DeepSeek-V4-Flash

**North star:** make DeepSeek-V4-Flash genuinely *good and fast* on one DGX Spark
(GB10/Blackwell, 128 GiB unified): high single-stream decode **and** real
concurrent serving. Quality + speed are the constraint; cost is not.

## Hardware we are NOT using yet
| unit | status | our source already uses it? |
|------|--------|------------------------------|
| FP8 (E4M3) tensor cores | idle | yes — attn/dense weights are FP8 in the checkpoint |
| FP4 (NVFP4) tensor cores | idle | yes — routed experts are FP4 (MXFP4) in the checkpoint |
| MTP head | unused — and DROPPED from current quant (no `blk.43`); needs re-quant | DeepSeek-V4-Flash ships one (`nextn_predict_layers=1`) |
| DSpark/DFlash drafts | no runtime in ds4 (MTP only) | official DSpark module exists upstream |

## Bottleneck map (decides which lever helps where)
- **Single-stream decode** = **memory-bandwidth bound** (read selected experts + KV/token).
  Levers: **MTP** (more useful tokens per memory sweep); smaller quant for the bulk.
  *Tensor cores DON'T help here* (batch=1, no compute to saturate).
- **Prefill** = **compute bound**. Levers: FP8 / NVFP4 tensor cores.
- **Concurrency (N streams)** = **compute bound** (batched grouped-GEMM). Lever: **NVFP4**.
  This is the serving play.

Measured (2026-06-29, oracle-fp8wh-99gb, FP8 workhorse path, no spec decode, un-clock-pinned):
decode ~25.5 t/s @depth0 / 24.5 @4k / 21.6 @8k (c1); prefill 431 / 399 / 389 t/s. That ~25.5
is ~74% of the ~32 t/s memory-BW ceiling (14B active) — the graph bug is fixed and we're now
near the bandwidth wall, so the remaining single-stream win is spec decode, not quant. (Old
smart-99gb baseline was ~12.7 — the 2x gap was the graph bug + greedy decode, now closed.)

## Phases

### P0 — HF→ds4 GGUF converter  [FOUNDATIONAL — owns both levers]
Today we re-quantize antirez's pre-made `template.gguf`, inheriting its conversion choices
(it **dropped the MTP head**; we can't change per-tensor precision at conversion time). Owning
the HF→ds4 conversion gives full control of speed AND quality: set precision per tensor, include
MTP, emit FP8/MXFP4 drafts matched to the main model. **De-risked:** the `hc_*` highway tensors
are present in the HF source (`layers.N.hc_attn_base/fn/scale`, `hc_ffn_*`) — NOT derived — so
this is a mechanical name-remap + expert-stack + weight/scale→block-pack, not architecture RE.
- **Source:** `DeepSeek-V4-Flash` HF safetensors (46 shards): `layers.N.attn.{wkv,wq_a,wq_b,wo_a,wo_b}`,
  `layers.N.hc_*`, `layers.N.ffn.experts.K.w{1,2,3}`, `mtp.0.*` (shard 46) — each `.weight`+`.scale`
  (native MXFP4 E2M1+E8M0 / FP8 E4M3).
- **Target:** match `template.gguf`'s exact block layout (reverse-engineer by diffing one known
  tensor HF-vs-template); remap to `blk.N.*`/`mtp.0.*`, stack experts into `ffn_{gate,up,down}_exps`,
  write `deepseek4.*` metadata.
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
DFlash + Eagle3), NOT DGX Spark (naming clash). Staged:

- **P1a — Native MTP baseline (OFF-THE-SHELF DRAFT, do first).**
  The MTP head is NOT embedded in the main GGUF — ds4 loads it as a **separate draft model**
  (`mtp_weights_bind(&e->mtp_weights, &e->mtp_model)`) via `--mtp FILE`. antirez ships the
  draft pre-made: **`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`** (~3.5 GB) in
  `antirez/deepseek-v4-gguf` (`download_model.sh mtp`). So this is a **3.5 GB download**, not a
  re-quant. (The `nextn_predict_layers=1` metadata in the main quant is a red herring — the
  HF source ships `mtp.0.*` 1575 tensors but our `template.gguf` dropped them; doesn't matter,
  the draft is standalone.) Serve `oracle-fp8wh --mtp DRAFT.gguf --mtp-draft 2 --mtp-margin 3`;
  DeepSeek native MTP accepts ~1.8-2.2 tok/step → clock-pinned **~45-55 t/s**. Proves the
  verify/accept loop on THIS model before investing in a custom DSpark runtime.

- **P1b — DSpark serving (the 60+ play, BIG lift, two builds).**
  Source: `deepseek-ai/DeepSeek-V4-Flash-DSpark` = same weights + an official pre-trained
  DSpark draft module (higher accept than vanilla MTP). Two prerequisites, BOTH currently
  missing in ds4:
  1. **DSpark drafter → GGUF.** Convert the DSpark draft module to ds4's draft format.
     OPEN: confirm a DS4-Flash DSpark/DFlash drafter is published (z-lab hosts DFlash for
     Qwen3.5/3.6, Kimi; drafter is model-specific — a Qwen one will NOT transfer).
  2. **DSpark runtime in ds4.c.** ds4 has MTP only — NO DSpark/DFlash/Eagle3 runtime.
     Needs: draft-model load, the multi-step draft → batched verify → accept loop, and
     draft attention (tree or linear). This is real engine code, not a flag.
  BUILD-vs-INHERIT: best case antirez adds DSpark upstream and we inherit on the clean fork
  — watch `antirez/main`. Otherwise we build the accept-loop ourselves (P1a's `--mtp` path
  is the scaffold to extend). Decide after P1a's measured native-MTP number.

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

## Curated format menu (every format must earn its kernel)
| bpw  | format   | hw path        | role |
|------|----------|----------------|------|
| 0    | PRUNE    | n/a (skip)     | dead experts for the workload (REAP/RIY) |
| 2.06 | IQ2_XXS  | int (native)   | bulk experts (memory-bound decode) |
| 2.6  | Q2_K     | int (native)   | bulk experts |
| 3.4  | Q3_K     | int (native)   | mid experts (granularity) |
| 4.5  | Q4_K     | int / mmq      | rich experts (memory-bound; lossy vs E2M1 source) |
| 4.25 | **MXFP4**| **FP4 cores**  | rich experts — LOSSLESS vs the E2M1+E8M0/32 source, replaces Q4_K |
| 4.5  | NVFP4    | FP4 cores      | only for quantizing from bf16 (block-16 E4M3 + global); not for experts |
| ~8.25| **FP8**  | **FP8 cores**  | attention/dense (source-matched) |
| 8.5  | Q8_0     | int            | precision-critical attn/output |
Rule: ~5-7 non-redundant, hardware-aligned tiers. The oracle picks; don't stack
redundant same-bpw formats (e.g. NVFP4 vs Q4_K is a *path* choice, not two slots).

## Idea verdicts (2026-06-28)
1. **FP8 + MXFP4 — YES, staged, both NVIDIA-first (vLLM optional).** FP8 (MXFP8) = cuBLASLt
   dense GEMM, no vLLM/CUTLASS (KV already in antirez). MXFP4 = the experts' *native* source
   format (E2M1+E8M0/32) → lossless, replaces lossy Q4_K, rides the same VEC32_UE8M0 swizzle.
   The matmul is NVIDIA (cuBLASLt/CUTLASS); vLLM only a shortcut for grouped-MoE fusion (the
   prefill/concurrency play, decode-neutral). Encoders are ours. NVFP4 = from-bf16 only.
2. **INT4 / AutoRound — NO for this model.** AutoRound optimizes rounding *from high precision*;
   our experts are FP4 at source, so the gain collapses to RTN. INT4 ≈ Q4_K we already have.
   Revisit only for a bf16-source model.
3. **Many formats — YES, curated.** Dispatch is free; kernels are the cost. ~5-7 hw-aligned tiers.
4. **Expert pruning (REAP/RIY) — YES, high value.** Workload-profiled dead-expert removal
   = the 0-bit tier; reclaims memory to keep live experts at FP4 instead of IQ2. The
   activation profile becomes the master sensitivity signal for prune+quant+stream.

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
- NEXT: P1a — re-quant including `blk.43` MTP head → native `--mtp` baseline → then P1b DSpark serving.
