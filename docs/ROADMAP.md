# ds4-spark Roadmap — best single-Spark DeepSeek-V4-Flash

**North star:** make DeepSeek-V4-Flash genuinely *good and fast* on one DGX Spark
(GB10/Blackwell, 128 GiB unified): high single-stream decode **and** real
concurrent serving. Quality + speed are the constraint; cost is not.

## Hardware we are NOT using yet
| unit | status | our source already uses it? |
|------|--------|------------------------------|
| FP8 (E4M3) tensor cores | idle | yes — attn/dense weights are FP8 in the checkpoint |
| FP4 (NVFP4) tensor cores | idle | yes — routed experts are FP4 (MXFP4) in the checkpoint |
| MTP head | unused (greedy decode) | DeepSeek-V4-Flash ships one |

## Bottleneck map (decides which lever helps where)
- **Single-stream decode** = **memory-bandwidth bound** (read selected experts + KV/token).
  Levers: **MTP** (more useful tokens per memory sweep); smaller quant for the bulk.
  *Tensor cores DON'T help here* (batch=1, no compute to saturate).
- **Prefill** = **compute bound**. Levers: FP8 / NVFP4 tensor cores.
- **Concurrency (N streams)** = **compute bound** (batched grouped-GEMM). Lever: **NVFP4**.
  This is the serving play.

Forum reference (DGX Spark, ds4, IQ2_XXS, fully resident): ~25-28 tok/s decode
@depth0, ~22 @16k. We currently measure ~12.7 on smart-99gb → ~2x gap on
single-stream decode. Prime suspect: MTP (we run without it).

## Phases

### P1 — Speculative decode  [HIGHEST LEVERAGE, DO FIRST]
Hits the actual bottleneck (single-stream decode, memory-bound). Two draft options:
- **Original MTP** — V4-Flash built-in MTP head; ds4 supports it (`--mtp DRAFT.gguf`).
- **DSpark (DeepSeek newer method, 2026-06)** — `deepseek-ai/DeepSeek-V4-Flash-DSpark`
  = same weights + an official pre-trained DSpark draft module (framework
  github.com/deepseek-ai/DeepSpec; also ships DFlash + Eagle3). Likely higher accept
  rate than vanilla MTP. "DSpark" = the spec-decode method, NOT DGX Spark (naming clash).
OPEN Q: does ds4 spec-decode accept a DSpark draft (GGUF-convert + scheme match) or need
new runtime support? Best case: antirez adds DSpark upstream and we inherit it on the
clean fork. Action: watch antirez/main for DSpark; validate original MTP as the baseline
win meanwhile. Expected ~1.5-2x decode either way.

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
- **2c FP8 weights via cuBLASLt** — attn/dense/shared Q8_0(8.5)→E4M3(8.0): lossless,
  ~6% smaller, FP8-core matmul. `cublasLtMatmul` with FP8 descriptors + scale tensors;
  dense GEMM, so a library drop-in (TN layout/alignment constraints, no kernel graft).
  Shared experts are always-on dense GEMMs → decode-relevant. **No vLLM, no CUTLASS.**

### P3 — NVFP4 experts on FP4 tensor cores  [HARD, the serving play]
Source experts are FP4; NVFP4 (~4.5 bpw, FP8 per-16 microscale + FP32 global) matches
them and runs on the 5th-gen FP4 cores (2x FP8 / 4x BF16). **DECODE-NEUTRAL** — same
4.5 bpw as Q4_K = same bytes = same decode bandwidth. Wins **prefill + concurrency**
(compute-bound) only. Defer unless the workload becomes prefill-heavy or multi-stream.
Bonus quality angle: NVFP4's per-16 microscale represents the massive-activation late-down
experts (blk.42/41/30… — see oracle findings) better than Q4_K's coarser blocks.

Dependency reality: **the math is NVIDIA; vLLM is optional and only for MoE fusion.**
- GEMM primitive: cuBLASLt added block-scaled FP4 (NVFP4/MXFP4) matmul in CUDA 12.8→13.x
  for Blackwell — CONFIRM the API covers the grouped/batched-expert case on sm_121 before
  committing. Otherwise CUTLASS NVFP4 grouped-GEMM collectives (also NVIDIA).
- The hard part is **MoE routing + fusion** (token→expert gather, per-expert block scales),
  NOT the matmul. Options: (a) build on CUTLASS directly; (b) **lift vLLM's fused
  NVFP4-MoE kernel** — itself a CUTLASS wrapper, Apache-2.0, via a C bridge (the maneuver
  Entrpi used for llama.cpp mmq). vLLM is a shortcut for the fusion, not a requirement.
- Quantizer: NVFP4 encoder is ours (two-level scaling; consider Hadamard rotation à la
  TRT-LLM). Do NOT hand-write tcgen05 FP4 MMA.
- Risk: CUTLASS/cuBLASLt NVFP4 grouped-MoE on sm_121 is bleeding edge (toolchain friction).

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
| 4.5  | Q4_K     | int / mmq      | sensitive experts (memory-bound) |
| 4.5  | **NVFP4**| **FP4 cores**  | sensitive experts (compute-bound: prefill/concurrency) |
| ~8.25| **FP8**  | **FP8 cores**  | attention/dense (source-matched) |
| 8.5  | Q8_0     | int            | precision-critical attn/output |
Rule: ~5-7 non-redundant, hardware-aligned tiers. The oracle picks; don't stack
redundant same-bpw formats (e.g. NVFP4 vs Q4_K is a *path* choice, not two slots).

## Idea verdicts (2026-06-28)
1. **FP8 + NVFP4 — YES, staged, both NVIDIA-first (vLLM optional).** FP8 = cuBLASLt
   dense GEMM, no vLLM/CUTLASS (the decode play; KV already in antirez). NVFP4 = NVIDIA
   block-scaled FP4 (cuBLASLt/CUTLASS) for the matmul; vLLM only as a shortcut for the
   MoE routing+fusion (the prefill/concurrency play, decode-neutral). Encoders are ours.
2. **INT4 / AutoRound — NO for this model.** AutoRound optimizes rounding *from high precision*;
   our experts are FP4 at source, so the gain collapses to RTN. INT4 ≈ Q4_K we already have.
   Revisit only for a bf16-source model.
3. **Many formats — YES, curated.** Dispatch is free; kernels are the cost. ~5-7 hw-aligned tiers.
4. **Expert pruning (REAP/RIY) — YES, high value.** Workload-profiled dead-expert removal
   = the 0-bit tier; reclaims memory to keep live experts at FP4 instead of IQ2. The
   activation profile becomes the master sensitivity signal for prune+quant+stream.

## Rejected / parked
- **CUDA-graph decode capture** — eager wins this (GPU-bound) workload; antirez ships no graphs. Parked.
- **q5/q6 on EXPERTS** — FP4 source ceiling makes them wasteful. (q5/q6 encoders kept for a future >FP4 use.)
- **Entrpi mmq graft** — benchmark-gated vs antirez native Q4 MoE; likely unnecessary given P2/P3.

## Foundation status
- Clean fork off antirez/main (native GB10, batched Q4 MoE, mixed-precision serving, SSD streaming).
- Prisma quantizer toolchain merged (encoders iq2/q2/q3/q4/q5/q6 + dequants + `--mse-probe` + allocator).
- Pending: requant on tank-backed CT, then run the oracle, then P1 (MTP).
