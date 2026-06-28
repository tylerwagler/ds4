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

### P1 — MTP speculative decode  [CHEAP, DO FIRST]
Hits the actual bottleneck (single-stream decode, memory-bound). DeepSeek-V4-Flash
has a native MTP head; `ds4-bench`/server take `--mtp DRAFT.gguf`. Expected ~1.5-2x
decode. Effort: acquire/build the MTP draft + wire serving flags + validate accept rate.

### P2 — FP8 dense/attention, then FP8 experts  [MEDIUM]
Light up the idle FP8 cores; matches the FP8 source so it's near-lossless.
- **2a Dense/attn via cuBLASLt FP8** — ds4 already links cuBLAS; mature API, smallest lift.
  Lets the allocator drop less-sensitive FP8 attn from Q8_0(8.5) to FP8(~8.25) on hw path.
- **2b FP8 routed-expert grouped GEMM** — bigger; CUTLASS/vLLM grouped-MoE FP8.

### P3 — NVFP4 experts on FP4 tensor cores  [HARD, the serving play]
Source experts are FP4; NVFP4 (~4.5 bpw, FP8 per-16 microscale + FP32 global) matches
them and runs on the 5th-gen FP4 cores (2x FP8 / 4x BF16). Wins **prefill + concurrency**,
NOT single-stream decode (memory-bound). 
- Quantizer: NVFP4 encoder (two-level scaling; consider Hadamard rotation à la TRT-LLM).
- Runtime: **graft vLLM's CUTLASS NVFP4 grouped-MoE GEMM** (Apache-2.0) via a C bridge —
  same maneuver Entrpi used for llama.cpp mmq. Do NOT hand-write tcgen05 FP4 MMA.
- Risk: CUTLASS NVFP4 grouped-MoE on sm_121 is bleeding edge (expect toolchain friction).

### P4 — Oracle-driven multi-format allocation across the full menu
Extend `--mse-probe` + prisma to allocate over ALL tensors (experts + dense/attn) and
the curated format menu below, picking by imatrix-weighted real error per (tensor,format).

## Curated format menu (every format must earn its kernel)
| bpw  | format   | hw path        | role |
|------|----------|----------------|------|
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
1. **FP8 + NVFP4 — YES, staged.** FP8 via cuBLASLt (easy); NVFP4 via vLLM/CUTLASS graft (hard).
2. **INT4 / AutoRound — NO for this model.** AutoRound optimizes rounding *from high precision*;
   our experts are FP4 at source, so the gain collapses to RTN. INT4 ≈ Q4_K we already have.
   Revisit only for a bf16-source model.
3. **Many formats — YES, curated.** Dispatch is free; kernels are the cost. ~5-7 hw-aligned tiers.

## Rejected / parked
- **CUDA-graph decode capture** — eager wins this (GPU-bound) workload; antirez ships no graphs. Parked.
- **q5/q6 on EXPERTS** — FP4 source ceiling makes them wasteful. (q5/q6 encoders kept for a future >FP4 use.)
- **Entrpi mmq graft** — benchmark-gated vs antirez native Q4 MoE; likely unnecessary given P2/P3.

## Foundation status
- Clean fork off antirez/main (native GB10, batched Q4 MoE, mixed-precision serving, SSD streaming).
- Prisma quantizer toolchain merged (encoders iq2/q2/q3/q4/q5/q6 + dequants + `--mse-probe` + allocator).
- Pending: requant on tank-backed CT, then run the oracle, then P1 (MTP).
