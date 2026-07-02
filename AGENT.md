# Agent Notes

DwarfStar (`ds4`) is a DeepSeek V4 Flash/PRO specific inference engine, not a
generic GGUF runner. This tree is the **CUDA-only fork** of antirez's upstream
project, targeting the NVIDIA DGX Spark (GB10, `sm_121`, ~121 GB usable
unified memory). The Metal and ROCm backends were fully removed. The two
backends are `cuda` (default) and `cpu` (CPU inference retained only as
reference/diagnostic code, slated for eventual removal).

## Goals

- Keep the production path as whole-model CUDA graph inference on GB10.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF. Keep SSD
  streaming of routed experts explicit and overlap loads with compute.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with
  unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Don't introduce
  slop: fragile case-patching, dead code, or needless complexity.
- Comment inference code where model mechanics, cache lifetime, memory policy,
  or API orchestration are not obvious locally. Prefer comments beside the
  implementation over separate design documents.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- C++ is confined to the single CUTLASS TU (`src/cuda/ds4_mxfp4_cutlass.cu`).
  Everything else is C (or CUDA C).

## Layout

Public headers: `src/ds4.h` (engine API) and `src/ds4_gpu.h` (GPU graph API).

- `src/engine/` — 21 TUs + `ds4_engine_internal.h`: GGUF parsing, tokenizer,
  weight binder (`weights.c`), quant format/kernel tables, GPU graph
  orchestration (`gpu_graph_alloc.c`, `gpu_graph_state.c`, `gpu_prefill.c`,
  `gpu_decode.c`, `gpu_diag.c`), sessions + KV payload serialization, imatrix
  collection, steering, and the CPU reference path (`cpu_*.c`).
- `src/cuda/` — 7 kernel TUs + `ds4_cuda_internal.h`: runtime/memory
  (`ds4_cuda_runtime.cu`), matmuls incl. the cuBLASLt MXFP8 path
  (`ds4_cuda_matmul.cu`), attention, MoE, indexer, norm/KV, HC router; plus
  `ds4_mxfp4_cutlass.cu` (CUTLASS grouped GEMM for MXFP4 experts) and
  `ds4_iq2_tables_cuda.inc`.
- `src/server/` — 14 TUs + `ds4_server_internal.h`: HTTP server,
  OpenAI/Responses/Anthropic endpoints and streaming, prompt rendering, disk
  KV cache, exact-DSML tool replay.
- `src/agent/` — 20 TUs + `ds4_agent_internal.h`: native coding agent (tools,
  terminal UI, sessions, compaction).
- `src/cli/` — `ds4_cli.c`, `ds4_bench.c`, `ds4_eval.c` entry points.
- `src/lib/` — shared pieces: distributed inference, help text, kvstore, SSD
  streaming, web fetch.
- `src/vendor/` — linenoise, rax.
- `tests/` — C runners; `ds4_test.c` and `ds4_agent_test.c` are unity builds
  that `#include` the server/agent source lists.
- `cutlass/` — git submodule (v4.5.2), **required** for the MXFP4 expert path.
- `gguf-tools/` — offline quantization/imatrix tooling that produces the GGUFs
  this fork loads.

Internal-header convention: a symbol is de-static'd and declared in the
module's `ds4_*_internal.h` only when another TU of the same module needs it.
Everything else stays `static`. The cross-module surface is `src/ds4.h`,
`src/ds4_gpu.h`, and the `src/lib/*.h` headers only.

## Build

```sh
git submodule update --init cutlass   # once
make cuda-spark          # DGX Spark / GB10 (no explicit -arch; fastest on GB10)
make cuda-generic        # other local CUDA GPUs (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_N # explicit -arch, e.g. cross-builds
```

Binaries land in the repo root: `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`,
`ds4-eval`. Note `ds4_mxfp4_cutlass.cu` needs the `sm_120f` family for the
mxf4 block-scale MMA; the Makefile handles its flags.

## Testing

- `make test` runs `./ds4-eval --self-test-extractors`, `./ds4_agent_test`,
  and `./ds4_test`. The eval self-test and agent test need no model;
  `ds4_test` loads a model (`DS4_TEST_MODEL`, default `./ds4flash.gguf`).
- `make cuda-regression` runs `tests/cuda_long_context_smoke`: GPU kernel
  smoke tests, no model required.
- `./ds4_test --logprob-vectors` compares against official-API vectors and
  pins `DS4_CUDA_PREFILL_CHUNK=2048`.
- imatrix collection (`--imatrix-dataset` / `--imatrix-out`) requires `--cuda`.
- GPU/CPU cross-check diagnostics: `--gpu-graph-test`, `--gpu-graph-full-test`,
  `--gpu-graph-prompt-test`.

## Validation Culture

- **Refactors** (code movement, TU splits, renames) must produce byte-identical
  greedy output versus the pre-change binary.
- **Numerics changes** (kernels, quantization, activation formats) are
  perplexity-gated (`--perplexity-file`) plus the deterministic `ds4-eval`
  q1..q4 token-count gate; recent reference point on the 97 GB zero-Q8 Flash
  oracle: ppl 7.3216, decode 12.35 t/s, long-prompt prefill 162 t/s.
- **One ds4 process at a time on the GB10.** Two ~97 GB model mappings OOM the
  box; the instance lock (`DS4_LOCK_FILE`) is intentional.

## Supported Weight Formats (binder-enforced)

| Tensor group | Accepted formats |
| --- | --- |
| Attention projections, shared experts, MTP | MXFP8 (FP8 E4M3 + per-32 E8M0 scales) |
| Routed experts gate/up/down | exactly `IQ2_XXS`/`IQ2_XXS`/`Q2_K`, or all three `MXFP4` |
| Output head | `BF16` or MXFP8 |
| Norms, embeddings, indexer, HC | `F32`/`F16` |

Legacy `Q4_K`/`Q8_0` weights are rejected at load with one clear error
(`weights_reject_unsupported_types` in `src/engine/weights.c`). `Q8_K` exists
only as *activation* quantization inside the routed-expert (MoE) kernels.

## Compute Paths

- **Prefill:** dense matmuls run cuBLASLt block-scaled MXFP8×MXFP8
  (`VEC32_UE8M0`) on tensor cores, including the attention-output projections.
  MXFP4 expert prefill runs the CUTLASS mxf4×mxf4 block-scaled grouped GEMM
  (GB10 / `sm_120f` family).
- **Decode:** custom fused kernels, memory-bound; activations stay raw f32
  except inside MoE (Q8_K).
- **FP8 KV cache** is currently fake-quant: E4M3 rounding with f32 storage.

## Environment Variables

All runtime tuning/diagnostic gates use the `DS4_CUDA_*` prefix (this fork
renamed every `DS4_METAL_*` gate; there are no compatibility aliases).
Also: `DS4_FP8_NO_MXCORE`, `DS4_TEST_MODEL`, `DS4_LOCK_FILE`, `DS4_GGUF_DIR`
(download script).

## Deferred Work

- Real packed FP8 KV-cache storage (replace the fake-quant f32 path).
- Move MoE decode off Q8_K activation quantization.
- Eventually strip the CPU inference path entirely.
