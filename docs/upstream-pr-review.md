# Upstream PR Review — antirez/ds4 → ds4-gb10 (CUDA-only fork)

**Date:** 2026-07-04
**Source:** https://github.com/antirez/ds4/pulls (160 open, 140 closed)

## Tracking Legend

| Mark | Meaning |
|------|---------|
| `[ ]` | Not yet reviewed |
| `[~]` | In progress |
| `[x]` | Reviewed — adopted / adapted / decided not to merge |
| `skip` | Not applicable (Metal/ROCm/macOS/benchmarks-only) |

## Progress

| Category | Total | Done | Remaining |
|----------|-------|------|-----------|
| Direct hits | 23 | 0 | 23 |
| Applicable | 95 | 0 | 95 |
| Not applicable | 42 | 42 | 0 |

---

PRs are classified by relevance to the CUDA-only GB10 fork:
- **Direct hit** — touches `src/cuda/` directly, or targets GB10-class hardware (unified memory, ≥112 GiB).
- **Applicable** — server/agent/engine/core code that is backend-agnostic.
- **Not applicable** — Metal/ROCm/Windows-specific, or already handled differently.

---

## Direct Hits (must-review for GB10)

| St | PR | Title | Why |
|----|-----|-------|-----|
| [ ] | [#158](https://github.com/antirez/ds4/pull/158) | cuda: fix HMM path on coherent unified-memory systems (GB10 NVLink-C2C) | GB10-specific HMM fix |
| [ ] | [#472](https://github.com/antirez/ds4/pull/472) | CUDA: scale q8→f16 cache reserve on ≥112 GiB cards | Fixes session OOM on GB10-sized models |
| [ ] | [#487](https://github.com/antirez/ds4/pull/487) | cuda: fall back to pinned host memory when the model arena runs out of VRAM | Critical for ~121 GB models near the memory limit |
| [ ] | [#466](https://github.com/antirez/ds4/pull/466) | Fix CUDA MoE router hardcoded to 256 experts | Generalizes expert count; Pro/non-standard models |
| [ ] | [#435](https://github.com/antirez/ds4/pull/435) | cuda: generalize router-select for arbitrary expert count (fixes Pro on CUDA) | Same class as #466, may supersede |
| [ ] | [#478](https://github.com/antirez/ds4/pull/478) | CUDA: make DeepSeek-V4-Pro correct on indexed-attention path (top_k 512→1024) + enable decode LUT gate for in_dim>4096 | Pro correctness |
| [ ] | [#349](https://github.com/antirez/ds4/pull/349) | Port CUDA SSD streaming support | SSD expert streaming — core GB10 goal |
| [ ] | [#497](https://github.com/antirez/ds4/pull/497) | cuda: run streaming selected load for single-token ffn batch encodes | Optimizes streaming on decode |
| [ ] | [#460](https://github.com/antirez/ds4/pull/460) | CUDA: batch gate/up/down uploads for selected expert cache misses | Cache-miss optimization |
| [ ] | [#153](https://github.com/antirez/ds4/pull/153) | cuda: add direct-model partial weight cache | Weight caching strategy |
| [ ] | [#266](https://github.com/antirez/ds4/pull/266) | cuda: increase matmul rows/block from 8 to 16 | Matmul perf |
| [ ] | [#145](https://github.com/antirez/ds4/pull/145) | cuda: add launch-bounded tile8 MoE down path | MoE kernel perf |
| [ ] | [#121](https://github.com/antirez/ds4/pull/121) | cuda: skip ordered f16 matmul on Blackwell | sm_121-specific (Blackwell arch) |
| [ ] | [#494](https://github.com/antirez/ds4/pull/494) | cuda: support CUDA toolkits older than 12.2 | Build compatibility |
| [ ] | [#279](https://github.com/antirez/ds4/pull/279) | Add Q4_K MoE prefill kernels + CUDA 11.x compatibility for V100 | Additional quant format support |
| [ ] | [#187](https://github.com/antirez/ds4/pull/187) | Fast CUDA backend: vendored mmq matmul, VMM weight layout, speculative-decoding plumbing | Major CUDA backend work |
| [ ] | [#377](https://github.com/antirez/ds4/pull/377) | [2/N] add cuda imatrix support for custom RL model | CUDA imatrix collection |
| [ ] | [#402](https://github.com/antirez/ds4/pull/402) | [3/N] add prefetch support for CUDA backend (2.75× faster) | CUDA prefetch |
| [ ] | [#178](https://github.com/antirez/ds4/pull/178) | [codex] Fix CUDA object rebuild on config changes | CUDA build system |
| [ ] | [#488](https://github.com/antirez/ds4/pull/488) | cuda: enable streaming auto cache (implement recommended_working_set_size) | Memory pressure auto-tuning |
| [ ] | [#86](https://github.com/antirez/ds4/pull/86) | docker & compose envelope for CUDA systems | Build/deployment |
| [ ] | [#317](https://github.com/antirez/ds4/pull/317) | fix: limit direct/full-map fallbacks to layer span cover for distributed CUDA workers | Distributed CUDA |
| [ ] | [#109](https://github.com/antirez/ds4/pull/109) | docs: add Jetson Thor CUDA notes | Jetson doc — tangentially relevant |

---

## Highly Applicable (server/agent/engine, backend-agnostic)

### Performance / Speculative Decoding

| St | PR | Title |
|----|-----|-------|
| [ ] | [#371](https://github.com/antirez/ds4/pull/371) | Add continuous depth-1 MTP speculation (DS4_MTP_CONTINUOUS) |
| [ ] | [#396](https://github.com/antirez/ds4/pull/396) | Add env-gated prompt-lookup speculative decoding for greedy generation |
| [ ] | [#261](https://github.com/antirez/ds4/pull/261) | Add suffix-tree speculative decoding for repetitive/agentic generation patterns |
| [ ] | [#385](https://github.com/antirez/ds4/pull/385) | Skip MTP probes while drafts go unused |
| [ ] | [#381](https://github.com/antirez/ds4/pull/381) | Clamp MTP draft depth to the prefill capacity |
| [ ] | [#206](https://github.com/antirez/ds4/pull/206) | mtp: make speculation disable skip draft work |
| [ ] | [#411](https://github.com/antirez/ds4/pull/411) | Fix bug with impact on DeepSeek V4 Pro MTP Drafter usage |
| [ ] | [#205](https://github.com/antirez/ds4/pull/205) | bench: add MTP options to ds4-bench |
| [ ] | [#480](https://github.com/antirez/ds4/pull/480) | Add DSpark speculative draft runtime |
| [ ] | [#482](https://github.com/antirez/ds4/pull/482) | DSpark B2 rejection sampling + adaptive block sizing |

### Server / HTTP API

| St | PR | Title |
|----|-----|-------|
| [ ] | [#489](https://github.com/antirez/ds4/pull/489) | Server: fix agent-loop cache misses, add cancellation, observability, and robustness fixes |
| [ ] | [#269](https://github.com/antirez/ds4/pull/269) / [#53](https://github.com/antirez/ds4/pull/53) | Add --api-key Bearer token authentication to ds4-server |
| [ ] | [#376](https://github.com/antirez/ds4/pull/376) / [#242](https://github.com/antirez/ds4/pull/242) | server: abort prefill when the client disconnects |
| [ ] | [#374](https://github.com/antirez/ds4/pull/374) | ds4_server: Add /health endpoint once model is fully loaded |
| [ ] | [#326](https://github.com/antirez/ds4/pull/326) | Add read-only introspection endpoints: /health, /info, /config |
| [ ] | [#81](https://github.com/antirez/ds4/pull/81) | Add /props introspection endpoint and model meta block |
| [ ] | [#90](https://github.com/antirez/ds4/pull/90) | feat(server): add /v1/messages/count_tokens endpoint |
| [ ] | [#419](https://github.com/antirez/ds4/pull/419) | server: expose only the loaded model in /v1/models |
| [ ] | [#287](https://github.com/antirez/ds4/pull/287) | fix(server): expose loaded model metadata |
| [ ] | [#456](https://github.com/antirez/ds4/pull/456) | Add served model name option for server discovery |
| [ ] | [#263](https://github.com/antirez/ds4/pull/263) | Advertise deepseek-chat model alias |
| [ ] | [#61](https://github.com/antirez/ds4/pull/61) | feat(server): support llama.cpp-style raw completions |
| [ ] | [#423](https://github.com/antirez/ds4/pull/423) | Fix: ds4-server rejects HTTP requests using Transfer-Encoding: chunked |
| [ ] | [#45](https://github.com/antirez/ds4/pull/45) | Validate HTTP Content-Length strictly |
| [ ] | [#433](https://github.com/antirez/ds4/pull/433) | Fix server JSON duplicate-field cleanup |
| [ ] | [#280](https://github.com/antirez/ds4/pull/280) | Support live Responses previous_response_id reuse |
| [ ] | [#105](https://github.com/antirez/ds4/pull/105) | server: reserve context budget for DSML tool calls |
| [ ] | [#104](https://github.com/antirez/ds4/pull/104) | Harden DSML and JSON parsing in the server |
| [ ] | [#245](https://github.com/antirez/ds4/pull/245) / [#227](https://github.com/antirez/ds4/pull/227) / [#224](https://github.com/antirez/ds4/pull/224) | SSE keepalive during prefill/decode fixes |
| [ ] | [#302](https://github.com/antirez/ds4/pull/302) | Structured outputs (JSON etc.) via llguidance |
| [ ] | [#230](https://github.com/antirez/ds4/pull/230) | Continuous batching |
| [ ] | [#401](https://github.com/antirez/ds4/pull/401) | Disaggregated Architecture for LLM Serving |
| [ ] | [#430](https://github.com/antirez/ds4/pull/430) | Add reverse distributed topology with coordinator-owned output suffix |
| [ ] | [#351](https://github.com/antirez/ds4/pull/351) | Add JACCL expert-parallel distributed inference |

### KV Cache / Sessions

| St | PR | Title |
|----|-----|-------|
| [ ] | [#448](https://github.com/antirez/ds4/pull/448) | Protect incoming KV prefix during live miss |
| [ ] | [#378](https://github.com/antirez/ds4/pull/378) | Keep live KV reusable when clients strip transient metadata blocks |
| [ ] | [#353](https://github.com/antirez/ds4/pull/353) | Avoid writing local KV cache payloads twice |
| [ ] | [#327](https://github.com/antirez/ds4/pull/327) | checkpoint cache support with optional tail-cache |
| [ ] | [#67](https://github.com/antirez/ds4/pull/67) | feat: multi-session pool with zero-wait switching between agents and context windows |
| [ ] | [#146](https://github.com/antirez/ds4/pull/146) | server: guard disk KV cache with CRC32C |
| [ ] | [#134](https://github.com/antirez/ds4/pull/134) | Guard KV cache against page-cache pressure |
| [ ] | [#186](https://github.com/antirez/ds4/pull/186) | Compressed KV disk cache via streaming LZ4 |
| [ ] | [#243](https://github.com/antirez/ds4/pull/243) | Add opt-in KV cache compression (turbo3 / turbo4 / comp_cache + HISA indexer) |
| [ ] | [#258](https://github.com/antirez/ds4/pull/258) | Add HISA hierarchical indexer for long-context decode |
| [ ] | [#265](https://github.com/antirez/ds4/pull/265) | RFC: Planar3 KV-cache quantization for compressed attention (experimental) |

### Agent

| St | PR | Title |
|----|-----|-------|
| [ ] | [#380](https://github.com/antirez/ds4/pull/380) | feat: add native Agent Skills support to ds4-agent |
| [ ] | [#379](https://github.com/antirez/ds4/pull/379) | feat: add runtime file discovery chain |
| [ ] | [#373](https://github.com/antirez/ds4/pull/373) | Fix agent edit: accept [upto] markers indented or padded with blanks |
| [ ] | [#421](https://github.com/antirez/ds4/pull/421) | agent: reject edit calls whose new= text contains [upto] |
| [ ] | [#391](https://github.com/antirez/ds4/pull/391) | Add teaching mode to ds4-agent, with teach-bench benchmark |
| [ ] | [#354](https://github.com/antirez/ds4/pull/354) | Add Agent Client Protocol v1 support to ds4-agent |
| [ ] | [#345](https://github.com/antirez/ds4/pull/345) | Add headless agent control mode |
| [ ] | [#250](https://github.com/antirez/ds4/pull/250) | Add guarded Git support to ds4-agent |
| [ ] | [#252](https://github.com/antirez/ds4/pull/252) | Add KV-backed agent context checkpoints |
| [ ] | [#479](https://github.com/antirez/ds4/pull/479) | feat: add headless browser support with curl fallback for web tools |
| [ ] | [#323](https://github.com/antirez/ds4/pull/323) | Echo ds4-agent prompts and slash commands in scrollback |
| [ ] | [#336](https://github.com/antirez/ds4/pull/336) | Default save prompt and wrap assistant prose |
| [ ] | [#241](https://github.com/antirez/ds4/pull/241) | agent: resolve runtime assets from binary directory |
| [ ] | [#215](https://github.com/antirez/ds4/pull/215) | fix: handle CR in agent yes/no prompts |
| [ ] | [#310](https://github.com/antirez/ds4/pull/310) | Add ds4-onboarding skill (docs-grounded agent skill tutor) |
| [ ] | [#490](https://github.com/antirez/ds4/pull/490) | Draft of Inference Sandbox jj-dai.org |
| [ ] | [#382](https://github.com/antirez/ds4/pull/382) | Harden CDP page target creation |

### Steering / Sampler / Quality

| St | PR | Title |
|----|-----|-------|
| [ ] | [#148](https://github.com/antirez/ds4/pull/148) | server: add tool-safe directional steering policy |
| [ ] | [#168](https://github.com/antirez/ds4/pull/168) | Steering MoE LLMs via Expert (De)Activation |
| [ ] | [#282](https://github.com/antirez/ds4/pull/282) | Teacher-Forced Directional Steering |
| [ ] | [#195](https://github.com/antirez/ds4/pull/195) | opt-in sampler-level repetition guard |
| [ ] | [#197](https://github.com/antirez/ds4/pull/197) | fix: sanitize user content against special-token injection in chat prompt rendering |

### Model Support / Quantization

| St | PR | Title |
|----|-----|-------|
| [ ] | [#474](https://github.com/antirez/ds4/pull/474) | Support DeepSeek V4 Flash 4Expert (top-4) |
| [ ] | [#368](https://github.com/antirez/ds4/pull/368) | [1/N] add fp8 fp32 scale support for custom RL model |
| [ ] | [#72](https://github.com/antirez/ds4/pull/72) | Sub 2bit iq1s down |
| [ ] | [#281](https://github.com/antirez/ds4/pull/281) | feat: add REAP-compact GGUF support |
| [ ] | [#347](https://github.com/antirez/ds4/pull/347) | Remove hardcoded number of output groups and rank |

### CLI / UX / Misc

| St | PR | Title |
|----|-----|-------|
| [ ] | [#464](https://github.com/antirez/ds4/pull/464) | Fix slow decodes "poisoning" sleep times when using power throttling |
| [ ] | [#234](https://github.com/antirez/ds4/pull/234) | Validate CLI seed and server port arguments |
| [ ] | [#55](https://github.com/antirez/ds4/pull/55) | Keep the chat session when /ctx fails to recreate it |
| [ ] | [#54](https://github.com/antirez/ds4/pull/54) | Don't kill the REPL on /ctx with invalid input |
| [ ] | [#426](https://github.com/antirez/ds4/pull/426) | Handle modified Enter as newline in multiline linenoise |
| [ ] | [#322](https://github.com/antirez/ds4/pull/322) | help: fix terminal newline handling in raw-mode terminals |
| [ ] | [#119](https://github.com/antirez/ds4/pull/119) | flock: unlink lock file on release |
| [ ] | [#115](https://github.com/antirez/ds4/pull/115) | [codex] Make seeded tool IDs deterministic |
| [ ] | [#69](https://github.com/antirez/ds4/pull/69) | Make Linux CPU builds runnable |
| [ ] | [#305](https://github.com/antirez/ds4/pull/305) | cpu: add x86 AVX2/AVX512 SIMD fast paths for quantized MoE dot products |
| [ ] | [#400](https://github.com/antirez/ds4/pull/400) | Extract dashboard into standalone ds4_dashboard module |
| [ ] | [#307](https://github.com/antirez/ds4/pull/307) | bench: add routed expert locality profiler |
| [ ] | [#204](https://github.com/antirez/ds4/pull/204) | test: run logprob vectors through quality path |
| [ ] | [#434](https://github.com/antirez/ds4/pull/434) | Fix quality-score link after streaming refactor |

### Tooling / Docs

| St | PR | Title |
|----|-----|-------|
| [ ] | [#248](https://github.com/antirez/ds4/pull/248) | download_model.sh: prefer huggingface CLI when available |
| [ ] | [#225](https://github.com/antirez/ds4/pull/225) | download_model.sh: auto-resume interrupted downloads |
| [ ] | [#26](https://github.com/antirez/ds4/pull/26) | Refactor fetch_official_vectors.py |
| [ ] | [#223](https://github.com/antirez/ds4/pull/223) | Rename AGENT.md to AGENTS.md, add CLAUDE.md symlink |
| [ ] | [#100](https://github.com/antirez/ds4/pull/100) / [#453](https://github.com/antirez/ds4/pull/453) | Fix typos in README |
| [ ] | [#438](https://github.com/antirez/ds4/pull/438) | Add Quickstart section to README |
| [ ] | [#356](https://github.com/antirez/ds4/pull/356) | docs: add zfs instructions |
| [ ] | [#268](https://github.com/antirez/ds4/pull/268) | Document wide MoE prefill tile setting |
| [ ] | [#443](https://github.com/antirez/ds4/pull/443) | AGENTS.md rename (and server performance improvements?) |

---

## Not Applicable (Metal / ROCm / macOS / Windows specific)

These are pre-marked `skip` — already determined irrelevant for the CUDA-only fork.

| St | PR | Title | Reason |
|----|-----|-------|--------|
| skip | [#498](https://github.com/antirez/ds4/pull/498) | rocm: use numerically stable SiLU | ROCm-specific (but SiLU NaN may apply; check separately) |
| skip | [#499](https://github.com/antirez/ds4/pull/499) | Opt-in mlock of non-routed weights for SSD streaming (DS4_MLOCK_NONROUTED) | Already handled in fork |
| skip | [#484](https://github.com/antirez/ds4/pull/484) | Enable MI300X ROCm support | ROCm |
| skip | [#461](https://github.com/antirez/ds4/pull/461) | ROCm: discrete GPU memory management | ROCm |
| skip | [#451](https://github.com/antirez/ds4/pull/451) | Support SSD streaming for Q4_K routed experts on ROCm | ROCm |
| skip | [#446](https://github.com/antirez/ds4/pull/446) | Fix ROCm Q8→F16 cache reserve | ROCm |
| skip | [#407](https://github.com/antirez/ds4/pull/407) | rocm: fix distributed inference on unified-memory APUs | ROCm |
| skip | [#365](https://github.com/antirez/ds4/pull/365) | make: consistent ROCm targets | ROCm |
| skip | [#362](https://github.com/antirez/ds4/pull/362) | win: native Windows (AMD HIP SDK) ROCm + MinGW CPU build | ROCm / Windows |
| skip | [#361](https://github.com/antirez/ds4/pull/361) | rocm: DS4_CUDA_MANAGED opt-in | ROCm |
| skip | [#383](https://github.com/antirez/ds4/pull/383) | ds4_agent: fix rocm support | ROCm |
| skip | [#420](https://github.com/antirez/ds4/pull/420) | Metal: protect tensor alloc/free byte counters with a mutex | Metal |
| skip | [#418](https://github.com/antirez/ds4/pull/418) / [#416](https://github.com/antirez/ds4/pull/416) | Metal: FP8-packed compressed-KV cache | Metal |
| skip | [#92](https://github.com/antirez/ds4/pull/92) | Fix Metal startup when default device is unavailable | Metal |
| skip | [#73](https://github.com/antirez/ds4/pull/73) | fix(ds4): Implement MoE low-memory streaming to work around macOS's kernel bug | Mac/Metal |
| skip | [#276](https://github.com/antirez/ds4/pull/276) | Tune suffix decoding defaults for M5 Max Metal | Metal |
| skip | [#274](https://github.com/antirez/ds4/pull/274) | Add DS4_EMBED_KERNELS flag to embed Metal kernel sources | Metal |
| skip | [#271](https://github.com/antirez/ds4/pull/271) | Makefile: use -march=native on x86_64 Darwin | Mac/Darwin |
| skip | [#454](https://github.com/antirez/ds4/pull/454) | Metal: keep selected-address SSD prefill opt-in by default | Metal |
| skip | [#83](https://github.com/antirez/ds4/pull/83) | Pooling VRAM between Macs | Metal |
| skip | [#395](https://github.com/antirez/ds4/pull/395) | Add DwarfStar logo SVG | Branding — up to you |
| skip | [#283](https://github.com/antirez/ds4/pull/283) | llguidance integration (overlaps with #302) | Subsumed by #302 |
| skip | [#207](https://github.com/antirez/ds4/pull/207) | Fix wording in agent feature description | Trivial wording |
| skip | [#196](https://github.com/antirez/ds4/pull/196) | docs: explain why README recommends openai-completions for Pi agent client | Docs-only |
| skip | [#93](https://github.com/antirez/ds4/pull/93) | README.md: add usage instructions for swival.dev | Docs-only |
| skip | [#103](https://github.com/antirez/ds4/pull/103) / [#97](https://github.com/antirez/ds4/pull/97) / [#89](https://github.com/antirez/ds4/pull/89) / [#143](https://github.com/antirez/ds4/pull/143) / [#255](https://github.com/antirez/ds4/pull/255) / [#324](https://github.com/antirez/ds4/pull/324) / [#256](https://github.com/antirez/ds4/pull/256) / [#107](https://github.com/antirez/ds4/pull/107) / [#413](https://github.com/antirez/ds4/pull/413) | Speed benchmark reports (Mac Studio / M5 Max / RTX 6000 / M3 Ultra / RTX PRO 6000) | Docs/benchmarks only |

---

## Top-priority watchlist

1. **[#158](https://github.com/antirez/ds4/pull/158)** — HMM fix for GB10 NVLink-C2C
2. **[#472](https://github.com/antirez/ds4/pull/472)** / **[#487](https://github.com/antirez/ds4/pull/487)** — Memory pressure / OOM fixes for large models on 121 GB
3. **[#466](https://github.com/antirez/ds4/pull/466)** / **[#435](https://github.com/antirez/ds4/pull/435)** / **[#478](https://github.com/antirez/ds4/pull/478)** — MoE router generalization and Pro correctness
4. **[#349](https://github.com/antirez/ds4/pull/349)** / **[#497](https://github.com/antirez/ds4/pull/497)** / **[#460](https://github.com/antirez/ds4/pull/460)** / **[#153](https://github.com/antirez/ds4/pull/153)** — SSD streaming and weight caching
5. **[#121](https://github.com/antirez/ds4/pull/121)** — Blackwell matmul path skip (sm_121 is Blackwell)
6. **[#494](https://github.com/antirez/ds4/pull/494)** — Older CUDA toolkit compat (reduces friction)
7. **[#489](https://github.com/antirez/ds4/pull/489)** / **[#269](https://github.com/antirez/ds4/pull/269)** — Server robustness and auth
8. **[#197](https://github.com/antirez/ds4/pull/197)** — Special-token injection sanitization (security)
