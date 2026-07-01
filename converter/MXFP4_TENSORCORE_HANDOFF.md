# MXFP4 tensor-core MoE integration — state & remaining plan

Goal: run the 6 rich MXFP4 expert layers on GB10 tensor cores (CUTLASS `MXF8F6F4`, sm_120f) to
speed PREFILL (baseline 68 t/s). Keep MXFP4 (byte-lossless from source); NVIDIA hardware via CUTLASS,
not cuBLASLt (which only exposes NVFP4). One CUTLASS path for prefill + decode; dp4a to be deleted.
Nothing at runtime that isn't required — weights fully repacked OFFLINE; only activation packing is runtime.

## VALIDATED (all GB10 sm_120f, exact/Passed) — see temp/p0/
- `ds4_mxfp4_cutlass.cu` — THE engine TU. Contains:
  - `extern "C" ds4_cutlass_expert_ffn(out,x, Wg_d,Wg_sf, Wu_d,Wu_sf, Wd_d,Wd_sf, weights,clamp, T,in,mid,out)`
    — pack act (device, RowMajor A) → gate/up GEMM → SwiGLU(clamp+silu+routing weight, matches
    ds4_cuda.cu:10827-10835) → pack mid → down GEMM. f32 out. **max_rel=0 vs float ref.**
  - `ds4_cutlass_pack_source(Bd,Bsf, e2m1[N,K/2], e8m0[N,K/32], N,K)` — SOURCE arrays → CUTLASS B
    (ColumnMajor packed E2M1 + swizzled SFB). **data=MATCH sf=MATCH** vs direct packer. Host-side.
  - `ds4_cutlass_weight_sf_count(N,K)` — SF element count for allocation.
  - device `pack_act_rowmajor` (activation packer) + `swiglu_kernel`.
  - Guards: `-DDS4_MXFP4_STANDALONE` (float FFN test), `-DDS4_MXFP4_REPACK_CLI` (source→CUTLASS CLI).
- CLI built: `temp/p0/mxfp4_pack_source_cli  e2m1.bin e8m0.bin N K  out_data.bin out_sf.bin`
- Build integration DONE: engine compiles whole at sm_120f + links the TU (`ds4-server` has the symbols).
  Makefile: added `ds4_mxfp4_cutlass.o` to Linux CORE_OBJS + a rule (CUTLASS_INC, -std=c++17,
  --expt-relaxed-constexpr/-lambda). Build with `make -B ds4-server CUDA_ARCH=sm_120f`.

## KEY FACTS
- CUTLASS block-scaled GEMM is TN-only → weight MUST be ColumnMajor B (transpose from source row-major);
  activation is RowMajor A (device-packed at runtime, race-free). SF tensor indexed by FULL k (tile_atom
  folds the 32-group); scale at k=kb*32. build `-arch=sm_120f` (family; __CUDA_ARCH__==1200 guard).
- Rich layers: 30,34,37,39,41,42. GGUF tensors `blk.{il}.ffn_{gate,up,down}_exps.weight`, type 39, dims
  [K,N,256]: gate/up [4096,2048,256] (K=in=4096,N=mid=2048); down [2048,4096,256] (K=mid=2048,N=out=4096).
- SOURCE (pve2 CT117 /work/src/DeepSeek-V4-Flash, x86, NO nvcc/cutlass): per-expert
  `layers.{il}.ffn.experts.{0..255}.w{1,2,3}.weight` (E2M1 I8 [out,in/2]) + `.scale` (E8M0 [out,in/32]).
  Mapping gate=w1, up=w3, down=w2. Extraction script: pve2 `/work/tmp/pve_extract.py` (currently emits
  17-byte; needs a variant emitting separate E2M1+E8M0, OR emit and de-interleave).
- Engine MoE contract: mid = silu(clamp(gate))*clamp(up)*routing_weight; out = Σ_experts down_e (plain sum,
  weight already in mid). Gather infra exists: moe_count/prefix/scatter_sorted_pairs → sorted_pairs/offsets/counts.

## CONVERTER DONE (2026-06-30): oracle-cutlass-mxfp4-99gb.gguf built + verified
- Source = DSpark variant on /mnt/pve1-fast (25G NFS; experts byte-identical to base/oracle-zeroq8, GATE ALL MATCH).
  Draft/MTP differs in DSpark but we only extract ffn.experts, so it's safe.
- Pipeline (all Sparky): pve_extract_source.py (per rich layer, reads DSpark shard -> mxfp4_src/L{il}_{tgt}.{e2m1,e8m0})
  -> splice_cutlass_mxfp4.py (calls `mxfp4_pack_source_cli --stacked` per tensor -> type-40 GGUF, streams oracle-zeroq8 base).
- **TYPE 40 = CUTLASS MXFP4.** GGUF dims [K,N,n_expert] (gate/up K=4096 N=2048; down K=2048 N=4096; ne=256).
  Per-expert layout EXPERT-MAJOR: data (N*K/2 = 4,194,304 B, ColumnMajor packed E2M1) THEN SF (262,144 B, swizzled E8M0).
  Per-expert stride = 4,456,448 B. Tensor bytes = ne * 4,456,448. Engine loader must compute size this way for type 40,
  and give routed_moe_launch: expert e data @ off + e*stride, SF @ off + e*stride + N*K/2. sf_count = ds4_cutlass_weight_sf_count(N,K)=262144 (both shapes).
- Infra: pve1 /rpool/models export got 10.20.40.0/24 added (25G); /mnt/pve1-fast nconnect=16 (storage-capped ~1GB/s).
  Freed disk by deleting oracle-mxfp4rich. Intermediate: mxfp4_src/ (20GB) + cutlass_blobs/ (20GB) — deletable.

## REMAINING (engine-side, serve-validated)
1. CONVERTER (source→CUTLASS GGUF): pve2 emits per-rich-tensor stacked E2M1+E8M0 (256 experts) → transfer
   to Sparky (~20GB) → per expert run pack_source CLI → concat (data+SF per expert) → splice into new GGUF
   with a NEW type tag (e.g. 40); store data then SF per expert. Reuse splice_main.py streaming pattern.
2. ENGINE LOADER: recognize type 40; tensor bytes = data(N*K/2) + SF(sf_count) per expert; expose both
   offsets to routed_moe_launch.
3. ENGINE CALL-SITE (routed_moe_launch, flag-gated, serve-validated): for type-40 layers build sorted_pairs,
   per expert gather its tokens' x rows + routing weights, call ds4_cutlass_expert_ffn (weights = mmap'd
   data/SF), scatter-add down into out. Covers prefill (n_tok>1) AND decode. Delete dp4a mmvq after.
   Use caller-provided scratch (FFN currently cudaMallocs per call — fix for the engine).
4. SERVE + MEASURE: GB10 teardown, serve oracle GGUF, coherence-gate, prefill/decode t/s vs 68 baseline.

## Notes
- Test binaries in job tmp: validate2, devpack, ffn3/4, src_test, mxfp4_grouped2, fp4_cublaslt_probe.
- CUTLASS cloned at ~/Projects/AI/cutlass. Engine tree diff so far: Makefile + ds4_mxfp4_cutlass.cu (new).
