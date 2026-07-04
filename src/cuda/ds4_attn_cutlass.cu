// ds4_attn_cutlass.cu — CUTLASS block-scaled MX GEMM engine for fused attention (sm_120/121).
//
// Provides the two matmuls a flash-attention needs, with microscaling (MX) operands:
//   QK^T : scores[Q,KV] = Qmx[Q,D] . Kmx[KV,D]     (contraction D = head_dim)
//   P V  : out[Q,D]     = Pmx[Q,KV] . Vmx[D,KV]    (contraction KV = key tile)
// Both are the CUTLASS convention D[M,N] = A[M,K] . B[N,K] (B ColumnMajor). Q and P are
// packed to MXFP8; K and V come from the KV cache as MXFP8 (528 B/row) or MXFP4 (272 B/row),
// so the GEMM is symmetric (mxfp8 x mxfp8) or asymmetric (mxfp8 x mxfp4) — both confirmed
// supported by can_implement on Sm120. The swizzled scale-factor (SF) layout is built via
// Sm1xxBlkScaledConfig exactly as the proven MoE path (ds4_mxfp4_cutlass.cu) does.
//
// Standalone parity test:
//   nvcc -std=c++17 -arch=sm_121 --expt-relaxed-constexpr --expt-extended-lambda \
//        -DDS4_ATTN_CUTLASS_STANDALONE -I cutlass/include -I cutlass/tools/util/include \
//        src/cuda/ds4_attn_cutlass.cu -o temp/attn_cutlass_test
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#ifndef DS4_ATTN_CUTLASS_STANDALONE
#include "ds4_cuda_internal.h"   // ds4_gpu_tensor (->ptr), cuda_model_range_ptr (engine build only)
#endif
using namespace cute;

using ElementSF = cutlass::float_ue8m0_t;

// ---- GEMM factory: D[M,N] f32 = A[M,K] (RowMajor) . B[N,K] (ColumnMajor), MX operands. ----
template <class EA, class EB>
struct MXG {
  using ElementA = EA; using ElementB = EB;
  using ElementD = float; using ElementC = float; using ElementAcc = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignA = 32, AlignB = 32;
  static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using TileShape = Shape<_128,_128,_128>;
  using ClusterShape = Shape<_1,_1,_1>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementAcc, ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Cfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};
using MX8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using MX4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using G88 = MXG<MX8, MX8>;   // Q/P (mxfp8) x K/V (mxfp8)
using G84 = MXG<MX8, MX4>;   // Q/P (mxfp8) x K/V (mxfp4)

template <class G>
static typename G::Gemm::Arguments mx_args(float *D, const uint8_t *Ad, const ElementSF *Asf,
                                            const uint8_t *Bd, const ElementSF *Bsf,
                                            int M, int N, int K) {
  auto sA = cutlass::make_cute_packed_stride(typename G::Kernel::StrideA{}, {M,K,1});
  auto sB = cutlass::make_cute_packed_stride(typename G::Kernel::StrideB{}, {N,K,1});
  auto sC = cutlass::make_cute_packed_stride(typename G::Kernel::StrideC{}, {M,N,1});
  auto sD = cutlass::make_cute_packed_stride(typename G::Kernel::StrideD{}, {M,N,1});
  auto lSFA = G::Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB = G::Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  return typename G::Gemm::Arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
    { reinterpret_cast<const typename G::ElementA::DataType*>(Ad), sA,
      reinterpret_cast<const typename G::ElementB::DataType*>(Bd), sB,
      Asf, lSFA, Bsf, lSFB },
    { {1.0f,0.0f}, D, sC, D, sD } };
}

template <class G>
static int mx_run(float *D, const uint8_t *Ad, const ElementSF *Asf,
                  const uint8_t *Bd, const ElementSF *Bsf, int M, int N, int K, void *ws) {
  auto args = mx_args<G>(D, Ad, Asf, Bd, Bsf, M, N, K);
  typename G::Gemm g;
  if (g.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (g.initialize(args, ws) != cutlass::Status::kSuccess) return 2;
  return g.run() == cutlass::Status::kSuccess ? 0 : 3;
}

template <class G>
static size_t mx_ws_bytes(int M, int N, int K) {
  return G::Gemm::get_workspace_size(mx_args<G>(nullptr,nullptr,nullptr,nullptr,nullptr,M,N,K));
}

// ---- device MX encoders (E4M3 / E2M1 / E8M0), matching ds4_cuda_norm_kv.cu ----
__device__ __forceinline__ float a_e4m3_val(int i) {
  int e = (i >> 3) & 15, m = i & 7;
  return e == 0 ? (float)m * 0.001953125f : (1.0f + (float)m * 0.125f) * exp2f((float)e - 7.0f);
}
__device__ __forceinline__ uint8_t a_e4m3_enc(float x) {
  float ax = fminf(fabsf(x), 448.0f);
  int lo = 0, hi = 126;
  while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (a_e4m3_val(mid) <= ax) lo = mid; else hi = mid - 1; }
  int b = lo;
  if (b < 126) { float bd = fabsf(ax - a_e4m3_val(b)), nd = fabsf(ax - a_e4m3_val(b + 1)); if (nd < bd) b++; }
  return (uint8_t)(b | ((x < 0.0f) ? 0x80 : 0));
}
__device__ __constant__ float a_kE2M1[16] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
__device__ __forceinline__ uint8_t a_e2m1_enc(float v) {
  float best = 1e30f; uint8_t bn = 0;
  for (uint8_t n = 0; n < 16; n++) { float d = fabsf(v - a_kE2M1[n]); if (d < best) { best = d; bn = n; } }
  return bn;
}
__device__ __forceinline__ uint8_t a_e8m0_enc(float amax, float maxrep) {
  float r = fmaxf(amax, 1e-20f) / maxrep;
  int e = (int)ceilf(log2f(r)) + 127;
  if (e < 0) e = 0; if (e > 254) e = 254;
  return (uint8_t)e;
}

// Pack A (RowMajor [M,K]) to MXFP8: plain E4M3 data bytes + swizzled SFA. One thread per (m,kb).
template <class TSF>
__global__ void pack_a8(uint8_t *Ad, TSF tSFA, const float *x, int M, int K) {
  int nblk = K / 32; long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)M * nblk) return;
  int m = (int)(idx / nblk), kb = (int)(idx % nblk);
  const float *xr = x + (size_t)m * K + (size_t)kb * 32;
  float mx = 0.f; for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(xr[i]));
  uint8_t e8 = a_e8m0_enc(mx, 448.0f); float sc = exp2f((float)((int)e8 - 127));
  uint8_t *outb = Ad + (size_t)m * K + (size_t)kb * 32;
  for (int i = 0; i < 32; i++) outb[i] = a_e4m3_enc(xr[i] / sc);
  tSFA(m, kb * 32, 0) = ElementSF::bitcast(e8);
}
// Pack B (ColumnMajor [N,K]) to MXFP8: E4M3 data via cute tensor tB + swizzled SFB.
template <class TB, class TSF>
__global__ void pack_b8(TB tB, TSF tSFB, const float *x, int N, int K) {
  int nblk = K / 32; long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)N * nblk) return;
  int n = (int)(idx / nblk), kb = (int)(idx % nblk);
  const float *xr = x + (size_t)n * K + (size_t)kb * 32;
  float mx = 0.f; for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(xr[i]));
  uint8_t e8 = a_e8m0_enc(mx, 448.0f); float sc = exp2f((float)((int)e8 - 127));
  for (int i = 0; i < 32; i++) tB(n, kb * 32 + i, 0) = cutlass::float_e4m3_t::bitcast(a_e4m3_enc(xr[i] / sc));
  tSFB(n, kb * 32, 0) = ElementSF::bitcast(e8);
}
// Pack B (ColumnMajor [N,K]) to MXFP4: E2M1 nibbles via cute tensor tB + swizzled SFB.
template <class TB, class TSF>
__global__ void pack_b4(TB tB, TSF tSFB, const float *x, int N, int K) {
  int nblk = K / 32; long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)N * nblk) return;
  int n = (int)(idx / nblk), kb = (int)(idx % nblk);
  const float *xr = x + (size_t)n * K + (size_t)kb * 32;
  float mx = 0.f; for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(xr[i]));
  uint8_t e8 = a_e8m0_enc(mx, 6.0f); float sc = exp2f((float)((int)e8 - 127));
  for (int i = 0; i < 32; i++) tB(n, kb * 32 + i, 0) = cutlass::float_e2m1_t::bitcast(a_e2m1_enc(xr[i] / sc));
  tSFB(n, kb * 32, 0) = ElementSF::bitcast(e8);
}

// Host-side pack helpers that build the swizzled SF cute tensors then launch.
static void mx_pack_a8(uint8_t *Ad, ElementSF *Asf, const float *x, int M, int K) {
  auto lSFA = G88::Cfg::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(Asf), lSFA);
  int nb = M * (K / 32), t = 128, b = (nb + t - 1) / t;
  pack_a8<<<b, t>>>(Ad, tSFA, x, M, K);
}
template <class G>
static void mx_pack_b8(uint8_t *Bd, ElementSF *Bsf, const float *x, int N, int K) {
  auto lB = cutlass::make_cute_packed_stride(typename G::Kernel::StrideB{}, {N,K,1});
  auto layB = make_layout(make_shape(N,K,1), lB);
  auto lSFB = G::Cfg::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
  auto tB = make_tensor(recast_ptr<cutlass::float_e4m3_t>(Bd), layB);
  auto tSFB = make_tensor(make_gmem_ptr(Bsf), lSFB);
  int nb = N * (K / 32), t = 128, b = (nb + t - 1) / t;
  pack_b8<<<b, t>>>(tB, tSFB, x, N, K);
}
template <class G>
static void mx_pack_b4(uint8_t *Bd, ElementSF *Bsf, const float *x, int N, int K) {
  auto lB = cutlass::make_cute_packed_stride(typename G::Kernel::StrideB{}, {N,K,1});
  auto layB = make_layout(make_shape(N,K,1), lB);
  auto lSFB = G::Cfg::tile_atom_to_shape_SFB(make_shape(1, N, K, 1));
  auto tB = make_tensor(recast_ptr<cutlass::float_e2m1_t>(Bd), layB);
  auto tSFB = make_tensor(make_gmem_ptr(Bsf), lSFB);
  int nb = N * (K / 32), t = 128, b = (nb + t - 1) / t;
  pack_b4<<<b, t>>>(tB, tSFB, x, N, K);
}
static size_t mx_sfa_count(int M, int K) {
  return cute::size(cute::filter_zeros(G88::Cfg::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1))));
}
template <class G>
static size_t mx_sfb_count(int N, int K) {
  return cute::size(cute::filter_zeros(G::Cfg::tile_atom_to_shape_SFB(make_shape(1, N, K, 1))));
}

// ---- Batched (L = n_head) variants: A/B/D are L contiguous matrices; SF layouts carry L. ----
template <class TSF>
__global__ void pack_a8_bat(uint8_t *Ad, TSF tSFA, const float *x, int M, int K, int L) {
  int nblk = K / 32; long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)L * M * nblk) return;
  int l = (int)(idx / ((long)M * nblk)); long r = idx % ((long)M * nblk);
  int m = (int)(r / nblk), kb = (int)(r % nblk);
  const float *xr = x + ((size_t)l * M + m) * K + (size_t)kb * 32;
  float mx = 0.f; for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(xr[i]));
  uint8_t e8 = a_e8m0_enc(mx, 448.f); float sc = exp2f((float)((int)e8 - 127));
  uint8_t *outb = Ad + ((size_t)l * M + m) * K + (size_t)kb * 32;
  for (int i = 0; i < 32; i++) outb[i] = a_e4m3_enc(xr[i] / sc);
  tSFA(m, kb * 32, l) = ElementSF::bitcast(e8);
}
template <class TB, class TSF>
__global__ void pack_b8_bat(TB tB, TSF tSFB, const float *x, int N, int K, int L) {
  int nblk = K / 32; long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long)L * N * nblk) return;
  int l = (int)(idx / ((long)N * nblk)); long r = idx % ((long)N * nblk);
  int n = (int)(r / nblk), kb = (int)(r % nblk);
  const float *xr = x + ((size_t)l * N + n) * K + (size_t)kb * 32;
  float mx = 0.f; for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(xr[i]));
  uint8_t e8 = a_e8m0_enc(mx, 448.f); float sc = exp2f((float)((int)e8 - 127));
  for (int i = 0; i < 32; i++) tB(n, kb * 32 + i, l) = cutlass::float_e4m3_t::bitcast(a_e4m3_enc(xr[i] / sc));
  tSFB(n, kb * 32, l) = ElementSF::bitcast(e8);
}
static void mx_pack_a8_bat(uint8_t *Ad, ElementSF *Asf, const float *x, int M, int K, int L) {
  auto lSFA = G88::Cfg::tile_atom_to_shape_SFA(make_shape(M, 0, K, L));
  auto tSFA = make_tensor(make_gmem_ptr(Asf), lSFA);
  long nb = (long)L * M * (K / 32), t = 128, b = (nb + t - 1) / t;
  pack_a8_bat<<<b, t>>>(Ad, tSFA, x, M, K, L);
}
template <class G>
static void mx_pack_b8_bat(uint8_t *Bd, ElementSF *Bsf, const float *x, int N, int K, int L) {
  auto lB = cutlass::make_cute_packed_stride(typename G::Kernel::StrideB{}, {N, K, L});
  auto layB = make_layout(make_shape(N, K, L), lB);
  auto lSFB = G::Cfg::tile_atom_to_shape_SFB(make_shape(1, N, K, L));
  auto tB = make_tensor(recast_ptr<cutlass::float_e4m3_t>(Bd), layB);
  auto tSFB = make_tensor(make_gmem_ptr(Bsf), lSFB);
  long nb = (long)L * N * (K / 32), t = 128, b = (nb + t - 1) / t;
  pack_b8_bat<<<b, t>>>(tB, tSFB, x, N, K, L);
}
template <class G>
static int mx_run_bat(float *D, const uint8_t *Ad, const ElementSF *Asf,
                      const uint8_t *Bd, const ElementSF *Bsf, int M, int N, int K, int L, void *ws) {
  auto sA = cutlass::make_cute_packed_stride(typename G::Kernel::StrideA{}, {M,K,L});
  auto sB = cutlass::make_cute_packed_stride(typename G::Kernel::StrideB{}, {N,K,L});
  auto sC = cutlass::make_cute_packed_stride(typename G::Kernel::StrideC{}, {M,N,L});
  auto sD = cutlass::make_cute_packed_stride(typename G::Kernel::StrideD{}, {M,N,L});
  auto lSFA = G::Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,L));
  auto lSFB = G::Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,L));
  typename G::Gemm::Arguments args{ cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,L},
    { reinterpret_cast<const typename G::ElementA::DataType*>(Ad), sA,
      reinterpret_cast<const typename G::ElementB::DataType*>(Bd), sB, Asf, lSFA, Bsf, lSFB },
    { {1.f,0.f}, D, sC, D, sD } };
  typename G::Gemm g;
  if (g.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (g.initialize(args, ws) != cutlass::Status::kSuccess) return 2;
  return g.run() == cutlass::Status::kSuccess ? 0 : 3;
}
template <class G>
static size_t mx_ws_bytes_bat(int M, int N, int K, int L) {
  auto sA=cutlass::make_cute_packed_stride(typename G::Kernel::StrideA{},{M,K,L});
  auto sB=cutlass::make_cute_packed_stride(typename G::Kernel::StrideB{},{N,K,L});
  auto sC=cutlass::make_cute_packed_stride(typename G::Kernel::StrideC{},{M,N,L});
  auto sD=cutlass::make_cute_packed_stride(typename G::Kernel::StrideD{},{M,N,L});
  auto lSFA=G::Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,L));
  auto lSFB=G::Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,L));
  typename G::Gemm::Arguments args{ cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,L},
    {(const typename G::ElementA::DataType*)nullptr,sA,(const typename G::ElementB::DataType*)nullptr,sB,(const ElementSF*)nullptr,lSFA,(const ElementSF*)nullptr,lSFB},
    {{1.f,0.f},(float*)nullptr,sC,(float*)nullptr,sD} };
  return G::Gemm::get_workspace_size(args);
}
static size_t mx_sfa_count_bat(int M, int K, int L) {
  return cute::size(cute::filter_zeros(G88::Cfg::tile_atom_to_shape_SFA(make_shape(M, 0, K, L))));
}
template <class G>
static size_t mx_sfb_count_bat(int N, int K, int L) {
  return cute::size(cute::filter_zeros(G::Cfg::tile_atom_to_shape_SFB(make_shape(1, N, K, L))));
}

// ---- MXFP8 decoders + K/V assembly for the decode attention entry point ----
__device__ __forceinline__ float a_e4m3_dec(uint8_t b, float scale) {
  float v = a_e4m3_val(b & 0x7f);
  return (b & 0x80) ? (-v * scale) : (v * scale);
}
__device__ __forceinline__ float a_e8m0_dec(uint8_t b) { return exp2f((float)((int)b - 127)); }

// Build the shared attention operands for one decode step: Kf [n_kv][head_dim]
// (raw f32 window rows then gathered MXFP8 compressed rows) and VbT [head_dim]
// [n_kv_pad] (V transposed for the PV GEMM, contraction = n_kv, zero-padded).
// comp caches are MXFP8 (block 32, E8M0), rowbytes = head_dim + head_dim/32.
__global__ void assemble_kv_mxfp8(float *Kf, float *VbT,
                                  const float *raw_k, const float *raw_v,
                                  const uint8_t *comp_k, const uint8_t *comp_v,
                                  const int32_t *comp_rows,
                                  uint32_t n_raw, uint32_t n_comp,
                                  uint32_t n_kv, uint32_t n_kv_pad, uint32_t head_dim) {
  uint32_t row = blockIdx.x;
  if (row >= n_kv_pad) return;
  const uint32_t nblk = head_dim / 32, rowbytes = head_dim + nblk;
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float kv = 0.f, vv = 0.f;
    if (row < n_raw) {
      kv = raw_k[(uint64_t)row * head_dim + d];
      vv = raw_v[(uint64_t)row * head_dim + d];
    } else if (row < n_kv) {
      int32_t r = comp_rows[row - n_raw];
      if (r >= 0) {
        const uint8_t *ck = comp_k + (uint64_t)r * rowbytes;
        const uint8_t *cv = comp_v + (uint64_t)r * rowbytes;
        kv = a_e4m3_dec(ck[d], a_e8m0_dec(ck[head_dim + d / 32]));
        vv = a_e4m3_dec(cv[d], a_e8m0_dec(cv[head_dim + d / 32]));
      }
    }
    Kf[(uint64_t)row * head_dim + d] = kv;                  // Kf is [n_kv_pad][head_dim] (pad=0)
    VbT[(uint64_t)d * n_kv_pad + row] = vv;                  // VbT is [head_dim][n_kv_pad] (pad=0)
  }
}

// ---- Softmax with attention sink (one block per query row) ----
// scores[n_q, n_kv] (raw dot products) -> P[n_q, n_kv] probabilities, applying
// scale = 1/sqrt(head_dim) and folding the per-query sink into the denominator
// only (exp(sink - m), no value contribution). `valid` bounds the visible keys
// per query (causal/window); keys >= valid[q] are masked out. sink is per-row
// here (single head); the batched path passes sinks[head].
// n_valid (>0) masks keys >= n_valid uniformly (used when n_kv is padded up to a
// multiple of 32 for the PV contraction); `valid` gives a per-row bound instead.
__global__ void attn_softmax_sink(float *P, const float *scores, const float *sink_per_row,
                                  const int32_t *valid, float scale, uint32_t n_q, uint32_t n_kv,
                                  uint32_t n_valid) {
  uint32_t q = blockIdx.x;
  if (q >= n_q) return;
  const float *sr = scores + (uint64_t)q * n_kv;
  float *pr = P + (uint64_t)q * n_kv;
  uint32_t lim = valid ? (uint32_t)valid[q] : (n_valid ? n_valid : n_kv);
  if (lim > n_kv) lim = n_kv;
  const float sink = sink_per_row ? sink_per_row[q] : -INFINITY;

  __shared__ float red[256];
  // pass 1: row max (seeded with sink)
  float m = sink;
  for (uint32_t j = threadIdx.x; j < lim; j += blockDim.x) m = fmaxf(m, sr[j] * scale);
  red[threadIdx.x] = m; __syncthreads();
  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
  m = red[0]; __syncthreads();
  // pass 2: denom = sum exp(score-m) + exp(sink-m)
  float d = 0.0f;
  for (uint32_t j = threadIdx.x; j < lim; j += blockDim.x) d += expf(sr[j] * scale - m);
  red[threadIdx.x] = d; __syncthreads();
  for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
  d = red[0] + expf(sink - m);
  const float inv = d > 0.0f ? 1.0f / d : 0.0f;
  // write probabilities (masked keys -> 0)
  for (uint32_t j = threadIdx.x; j < n_kv; j += blockDim.x) pr[j] = (j < lim) ? expf(sr[j] * scale - m) * inv : 0.0f;
}

// ---- Decode attention entry point (MXFP8 KV) ----
// One decode step: n_head query rows attend over the SHARED KV = raw f32 window
// (n_raw rows) ++ gathered MXFP8 compressed rows (comp_rows[0..n_comp)). Produces
// heads[n_head][head_dim]. comp_k/comp_v are MXFP8 caches (row stride hd + hd/32).
//
// Scratch layout, sized at a max (n_head, head_dim, n_kv_pad) once and reused: the
// hot path never touches the CUDA allocator (mirrors the MoE FFN scratch pattern).
static size_t au(size_t n, size_t a) { return (n + a - 1) / a * a; }
struct attn_decode_scratch {
  size_t Kf, Vb, scores, P, Qd, Qs, Kd, Ks, Pd, Ps, Vd, Vs, w1, w2, total;
  size_t qa, kb, pa, vb, ws1, ws2;   // element/byte counts needed for memsets/zeroing
};
static attn_decode_scratch attn_decode_layout(int M, int D, int KV) {
  attn_decode_scratch L{}; const size_t A = 256; size_t o = 0;
  L.qa = mx_sfa_count(M, D); L.kb = mx_sfb_count<G88>(KV, D);
  L.pa = mx_sfa_count(M, KV); L.vb = mx_sfb_count<G88>(D, KV);
  L.ws1 = mx_ws_bytes<G88>(M, KV, D); L.ws2 = mx_ws_bytes<G88>(M, D, KV);
  L.Kf = o; o = au(o + (size_t)KV * D * 4, A);
  L.Vb = o; o = au(o + (size_t)D * KV * 4, A);
  L.scores = o; o = au(o + (size_t)M * KV * 4, A);
  L.P = o; o = au(o + (size_t)M * KV * 4, A);
  L.Qd = o; o = au(o + (size_t)M * D, A);       L.Qs = o; o = au(o + L.qa * 2, A);
  L.Kd = o; o = au(o + (size_t)KV * D, A);       L.Ks = o; o = au(o + L.kb * 2, A);
  L.Pd = o; o = au(o + (size_t)M * KV, A);       L.Ps = o; o = au(o + L.pa * 2, A);
  L.Vd = o; o = au(o + (size_t)D * KV, A);       L.Vs = o; o = au(o + L.vb * 2, A);
  L.w1 = o; o = au(o + L.ws1, A);                L.w2 = o; o = au(o + L.ws2, A);
  L.total = o; return L;
}
// Bytes of scratch for the worst case (max heads / head_dim / padded KV).
extern "C" size_t ds4_attn_mx_decode_scratch_bytes(uint32_t n_head, uint32_t head_dim, uint32_t max_n_kv) {
  uint32_t kvp = (max_n_kv + 31u) & ~31u; if (kvp == 0) kvp = 32u;
  return attn_decode_layout((int)n_head, (int)head_dim, (int)kvp).total;
}
// No allocation, no synchronization: launches on the caller's stream. The caller
// owns `scratch` (>= ds4_attn_mx_decode_scratch_bytes for its max shape) and orders.
extern "C" int ds4_attn_mx_decode_scratch(
    float *heads, const float *q,
    const float *raw_k, const float *raw_v,
    const uint8_t *comp_k, const uint8_t *comp_v, const int32_t *comp_rows,
    const float *sinks,
    uint32_t n_head, uint32_t head_dim, uint32_t n_raw, uint32_t n_comp,
    uint8_t *scratch, size_t scratch_bytes) {
  const uint32_t n_kv = n_raw + n_comp;
  if (n_kv == 0 || (head_dim % 32) != 0 || !scratch) return -1;
  const uint32_t n_kv_pad = (n_kv + 31u) & ~31u;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int M = (int)n_head, D = (int)head_dim, KV = (int)n_kv_pad;
  attn_decode_scratch L = attn_decode_layout(M, D, KV);
  if (scratch_bytes < L.total) return -2;
  float   *Kf = (float *)(scratch + L.Kf), *Vb = (float *)(scratch + L.Vb);
  float   *scores = (float *)(scratch + L.scores), *P = (float *)(scratch + L.P);
  uint8_t *Qd = scratch + L.Qd, *Kd = scratch + L.Kd, *Pd = scratch + L.Pd, *Vd = scratch + L.Vd;
  ElementSF *Qs = (ElementSF *)(scratch + L.Qs), *Ks = (ElementSF *)(scratch + L.Ks);
  ElementSF *Ps = (ElementSF *)(scratch + L.Ps), *Vs = (ElementSF *)(scratch + L.Vs);
  void *w1 = L.ws1 ? (void *)(scratch + L.w1) : nullptr, *w2 = L.ws2 ? (void *)(scratch + L.w2) : nullptr;
  // swizzled SF tensors can have physical holes -> zero before packing (as the MoE path does)
  cudaMemsetAsync(Qs, 0, L.qa * 2); cudaMemsetAsync(Ks, 0, L.kb * 2);
  cudaMemsetAsync(Ps, 0, L.pa * 2); cudaMemsetAsync(Vs, 0, L.vb * 2);
  assemble_kv_mxfp8<<<n_kv_pad, 256>>>(Kf, Vb, raw_k, raw_v, comp_k, comp_v, comp_rows,
                                       n_raw, n_comp, n_kv, n_kv_pad, head_dim);
  mx_pack_a8(Qd, Qs, q, M, D);
  mx_pack_b8<G88>(Kd, Ks, Kf, KV, D);
  mx_pack_b8<G88>(Vd, Vs, Vb, D, KV);
  int rc = mx_run<G88>(scores, Qd, Qs, Kd, Ks, M, KV, D, w1);              // QK^T -> scores[M][KV]
  attn_softmax_sink<<<M, 256>>>(P, scores, sinks, nullptr, scale, M, n_kv_pad, n_kv);  // full-width write; mask pad
  mx_pack_a8(Pd, Ps, P, M, KV);
  rc |= mx_run<G88>(heads, Pd, Ps, Vd, Vs, M, D, KV, w2);                  // P.V -> heads[M][D]
  return rc;
}
// f32-cache assembly (compute-path validation; no cache-format change). This
// model shares one KV latent per position (K==V) stored f32, so raw_cache and
// comp_cache are each the KV and feed both Kf and VbT. raw_cache is a ring
// (physical row = (raw_start + raw_first + r) % raw_cap); comp_rows selects
// compressed rows (NULL = identity 0..n_comp for the static/non-indexed path).
__global__ void assemble_kv_f32comp(float *Kf, float *VbT,
                                    const float *kv_raw, const float *kv_comp, const int32_t *comp_rows,
                                    uint32_t raw_start, uint32_t raw_cap, uint32_t raw_first,
                                    uint32_t n_raw, uint32_t n_kv, uint32_t n_kv_pad, uint32_t head_dim) {
  uint32_t row = blockIdx.x;
  if (row >= n_kv_pad) return;
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float v = 0.f;
    if (row < n_raw) {
      uint32_t phys = (raw_start + raw_first + row) % raw_cap;
      v = kv_raw[(uint64_t)phys * head_dim + d];
    } else if (row < n_kv) {
      uint32_t i = row - n_raw;
      int32_t r = comp_rows ? comp_rows[i] : (int32_t)i;
      if (r >= 0) v = kv_comp[(uint64_t)r * head_dim + d];
    }
    Kf[(uint64_t)row * head_dim + d] = v;
    VbT[(uint64_t)d * n_kv_pad + row] = v;
  }
}

// Phase B assembly: raw is f32 (ring), comp is a PERSISTENT MXFP8 cache (row
// stride head_dim + head_dim/32, block-32 E8M0), dequantized on the fly. K==V.
__global__ void assemble_kv_mxcomp(float *Kf, float *VbT,
                                   const float *kv_raw, const uint8_t *kv_comp_mx, const int32_t *comp_rows,
                                   uint32_t raw_start, uint32_t raw_cap, uint32_t raw_first,
                                   uint32_t n_raw, uint32_t n_kv, uint32_t n_kv_pad, uint32_t head_dim) {
  uint32_t row = blockIdx.x;
  if (row >= n_kv_pad) return;
  const uint32_t nblk = head_dim / 32, rowbytes = head_dim + nblk;
  for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float v = 0.f;
    if (row < n_raw) {
      uint32_t phys = (raw_start + raw_first + row) % raw_cap;
      v = kv_raw[(uint64_t)phys * head_dim + d];
    } else if (row < n_kv) {
      uint32_t i = row - n_raw;
      int32_t r = comp_rows ? comp_rows[i] : (int32_t)i;
      if (r >= 0) {
        const uint8_t *cr = kv_comp_mx + (uint64_t)r * rowbytes;
        v = a_e4m3_dec(cr[d], a_e8m0_dec(cr[head_dim + d / 32]));
      }
    }
    Kf[(uint64_t)row * head_dim + d] = v;
    VbT[(uint64_t)d * n_kv_pad + row] = v;
  }
}

// Phase B decode entry: raw f32 window + persistent MXFP8 comp cache. Identical
// pipeline to the f32 variant, differing only in the comp-cache read (dequant).
extern "C" int ds4_attn_mx_decode_mx_scratch(
    float *heads, const float *q,
    const float *kv_raw, const uint8_t *kv_comp_mx, const int32_t *comp_rows,
    const float *sinks,
    uint32_t n_head, uint32_t head_dim,
    uint32_t raw_start, uint32_t raw_cap, uint32_t raw_first, uint32_t n_raw, uint32_t n_comp,
    uint8_t *scratch, size_t scratch_bytes) {
  const uint32_t n_kv = n_raw + n_comp;
  if (n_kv == 0 || (head_dim % 32) != 0 || !scratch) return -1;
  const uint32_t n_kv_pad = (n_kv + 31u) & ~31u;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int M = (int)n_head, D = (int)head_dim, KV = (int)n_kv_pad;
  attn_decode_scratch L = attn_decode_layout(M, D, KV);
  if (scratch_bytes < L.total) return -2;
  float *Kf = (float *)(scratch + L.Kf), *Vb = (float *)(scratch + L.Vb);
  float *scores = (float *)(scratch + L.scores), *P = (float *)(scratch + L.P);
  uint8_t *Qd = scratch + L.Qd, *Kd = scratch + L.Kd, *Pd = scratch + L.Pd, *Vd = scratch + L.Vd;
  ElementSF *Qs = (ElementSF *)(scratch + L.Qs), *Ks = (ElementSF *)(scratch + L.Ks);
  ElementSF *Ps = (ElementSF *)(scratch + L.Ps), *Vs = (ElementSF *)(scratch + L.Vs);
  void *w1 = L.ws1 ? (void *)(scratch + L.w1) : nullptr, *w2 = L.ws2 ? (void *)(scratch + L.w2) : nullptr;
  cudaMemsetAsync(Qs, 0, L.qa * 2); cudaMemsetAsync(Ks, 0, L.kb * 2);
  cudaMemsetAsync(Ps, 0, L.pa * 2); cudaMemsetAsync(Vs, 0, L.vb * 2);
  assemble_kv_mxcomp<<<n_kv_pad, 256>>>(Kf, Vb, kv_raw, kv_comp_mx, comp_rows,
                                        raw_start, raw_cap, raw_first, n_raw, n_kv, n_kv_pad, head_dim);
  mx_pack_a8(Qd, Qs, q, M, D);
  mx_pack_b8<G88>(Kd, Ks, Kf, KV, D);
  mx_pack_b8<G88>(Vd, Vs, Vb, D, KV);
  int rc = mx_run<G88>(scores, Qd, Qs, Kd, Ks, M, KV, D, w1);
  attn_softmax_sink<<<M, 256>>>(P, scores, sinks, nullptr, scale, M, n_kv_pad, n_kv);
  mx_pack_a8(Pd, Ps, P, M, KV);
  rc |= mx_run<G88>(heads, Pd, Ps, Vd, Vs, M, D, KV, w2);
  return rc;
}

// Decode attention over the current f32 KV caches (no persistent MX storage yet):
// assemble the shared KV (ring raw window + selected f32 comp rows) -> pack to
// MXFP8 -> QK^T -> sink softmax -> PV. Caller-owned scratch, no alloc/sync.
extern "C" int ds4_attn_mx_decode_f32_scratch(
    float *heads, const float *q,
    const float *kv_raw, const float *kv_comp, const int32_t *comp_rows,
    const float *sinks,
    uint32_t n_head, uint32_t head_dim,
    uint32_t raw_start, uint32_t raw_cap, uint32_t raw_first, uint32_t n_raw, uint32_t n_comp,
    uint8_t *scratch, size_t scratch_bytes) {
  const uint32_t n_kv = n_raw + n_comp;
  if (n_kv == 0 || (head_dim % 32) != 0 || !scratch) return -1;
  const uint32_t n_kv_pad = (n_kv + 31u) & ~31u;
  const float scale = 1.0f / sqrtf((float)head_dim);
  const int M = (int)n_head, D = (int)head_dim, KV = (int)n_kv_pad;
  attn_decode_scratch L = attn_decode_layout(M, D, KV);
  if (scratch_bytes < L.total) return -2;
  float *Kf = (float *)(scratch + L.Kf), *Vb = (float *)(scratch + L.Vb);
  float *scores = (float *)(scratch + L.scores), *P = (float *)(scratch + L.P);
  uint8_t *Qd = scratch + L.Qd, *Kd = scratch + L.Kd, *Pd = scratch + L.Pd, *Vd = scratch + L.Vd;
  ElementSF *Qs = (ElementSF *)(scratch + L.Qs), *Ks = (ElementSF *)(scratch + L.Ks);
  ElementSF *Ps = (ElementSF *)(scratch + L.Ps), *Vs = (ElementSF *)(scratch + L.Vs);
  void *w1 = L.ws1 ? (void *)(scratch + L.w1) : nullptr, *w2 = L.ws2 ? (void *)(scratch + L.w2) : nullptr;
  cudaMemsetAsync(Qs, 0, L.qa * 2); cudaMemsetAsync(Ks, 0, L.kb * 2);
  cudaMemsetAsync(Ps, 0, L.pa * 2); cudaMemsetAsync(Vs, 0, L.vb * 2);
  assemble_kv_f32comp<<<n_kv_pad, 256>>>(Kf, Vb, kv_raw, kv_comp, comp_rows,
                                         raw_start, raw_cap, raw_first, n_raw, n_kv, n_kv_pad, head_dim);
  mx_pack_a8(Qd, Qs, q, M, D);
  mx_pack_b8<G88>(Kd, Ks, Kf, KV, D);
  mx_pack_b8<G88>(Vd, Vs, Vb, D, KV);
  int rc = mx_run<G88>(scores, Qd, Qs, Kd, Ks, M, KV, D, w1);
  attn_softmax_sink<<<M, 256>>>(P, scores, sinks, nullptr, scale, M, n_kv_pad, n_kv);
  mx_pack_a8(Pd, Ps, P, M, KV);
  rc |= mx_run<G88>(heads, Pd, Ps, Vd, Vs, M, D, KV, w2);
  return rc;
}

// Self-allocating convenience wrapper (tests / non-hot-path). Allocs, runs, syncs.
extern "C" int ds4_attn_mx_decode(
    float *heads, const float *q,
    const float *raw_k, const float *raw_v,
    const uint8_t *comp_k, const uint8_t *comp_v, const int32_t *comp_rows,
    const float *sinks,
    uint32_t n_head, uint32_t head_dim, uint32_t n_raw, uint32_t n_comp) {
  size_t nbytes = ds4_attn_mx_decode_scratch_bytes(n_head, head_dim, n_raw + n_comp);
  uint8_t *scratch = nullptr;
  if (cudaMalloc(&scratch, nbytes) != cudaSuccess) return -3;
  int rc = ds4_attn_mx_decode_scratch(heads, q, raw_k, raw_v, comp_k, comp_v, comp_rows, sinks,
                                      n_head, head_dim, n_raw, n_comp, scratch, nbytes);
  cudaDeviceSynchronize();
  cudaFree(scratch);
  return rc;
}

#ifndef DS4_ATTN_CUTLASS_STANDALONE
extern "C" void ds4_gpu_attn_mx_scratch_free(void *p) { if (p) cudaFree(p); }

// Engine-facing decode-attention wrapper: unpacks ds4_gpu_tensor pointers,
// resolves sinks from the model map, computes the visible KV set (raw SWA window
// + comp: topk when indexed, else visible_comp = (pos+1)/ratio), and runs the
// MXFP8 decode attention over the current f32 caches. Scratch is a caller-owned
// (per-graph) buffer that grows on demand in coarse steps; no per-call malloc or
// sync -- launches on the default stream, ordered with the rest of the decode
// graph, exactly like every other kernel here.
extern "C" int ds4_gpu_attn_mx_decode(
    ds4_gpu_tensor *heads, ds4_gpu_tensor *q,
    ds4_gpu_tensor *raw_cache, uint32_t raw_cap, uint32_t raw_start, uint32_t n_raw,
    ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *comp_selected, uint32_t n_comp, uint32_t n_selected,
    const void *model_map, uint64_t model_size, uint64_t sinks_offset,
    uint32_t pos, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim,
    uint8_t **scratch, uint64_t *scratch_bytes) {
  const float *sinks = (const float *)cuda_model_range_ptr(
      model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
  if (!sinks || !heads || !q || !raw_cache || !scratch || !scratch_bytes) return -1;
  const uint32_t eff_win = window ? window : n_raw;
  const uint32_t raw_vis = n_raw < eff_win ? n_raw : eff_win;
  const uint32_t raw_first = n_raw - raw_vis;
  const int32_t *comp_rows = nullptr;
  uint32_t comp_vis;
  if (comp_selected && n_selected) {
    comp_rows = (const int32_t *)comp_selected->ptr;
    comp_vis = n_selected;
  } else {
    const uint32_t vc = ratio ? (pos + 1u) / ratio : n_comp;
    comp_vis = vc < n_comp ? vc : n_comp;
  }
  const uint32_t n_kv = raw_vis + comp_vis;
  if (n_kv == 0) return -1;
  // Grow the reused scratch in coarse (512-KV) steps to limit realloc churn as
  // context grows one token at a time.
  const uint32_t n_kv_grow = (n_kv + 511u) & ~511u;
  const size_t need = ds4_attn_mx_decode_scratch_bytes(n_head, head_dim, n_kv_grow);
  if (need > (size_t)*scratch_bytes) {
    if (*scratch) cudaFree(*scratch);
    if (cudaMalloc(scratch, need) != cudaSuccess) { *scratch = nullptr; *scratch_bytes = 0; return -3; }
    *scratch_bytes = need;
  }
  const float *comp_ptr = comp_cache ? (const float *)comp_cache->ptr : (const float *)raw_cache->ptr;
  return ds4_attn_mx_decode_f32_scratch(
      (float *)heads->ptr, (const float *)q->ptr,
      (const float *)raw_cache->ptr, comp_ptr, comp_rows, sinks,
      n_head, head_dim, raw_start, raw_cap, raw_first, raw_vis, comp_vis, *scratch, *scratch_bytes);
}
#endif  // !DS4_ATTN_CUTLASS_STANDALONE

#ifdef DS4_ATTN_CUTLASS_STANDALONE
#include <vector>
#include <random>
#include <cmath>
// Host MX dequant references (mirror device encoders) so the reference GEMM uses the same
// quantized values the device sees — isolates GEMM correctness from quant noise.
static const float hE2M1[8] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
static float h_e4m3_val(int i){ int e=(i>>3)&15,m=i&7; return e==0? (float)m*0.001953125f : (1.f+(float)m*0.125f)*exp2f((float)e-7.f); }
static float h_e4m3_snap(float x){ float ax=fminf(fabsf(x),448.f); int lo=0,hi=126; while(lo<hi){int mid=(lo+hi+1)>>1; if(h_e4m3_val(mid)<=ax)lo=mid; else hi=mid-1;} int b=lo; if(b<126){float bd=fabsf(ax-h_e4m3_val(b)),nd=fabsf(ax-h_e4m3_val(b+1)); if(nd<bd)b++;} return (x<0?-1.f:1.f)*h_e4m3_val(b); }
static float h_e2m1_snap(float x){ float ax=fminf(fabsf(x),6.f); int b=0; float bd=fabsf(ax-hE2M1[0]); for(int i=1;i<8;i++){float d=fabsf(ax-hE2M1[i]); if(d<bd){b=i;bd=d;}} return (x<0?-1.f:1.f)*hE2M1[b]; }
static float h_scale(float amax,float maxrep){ float r=fmaxf(amax,1e-20f)/maxrep; int e=(int)ceilf(log2f(r))+127; if(e<0)e=0; if(e>254)e=254; return exp2f((float)(e-127)); }
// Dequantized value array of a [R,K] row-major matrix under MX (fp8 maxrep=448 / fp4 maxrep=6).
static void h_dq(std::vector<float>& dq, const std::vector<float>& X, int R, int K, float maxrep, int fp4){
  dq.assign((size_t)R*K, 0.f);
  for(int r=0;r<R;r++) for(int kb=0;kb<K/32;kb++){
    float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(X[(size_t)r*K+kb*32+i]));
    float sc=h_scale(mx,maxrep);
    for(int i=0;i<32;i++){ int k=kb*32+i; float v=X[(size_t)r*K+k]/sc; dq[(size_t)r*K+k]=(fp4?h_e2m1_snap(v):h_e4m3_snap(v))*sc; }
  }
}

template <class G>
static int run_case(const char* name, int M, int N, int K, int bfp4){
  std::mt19937 rng(1234 + M + N*3 + K*7); std::normal_distribution<float> nd(0.f,1.f);
  std::vector<float> A((size_t)M*K), B((size_t)N*K);
  for(auto&v:A) v=nd(rng); for(auto&v:B) v=nd(rng)*0.5f;
  // reference over dequantized values (A always fp8, B fp8 or fp4)
  std::vector<float> dqA, dqB;
  h_dq(dqA, A, M, K, 448.f, 0);
  h_dq(dqB, B, N, K, bfp4?6.f:448.f, bfp4);
  std::vector<float> ref((size_t)M*N, 0.f);
  for(int m=0;m<M;m++) for(int n=0;n<N;n++){ double a=0; for(int k=0;k<K;k++) a+=(double)dqA[(size_t)m*K+k]*dqB[(size_t)n*K+k]; ref[(size_t)m*N+n]=(float)a; }
  // device pack + gemm
  float *dA,*dB,*dD; cudaMalloc(&dA,A.size()*4); cudaMalloc(&dB,B.size()*4); cudaMalloc(&dD,(size_t)M*N*4);
  cudaMemcpy(dA,A.data(),A.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B.data(),B.size()*4,cudaMemcpyHostToDevice);
  size_t sfa_n=mx_sfa_count(M,K), sfb_n=mx_sfb_count<G>(N,K);
  uint8_t *Adat; ElementSF *Asf; cudaMalloc(&Adat,(size_t)M*K); cudaMalloc(&Asf,sfa_n*sizeof(ElementSF));
  uint8_t *Bdat; ElementSF *Bsf; cudaMalloc(&Bdat,(size_t)N*K/(bfp4?2:1)); cudaMalloc(&Bsf,sfb_n*sizeof(ElementSF));
  cudaMemset(Asf,0,sfa_n*sizeof(ElementSF)); cudaMemset(Bsf,0,sfb_n*sizeof(ElementSF));
  cudaMemset(Bdat,0,(size_t)N*K/(bfp4?2:1));
  mx_pack_a8(Adat,Asf,dA,M,K);
  if(bfp4) mx_pack_b4<G>(Bdat,Bsf,dB,N,K); else mx_pack_b8<G>(Bdat,Bsf,dB,N,K);
  size_t ws_n=mx_ws_bytes<G>(M,N,K); void*ws=nullptr; if(ws_n) cudaMalloc(&ws,ws_n);
  int rc=mx_run<G>(dD,Adat,Asf,Bdat,Bsf,M,N,K,ws);
  std::vector<float> got((size_t)M*N); cudaMemcpy(got.data(),dD,(size_t)M*N*4,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  double maxrel=0,maxabs=0; int bad=0;
  for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-ref[i]); double r=a/(fabs(ref[i])+1e-2); maxabs=fmax(maxabs,a); maxrel=fmax(maxrel,r); if(r>0.02&&a>0.05) bad++; }
  printf("%-22s M=%d N=%d K=%d rc=%d max_rel=%.4f max_abs=%.4f bad=%d/%zu -> %s\n",
         name,M,N,K,rc,maxrel,maxabs,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  cudaFree(dA);cudaFree(dB);cudaFree(dD);cudaFree(Adat);cudaFree(Asf);cudaFree(Bdat);cudaFree(Bsf); if(ws)cudaFree(ws);
  return (rc==0&&bad==0)?0:1;
}

// Single-head attention pipeline (MXFP8): QK GEMM -> softmax+sink -> PV GEMM, vs
// a host reference over the same MX-dequantized values. Proves the Stage-3 core
// mechanic before batching/gather/mixing. n_kv multiple of 32 (PV contraction).
static int run_attn_1head(int n_q, int n_kv, int head_dim, float sink) {
  std::mt19937 rng(99 + n_q + n_kv); std::normal_distribution<float> nd(0.f,1.f);
  std::vector<float> Q((size_t)n_q*head_dim), K((size_t)n_kv*head_dim), Vb((size_t)head_dim*n_kv);
  for(auto&v:Q)v=nd(rng)*0.5f; for(auto&v:K)v=nd(rng)*0.5f; for(auto&v:Vb)v=nd(rng);
  const float scale = 1.0f/sqrtf((float)head_dim);
  // host reference over dq values
  std::vector<float> dqQ,dqK,dqVb; h_dq(dqQ,Q,n_q,head_dim,448.f,0); h_dq(dqK,K,n_kv,head_dim,448.f,0); h_dq(dqVb,Vb,head_dim,n_kv,448.f,0);
  std::vector<float> Pref((size_t)n_q*n_kv), outref((size_t)n_q*head_dim,0.f);
  for(int q=0;q<n_q;q++){
    float m=sink; std::vector<float> s(n_kv);
    for(int j=0;j<n_kv;j++){ double a=0; for(int d=0;d<head_dim;d++) a+=(double)dqQ[(size_t)q*head_dim+d]*dqK[(size_t)j*head_dim+d]; s[j]=(float)a*scale; m=fmaxf(m,s[j]); }
    double den=expf(sink-m); for(int j=0;j<n_kv;j++) den+=expf(s[j]-m);
    for(int j=0;j<n_kv;j++) Pref[(size_t)q*n_kv+j]=(float)(expf(s[j]-m)/den);
  }
  std::vector<float> dqP; h_dq(dqP,Pref,n_q,n_kv,448.f,0);   // P is packed to mxfp8 for the PV GEMM
  for(int q=0;q<n_q;q++) for(int d=0;d<head_dim;d++){ double a=0; for(int j=0;j<n_kv;j++) a+=(double)dqP[(size_t)q*n_kv+j]*dqVb[(size_t)d*n_kv+j]; outref[(size_t)q*head_dim+d]=(float)a; }
  // device pipeline
  float *dQ,*dK,*dVb,*dScores,*dP,*dOut,*dSink;
  cudaMalloc(&dQ,Q.size()*4); cudaMalloc(&dK,K.size()*4); cudaMalloc(&dVb,Vb.size()*4);
  cudaMalloc(&dScores,(size_t)n_q*n_kv*4); cudaMalloc(&dP,(size_t)n_q*n_kv*4); cudaMalloc(&dOut,(size_t)n_q*head_dim*4);
  cudaMalloc(&dSink,(size_t)n_q*4);
  cudaMemcpy(dQ,Q.data(),Q.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dK,K.data(),K.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dVb,Vb.data(),Vb.size()*4,cudaMemcpyHostToDevice);
  { std::vector<float> sk(n_q,sink); cudaMemcpy(dSink,sk.data(),(size_t)n_q*4,cudaMemcpyHostToDevice); }
  // pack + QK
  size_t qa_n=mx_sfa_count(n_q,head_dim), kb_n=mx_sfb_count<G88>(n_kv,head_dim);
  uint8_t *Qd,*Kd; ElementSF *Qs,*Ks; cudaMalloc(&Qd,(size_t)n_q*head_dim); cudaMalloc(&Qs,qa_n*2); cudaMalloc(&Kd,(size_t)n_kv*head_dim); cudaMalloc(&Ks,kb_n*2);
  cudaMemset(Qs,0,qa_n*2); cudaMemset(Ks,0,kb_n*2);
  mx_pack_a8(Qd,Qs,dQ,n_q,head_dim); mx_pack_b8<G88>(Kd,Ks,dK,n_kv,head_dim);
  size_t ws1=mx_ws_bytes<G88>(n_q,n_kv,head_dim); void*w1=nullptr; if(ws1)cudaMalloc(&w1,ws1);
  int rc=mx_run<G88>(dScores,Qd,Qs,Kd,Ks,n_q,n_kv,head_dim,w1);
  // softmax + sink
  attn_softmax_sink<<<n_q,256>>>(dP,dScores,dSink,nullptr,scale,n_q,n_kv,0);
  // pack + PV  (A=P [n_q,n_kv], B=Vb [head_dim,n_kv])
  size_t pa_n=mx_sfa_count(n_q,n_kv), vb_n=mx_sfb_count<G88>(head_dim,n_kv);
  uint8_t *Pd,*Vd; ElementSF *Ps,*Vs; cudaMalloc(&Pd,(size_t)n_q*n_kv); cudaMalloc(&Ps,pa_n*2); cudaMalloc(&Vd,(size_t)head_dim*n_kv); cudaMalloc(&Vs,vb_n*2);
  cudaMemset(Ps,0,pa_n*2); cudaMemset(Vs,0,vb_n*2);
  mx_pack_a8(Pd,Ps,dP,n_q,n_kv); mx_pack_b8<G88>(Vd,Vs,dVb,head_dim,n_kv);
  size_t ws2=mx_ws_bytes<G88>(n_q,head_dim,n_kv); void*w2=nullptr; if(ws2)cudaMalloc(&w2,ws2);
  rc|=mx_run<G88>(dOut,Pd,Ps,Vd,Vs,n_q,head_dim,n_kv,w2);
  std::vector<float> got((size_t)n_q*head_dim); cudaMemcpy(got.data(),dOut,got.size()*4,cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
  double maxrel=0; int bad=0;
  for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-outref[i]); double r=a/(fabs(outref[i])+1e-2); maxrel=fmax(maxrel,r); if(r>0.03&&a>0.02)bad++; }
  printf("attn 1-head n_q=%d n_kv=%d sink=%.1f rc=%d max_rel=%.4f bad=%d/%zu -> %s\n",
         n_q,n_kv,(double)sink,rc,maxrel,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dK);cudaFree(dVb);cudaFree(dScores);cudaFree(dP);cudaFree(dOut);cudaFree(dSink);
  cudaFree(Qd);cudaFree(Qs);cudaFree(Kd);cudaFree(Ks);cudaFree(Pd);cudaFree(Ps);cudaFree(Vd);cudaFree(Vs); if(w1)cudaFree(w1); if(w2)cudaFree(w2);
  return (rc==0&&bad==0)?0:1;
}

// Batched multi-head attention (MXFP8): L=n_head heads, each a full QK->softmax+sink->PV.
// Q/K/Vb laid out head-major [head][*][d]; per-head sink; parity vs host reference.
static int run_attn_mha(int n_head, int n_q, int n_kv, int head_dim) {
  std::mt19937 rng(7 + n_head + n_q + n_kv); std::normal_distribution<float> nd(0.f,1.f);
  std::vector<float> Q((size_t)n_head*n_q*head_dim), K((size_t)n_head*n_kv*head_dim), Vb((size_t)n_head*head_dim*n_kv), sinks(n_head);
  for(auto&v:Q)v=nd(rng)*0.5f; for(auto&v:K)v=nd(rng)*0.5f; for(auto&v:Vb)v=nd(rng);
  for(int h=0;h<n_head;h++) sinks[h]=(float)(h%3)*1.5f;
  const float scale=1.0f/sqrtf((float)head_dim);
  // host reference (per head, over dq values)
  std::vector<float> dqQ,dqK,dqVb; h_dq(dqQ,Q,n_head*n_q,head_dim,448.f,0); h_dq(dqK,K,n_head*n_kv,head_dim,448.f,0); h_dq(dqVb,Vb,n_head*head_dim,n_kv,448.f,0);
  std::vector<float> Pref((size_t)n_head*n_q*n_kv), outref((size_t)n_head*n_q*head_dim,0.f);
  for(int h=0;h<n_head;h++) for(int q=0;q<n_q;q++){
    const float*qq=dqQ.data()+((size_t)h*n_q+q)*head_dim; float m=sinks[h]; std::vector<float> s(n_kv);
    for(int j=0;j<n_kv;j++){ const float*kk=dqK.data()+((size_t)h*n_kv+j)*head_dim; double a=0; for(int d=0;d<head_dim;d++)a+=(double)qq[d]*kk[d]; s[j]=(float)a*scale; m=fmaxf(m,s[j]); }
    double den=expf(sinks[h]-m); for(int j=0;j<n_kv;j++)den+=expf(s[j]-m);
    for(int j=0;j<n_kv;j++) Pref[((size_t)h*n_q+q)*n_kv+j]=(float)(expf(s[j]-m)/den);
  }
  std::vector<float> dqP; h_dq(dqP,Pref,n_head*n_q,n_kv,448.f,0);
  for(int h=0;h<n_head;h++) for(int q=0;q<n_q;q++) for(int d=0;d<head_dim;d++){ const float*pp=dqP.data()+((size_t)h*n_q+q)*n_kv; const float*vv=dqVb.data()+((size_t)h*head_dim+d)*n_kv; double a=0; for(int j=0;j<n_kv;j++)a+=(double)pp[j]*vv[j]; outref[((size_t)h*n_q+q)*head_dim+d]=(float)a; }
  // device
  float *dQ,*dK,*dVb,*dScores,*dP,*dOut,*dSinkRow;
  cudaMalloc(&dQ,Q.size()*4); cudaMalloc(&dK,K.size()*4); cudaMalloc(&dVb,Vb.size()*4);
  cudaMalloc(&dScores,(size_t)n_head*n_q*n_kv*4); cudaMalloc(&dP,(size_t)n_head*n_q*n_kv*4); cudaMalloc(&dOut,(size_t)n_head*n_q*head_dim*4);
  cudaMalloc(&dSinkRow,(size_t)n_head*n_q*4);
  cudaMemcpy(dQ,Q.data(),Q.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dK,K.data(),K.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dVb,Vb.data(),Vb.size()*4,cudaMemcpyHostToDevice);
  { std::vector<float> sr((size_t)n_head*n_q); for(int h=0;h<n_head;h++)for(int q=0;q<n_q;q++)sr[(size_t)h*n_q+q]=sinks[h]; cudaMemcpy(dSinkRow,sr.data(),sr.size()*4,cudaMemcpyHostToDevice); }
  size_t qa=mx_sfa_count_bat(n_q,head_dim,n_head), kb=mx_sfb_count_bat<G88>(n_kv,head_dim,n_head);
  uint8_t*Qd,*Kd; ElementSF*Qs,*Ks; cudaMalloc(&Qd,Q.size()); cudaMalloc(&Qs,qa*2); cudaMalloc(&Kd,K.size()); cudaMalloc(&Ks,kb*2);
  cudaMemset(Qs,0,qa*2); cudaMemset(Ks,0,kb*2);
  mx_pack_a8_bat(Qd,Qs,dQ,n_q,head_dim,n_head); mx_pack_b8_bat<G88>(Kd,Ks,dK,n_kv,head_dim,n_head);
  size_t ws1=mx_ws_bytes_bat<G88>(n_q,n_kv,head_dim,n_head); void*w1=nullptr; if(ws1)cudaMalloc(&w1,ws1);
  int rc=mx_run_bat<G88>(dScores,Qd,Qs,Kd,Ks,n_q,n_kv,head_dim,n_head,w1);
  attn_softmax_sink<<<n_head*n_q,256>>>(dP,dScores,dSinkRow,nullptr,scale,n_head*n_q,n_kv,0);
  size_t pa=mx_sfa_count_bat(n_q,n_kv,n_head), vb=mx_sfb_count_bat<G88>(head_dim,n_kv,n_head);
  uint8_t*Pd,*Vd; ElementSF*Ps,*Vs; cudaMalloc(&Pd,(size_t)n_head*n_q*n_kv); cudaMalloc(&Ps,pa*2); cudaMalloc(&Vd,(size_t)n_head*head_dim*n_kv); cudaMalloc(&Vs,vb*2);
  cudaMemset(Ps,0,pa*2); cudaMemset(Vs,0,vb*2);
  mx_pack_a8_bat(Pd,Ps,dP,n_q,n_kv,n_head); mx_pack_b8_bat<G88>(Vd,Vs,dVb,head_dim,n_kv,n_head);
  size_t ws2=mx_ws_bytes_bat<G88>(n_q,head_dim,n_kv,n_head); void*w2=nullptr; if(ws2)cudaMalloc(&w2,ws2);
  rc|=mx_run_bat<G88>(dOut,Pd,Ps,Vd,Vs,n_q,head_dim,n_kv,n_head,w2);
  std::vector<float> got((size_t)n_head*n_q*head_dim); cudaMemcpy(got.data(),dOut,got.size()*4,cudaMemcpyDeviceToHost); cudaDeviceSynchronize();
  double maxrel=0; int bad=0;
  for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-outref[i]); double r=a/(fabs(outref[i])+1e-2); maxrel=fmax(maxrel,r); if(r>0.03&&a>0.02)bad++; }
  printf("attn MHA n_head=%d n_q=%d n_kv=%d rc=%d max_rel=%.4f bad=%d/%zu -> %s\n",
         n_head,n_q,n_kv,rc,maxrel,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dK);cudaFree(dVb);cudaFree(dScores);cudaFree(dP);cudaFree(dOut);cudaFree(dSinkRow);
  cudaFree(Qd);cudaFree(Qs);cudaFree(Kd);cudaFree(Ks);cudaFree(Pd);cudaFree(Ps);cudaFree(Vd);cudaFree(Vs); if(w1)cudaFree(w1); if(w2)cudaFree(w2);
  return (rc==0&&bad==0)?0:1;
}

// Host MXFP8 pack (matches device format: E4M3 data + E8M0 scales, block 32).
static uint8_t h_e4m3_byte(float x){ float ax=fminf(fabsf(x),448.f); int lo=0,hi=126; while(lo<hi){int mid=(lo+hi+1)>>1; if(h_e4m3_val(mid)<=ax)lo=mid; else hi=mid-1;} int b=lo; if(b<126){float bd=fabsf(ax-h_e4m3_val(b)),nd=fabsf(ax-h_e4m3_val(b+1)); if(nd<bd)b++;} return (uint8_t)(b|((x<0)?0x80:0)); }
static uint8_t h_e8m0_byte(float amax){ float r=fmaxf(amax,1e-20f)/448.f; int e=(int)ceilf(log2f(r))+127; if(e<0)e=0; if(e>254)e=254; return (uint8_t)e; }
static void host_pack_mxfp8(std::vector<uint8_t>& cache, const std::vector<float>& src, int cap, int hd){
  int nblk=hd/32, rowbytes=hd+nblk; cache.assign((size_t)cap*rowbytes,0);
  for(int r=0;r<cap;r++) for(int b=0;b<nblk;b++){ float amax=0; for(int j=0;j<32;j++)amax=fmaxf(amax,fabsf(src[(size_t)r*hd+b*32+j]));
    uint8_t e8=h_e8m0_byte(amax); float sc=exp2f((float)((int)e8-127));
    for(int j=0;j<32;j++) cache[(size_t)r*rowbytes+b*32+j]=h_e4m3_byte(src[(size_t)r*hd+b*32+j]/sc);
    cache[(size_t)r*rowbytes+hd+b]=e8; }
}
// Full decode attention entry-point parity test.
static int run_decode_test(int n_head, int n_raw, int n_comp, int cap, int head_dim){
  std::mt19937 rng(31+n_head+n_raw+n_comp); std::normal_distribution<float> nd(0.f,1.f);
  int n_kv=n_raw+n_comp;
  std::vector<float> Q((size_t)n_head*head_dim), rawK((size_t)n_raw*head_dim), rawV((size_t)n_raw*head_dim),
    srcCK((size_t)cap*head_dim), srcCV((size_t)cap*head_dim), sinks(n_head);
  std::vector<int32_t> rows(n_comp);
  for(auto&v:Q)v=nd(rng)*0.5f; for(auto&v:rawK)v=nd(rng)*0.5f; for(auto&v:rawV)v=nd(rng);
  for(auto&v:srcCK)v=nd(rng)*0.5f; for(auto&v:srcCV)v=nd(rng);
  for(int h=0;h<n_head;h++) sinks[h]=(float)(h%3);
  for(int i=0;i<n_comp;i++) rows[i]=(i*29+7)%cap;
  const float scale=1.0f/sqrtf((float)head_dim);
  // reference: assemble Kf (raw f32 ++ dq(comp)), VbT; then quantize operands and do the attention math
  const int n_kv_pad=(n_kv+31)&~31;   // V and P are quantized/contracted at padded width, like the device
  std::vector<float> dqCK,dqCV; h_dq(dqCK,srcCK,cap,head_dim,448.f,0); h_dq(dqCV,srcCV,cap,head_dim,448.f,0);
  std::vector<float> Kf((size_t)n_kv*head_dim), Vb((size_t)head_dim*n_kv_pad,0.f);   // Vb padded cols = 0
  for(int row=0;row<n_kv;row++) for(int d=0;d<head_dim;d++){
    float k,v; if(row<n_raw){k=rawK[(size_t)row*head_dim+d];v=rawV[(size_t)row*head_dim+d];}
    else {int r=rows[row-n_raw]; k=dqCK[(size_t)r*head_dim+d]; v=dqCV[(size_t)r*head_dim+d];}
    Kf[(size_t)row*head_dim+d]=k; Vb[(size_t)d*n_kv_pad+row]=v; }
  std::vector<float> dqQ,dqKf,dqVb; h_dq(dqQ,Q,n_head,head_dim,448.f,0); h_dq(dqKf,Kf,n_kv,head_dim,448.f,0); h_dq(dqVb,Vb,head_dim,n_kv_pad,448.f,0);
  std::vector<float> Pref((size_t)n_head*n_kv_pad,0.f), outref((size_t)n_head*head_dim,0.f);
  for(int h=0;h<n_head;h++){ const float*qq=dqQ.data()+(size_t)h*head_dim; float m=sinks[h]; std::vector<float> s(n_kv);
    for(int j=0;j<n_kv;j++){ const float*kk=dqKf.data()+(size_t)j*head_dim; double a=0; for(int d=0;d<head_dim;d++)a+=(double)qq[d]*kk[d]; s[j]=(float)a*scale; m=fmaxf(m,s[j]); }
    double den=expf(sinks[h]-m); for(int j=0;j<n_kv;j++)den+=expf(s[j]-m);
    for(int j=0;j<n_kv;j++)Pref[(size_t)h*n_kv_pad+j]=(float)(expf(s[j]-m)/den); }   // padded cols stay 0
  std::vector<float> dqP; h_dq(dqP,Pref,n_head,n_kv_pad,448.f,0);
  for(int h=0;h<n_head;h++) for(int d=0;d<head_dim;d++){ const float*pp=dqP.data()+(size_t)h*n_kv_pad; const float*vv=dqVb.data()+(size_t)d*n_kv_pad; double a=0; for(int j=0;j<n_kv_pad;j++)a+=(double)pp[j]*vv[j]; outref[(size_t)h*head_dim+d]=(float)a; }
  // device
  std::vector<uint8_t> cacheK,cacheV; host_pack_mxfp8(cacheK,srcCK,cap,head_dim); host_pack_mxfp8(cacheV,srcCV,cap,head_dim);
  float *dQ,*dRK,*dRV,*dSink,*dHeads; uint8_t*dCK,*dCV; int32_t*dRows;
  cudaMalloc(&dQ,Q.size()*4); cudaMalloc(&dRK,rawK.size()*4); cudaMalloc(&dRV,rawV.size()*4);
  cudaMalloc(&dSink,(size_t)n_head*4); cudaMalloc(&dHeads,(size_t)n_head*head_dim*4);
  cudaMalloc(&dCK,cacheK.size()); cudaMalloc(&dCV,cacheV.size()); cudaMalloc(&dRows,(size_t)n_comp*4);
  cudaMemcpy(dQ,Q.data(),Q.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dRK,rawK.data(),rawK.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dRV,rawV.data(),rawV.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dSink,sinks.data(),(size_t)n_head*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dCK,cacheK.data(),cacheK.size(),cudaMemcpyHostToDevice); cudaMemcpy(dCV,cacheV.data(),cacheV.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dRows,rows.data(),(size_t)n_comp*4,cudaMemcpyHostToDevice);
  int rc=ds4_attn_mx_decode(dHeads,dQ,dRK,dRV,dCK,dCV,dRows,dSink,n_head,head_dim,n_raw,n_comp);
  std::vector<float> got((size_t)n_head*head_dim); cudaMemcpy(got.data(),dHeads,got.size()*4,cudaMemcpyDeviceToHost);
  double maxrel=0; int bad=0; for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-outref[i]); double r=a/(fabs(outref[i])+1e-2); maxrel=fmax(maxrel,r); if(r>0.04&&a>0.03)bad++; }
  printf("decode entry n_head=%d n_raw=%d n_comp=%d (n_kv=%d) rc=%d max_rel=%.4f bad=%d/%zu -> %s\n",
         n_head,n_raw,n_comp,n_kv,rc,maxrel,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dRK);cudaFree(dRV);cudaFree(dSink);cudaFree(dHeads);cudaFree(dCK);cudaFree(dCV);cudaFree(dRows);
  return (rc==0&&bad==0)?0:1;
}

// f32-cache decode path (compute-path validation): ring raw window + f32 comp
// (identity or indexed) -> MX attention, vs host reference over the same values.
static int run_decode_f32_test(const char* tag, int n_head, int raw_cap, int raw_start, int raw_first,
                               int n_raw, int cap_comp, int n_comp, int use_index, int head_dim){
  std::mt19937 rng(53+n_head+n_raw+n_comp+use_index); std::normal_distribution<float> nd(0.f,1.f);
  int n_kv=n_raw+n_comp;
  std::vector<float> Q((size_t)n_head*head_dim), kvRaw((size_t)raw_cap*head_dim), kvComp((size_t)cap_comp*head_dim), sinks(n_head);
  std::vector<int32_t> idx(n_comp);
  for(auto&v:Q)v=nd(rng)*0.5f; for(auto&v:kvRaw)v=nd(rng); for(auto&v:kvComp)v=nd(rng);
  for(int h=0;h<n_head;h++) sinks[h]=(float)(h%3);
  for(int i=0;i<n_comp;i++) idx[i]=(i*23+5)%cap_comp;
  const float scale=1.0f/sqrtf((float)head_dim);
  const int n_kv_pad=(n_kv+31)&~31;
  // reference assemble (K==V)
  std::vector<float> Kf((size_t)n_kv*head_dim), Vb((size_t)head_dim*n_kv_pad,0.f);
  for(int row=0;row<n_kv;row++) for(int d=0;d<head_dim;d++){
    float v; if(row<n_raw){ int phys=(raw_start+raw_first+row)%raw_cap; v=kvRaw[(size_t)phys*head_dim+d]; }
    else { int r=use_index?idx[row-n_raw]:(row-n_raw); v=kvComp[(size_t)r*head_dim+d]; }
    Kf[(size_t)row*head_dim+d]=v; Vb[(size_t)d*n_kv_pad+row]=v; }
  std::vector<float> dqQ,dqKf,dqVb; h_dq(dqQ,Q,n_head,head_dim,448.f,0); h_dq(dqKf,Kf,n_kv,head_dim,448.f,0); h_dq(dqVb,Vb,head_dim,n_kv_pad,448.f,0);
  std::vector<float> Pref((size_t)n_head*n_kv_pad,0.f), outref((size_t)n_head*head_dim,0.f);
  for(int h=0;h<n_head;h++){ const float*qq=dqQ.data()+(size_t)h*head_dim; float m=sinks[h]; std::vector<float> s(n_kv);
    for(int j=0;j<n_kv;j++){ const float*kk=dqKf.data()+(size_t)j*head_dim; double a=0; for(int d=0;d<head_dim;d++)a+=(double)qq[d]*kk[d]; s[j]=(float)a*scale; m=fmaxf(m,s[j]); }
    double den=expf(sinks[h]-m); for(int j=0;j<n_kv;j++)den+=expf(s[j]-m);
    for(int j=0;j<n_kv;j++)Pref[(size_t)h*n_kv_pad+j]=(float)(expf(s[j]-m)/den); }
  std::vector<float> dqP; h_dq(dqP,Pref,n_head,n_kv_pad,448.f,0);
  for(int h=0;h<n_head;h++) for(int d=0;d<head_dim;d++){ const float*pp=dqP.data()+(size_t)h*n_kv_pad; const float*vv=dqVb.data()+(size_t)d*n_kv_pad; double a=0; for(int j=0;j<n_kv_pad;j++)a+=(double)pp[j]*vv[j]; outref[(size_t)h*head_dim+d]=(float)a; }
  // device
  float *dQ,*dRaw,*dComp,*dSink,*dHeads; int32_t*dIdx;
  cudaMalloc(&dQ,Q.size()*4); cudaMalloc(&dRaw,kvRaw.size()*4); cudaMalloc(&dComp,kvComp.size()*4); cudaMalloc(&dSink,(size_t)n_head*4); cudaMalloc(&dHeads,(size_t)n_head*head_dim*4); cudaMalloc(&dIdx,(size_t)n_comp*4);
  cudaMemcpy(dQ,Q.data(),Q.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dRaw,kvRaw.data(),kvRaw.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dComp,kvComp.data(),kvComp.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dSink,sinks.data(),(size_t)n_head*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dIdx,idx.data(),(size_t)n_comp*4,cudaMemcpyHostToDevice);
  size_t nb=ds4_attn_mx_decode_scratch_bytes(n_head,head_dim,n_kv); uint8_t*sc=nullptr; cudaMalloc(&sc,nb);
  int rc=ds4_attn_mx_decode_f32_scratch(dHeads,dQ,dRaw,dComp,use_index?dIdx:nullptr,dSink,n_head,head_dim,
                                        (uint32_t)raw_start,(uint32_t)raw_cap,(uint32_t)raw_first,(uint32_t)n_raw,(uint32_t)n_comp,sc,nb);
  cudaDeviceSynchronize();
  std::vector<float> got((size_t)n_head*head_dim); cudaMemcpy(got.data(),dHeads,got.size()*4,cudaMemcpyDeviceToHost);
  double maxrel=0; int bad=0; for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-outref[i]); double r=a/(fabs(outref[i])+1e-2); maxrel=fmax(maxrel,r); if(r>0.04&&a>0.03)bad++; }
  printf("decode-f32 %-10s n_head=%d n_raw=%d n_comp=%d idx=%d (n_kv=%d) rc=%d max_rel=%.4f bad=%d/%zu -> %s\n",
         tag,n_head,n_raw,n_comp,use_index,n_kv,rc,maxrel,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dRaw);cudaFree(dComp);cudaFree(dSink);cudaFree(dHeads);cudaFree(dIdx);cudaFree(sc);
  return (rc==0&&bad==0)?0:1;
}

// Phase B: raw f32 ring + PERSISTENT MXFP8 comp cache -> MX attention, vs ref.
static int run_decode_mx_test(const char* tag, int n_head, int raw_cap, int raw_start, int raw_first,
                              int n_raw, int cap_comp, int n_comp, int use_index, int head_dim){
  std::mt19937 rng(71+n_head+n_raw+n_comp+use_index); std::normal_distribution<float> nd(0.f,1.f);
  int n_kv=n_raw+n_comp;
  std::vector<float> Q((size_t)n_head*head_dim), kvRaw((size_t)raw_cap*head_dim), compSrc((size_t)cap_comp*head_dim), sinks(n_head);
  std::vector<int32_t> idx(n_comp);
  for(auto&v:Q)v=nd(rng)*0.5f; for(auto&v:kvRaw)v=nd(rng); for(auto&v:compSrc)v=nd(rng);
  for(int h=0;h<n_head;h++) sinks[h]=(float)(h%3);
  for(int i=0;i<n_comp;i++) idx[i]=(i*23+5)%cap_comp;
  const float scale=1.0f/sqrtf((float)head_dim); const int n_kv_pad=(n_kv+31)&~31;
  std::vector<float> dqComp; h_dq(dqComp,compSrc,cap_comp,head_dim,448.f,0);   // comp stored MXFP8 -> dequant values
  std::vector<float> Kf((size_t)n_kv*head_dim), Vb((size_t)head_dim*n_kv_pad,0.f);
  for(int row=0;row<n_kv;row++) for(int d=0;d<head_dim;d++){
    float v; if(row<n_raw){ int phys=(raw_start+raw_first+row)%raw_cap; v=kvRaw[(size_t)phys*head_dim+d]; }
    else { int r=use_index?idx[row-n_raw]:(row-n_raw); v=dqComp[(size_t)r*head_dim+d]; }
    Kf[(size_t)row*head_dim+d]=v; Vb[(size_t)d*n_kv_pad+row]=v; }
  std::vector<float> dqQ,dqKf,dqVb; h_dq(dqQ,Q,n_head,head_dim,448.f,0); h_dq(dqKf,Kf,n_kv,head_dim,448.f,0); h_dq(dqVb,Vb,head_dim,n_kv_pad,448.f,0);
  std::vector<float> Pref((size_t)n_head*n_kv_pad,0.f), outref((size_t)n_head*head_dim,0.f);
  for(int h=0;h<n_head;h++){ const float*qq=dqQ.data()+(size_t)h*head_dim; float m=sinks[h]; std::vector<float> s(n_kv);
    for(int j=0;j<n_kv;j++){ const float*kk=dqKf.data()+(size_t)j*head_dim; double a=0; for(int d=0;d<head_dim;d++)a+=(double)qq[d]*kk[d]; s[j]=(float)a*scale; m=fmaxf(m,s[j]); }
    double den=expf(sinks[h]-m); for(int j=0;j<n_kv;j++)den+=expf(s[j]-m);
    for(int j=0;j<n_kv;j++)Pref[(size_t)h*n_kv_pad+j]=(float)(expf(s[j]-m)/den); }
  std::vector<float> dqP; h_dq(dqP,Pref,n_head,n_kv_pad,448.f,0);
  for(int h=0;h<n_head;h++) for(int d=0;d<head_dim;d++){ const float*pp=dqP.data()+(size_t)h*n_kv_pad; const float*vv=dqVb.data()+(size_t)d*n_kv_pad; double a=0; for(int j=0;j<n_kv_pad;j++)a+=(double)pp[j]*vv[j]; outref[(size_t)h*head_dim+d]=(float)a; }
  std::vector<uint8_t> cache; host_pack_mxfp8(cache,compSrc,cap_comp,head_dim);
  float *dQ,*dRaw,*dSink,*dHeads; uint8_t*dComp; int32_t*dIdx;
  cudaMalloc(&dQ,Q.size()*4); cudaMalloc(&dRaw,kvRaw.size()*4); cudaMalloc(&dSink,(size_t)n_head*4); cudaMalloc(&dHeads,(size_t)n_head*head_dim*4); cudaMalloc(&dComp,cache.size()); cudaMalloc(&dIdx,(size_t)n_comp*4);
  cudaMemcpy(dQ,Q.data(),Q.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dRaw,kvRaw.data(),kvRaw.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dSink,sinks.data(),(size_t)n_head*4,cudaMemcpyHostToDevice); cudaMemcpy(dComp,cache.data(),cache.size(),cudaMemcpyHostToDevice); cudaMemcpy(dIdx,idx.data(),(size_t)n_comp*4,cudaMemcpyHostToDevice);
  size_t nb=ds4_attn_mx_decode_scratch_bytes(n_head,head_dim,n_kv); uint8_t*sc=nullptr; cudaMalloc(&sc,nb);
  int rc=ds4_attn_mx_decode_mx_scratch(dHeads,dQ,dRaw,dComp,use_index?dIdx:nullptr,dSink,n_head,head_dim,
                                       (uint32_t)raw_start,(uint32_t)raw_cap,(uint32_t)raw_first,(uint32_t)n_raw,(uint32_t)n_comp,sc,nb);
  cudaDeviceSynchronize();
  std::vector<float> got((size_t)n_head*head_dim); cudaMemcpy(got.data(),dHeads,got.size()*4,cudaMemcpyDeviceToHost);
  double maxrel=0; int bad=0; for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-outref[i]); double r=a/(fabs(outref[i])+1e-2); maxrel=fmax(maxrel,r); if(r>0.04&&a>0.03)bad++; }
  printf("decode-MX  %-10s n_head=%d n_raw=%d n_comp=%d idx=%d (n_kv=%d) rc=%d max_rel=%.4f bad=%d/%zu -> %s\n",
         tag,n_head,n_raw,n_comp,use_index,n_kv,rc,maxrel,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  cudaFree(dQ);cudaFree(dRaw);cudaFree(dSink);cudaFree(dHeads);cudaFree(dComp);cudaFree(dIdx);cudaFree(sc);
  return (rc==0&&bad==0)?0:1;
}

int main(){
  int rc=0;
  // QK^T: M=n_q, N=n_kv, K=head_dim=512
  rc|=run_case<G88>("QK mxfp8xmxfp8",128,256,512,0);
  rc|=run_case<G84>("QK mxfp8xmxfp4",128,256,512,1);
  // PV: M=n_q, N=head_dim=512, K=kv_tile=128
  rc|=run_case<G88>("PV mxfp8xmxfp8",128,512,128,0);
  rc|=run_case<G84>("PV mxfp8xmxfp4",128,512,128,1);
  // Stage 3 mechanic: full single-head attention (QK -> softmax+sink -> PV)
  rc|=run_attn_1head(128,256,512,0.0f);
  rc|=run_attn_1head(64,512,512,2.5f);
  rc|=run_attn_1head(1,128,512,1.0f);   // decode-shaped (single query)
  // Stage 3 batched multi-head
  rc|=run_attn_mha(16,64,256,512);
  rc|=run_attn_mha(32,1,128,512);       // decode: 32 heads, single query each
  // Stage 4 decode entry point (gather raw+comp -> attention), n_kv not mult-32 -> exercises padding
  rc|=run_decode_test(64,56,200,4096,512);   // n_kv=256 (mult-32, no padding)
  rc|=run_decode_test(64,40,200,4096,512);   // n_kv=240 -> pad 256
  rc|=run_decode_test(128,17,100,2048,512);  // n_kv=117 -> pad 128
  // f32-cache decode path (engine wiring validation): raw ring + f32 comp
  rc|=run_decode_f32_test("static",   64, 512, 0,   0,  40, 4096, 200, 0, 512);  // identity comp, raw_first=0
  rc|=run_decode_f32_test("indexed",  64, 512, 0,   0,  40, 4096, 200, 1, 512);  // topk comp
  rc|=run_decode_f32_test("ring+win", 96, 300, 250, 8,  32, 2048, 100, 1, 512);  // raw_start wraps, raw_first>0
  // Phase B: persistent MXFP8 comp cache
  rc|=run_decode_mx_test("static",   64, 512, 0,   0,  40, 4096, 200, 0, 512);
  rc|=run_decode_mx_test("indexed",  64, 512, 0,   0,  40, 4096, 200, 1, 512);
  rc|=run_decode_mx_test("ring+win", 96, 300, 250, 8,  32, 2048, 100, 1, 512);
  printf("attn cutlass gemm: %s\n", rc==0?"OK":"FAIL");
  return rc;
}
#endif
