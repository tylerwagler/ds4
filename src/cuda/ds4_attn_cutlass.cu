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

// ---- Softmax with attention sink (one block per query row) ----
// scores[n_q, n_kv] (raw dot products) -> P[n_q, n_kv] probabilities, applying
// scale = 1/sqrt(head_dim) and folding the per-query sink into the denominator
// only (exp(sink - m), no value contribution). `valid` bounds the visible keys
// per query (causal/window); keys >= valid[q] are masked out. sink is per-row
// here (single head); the batched path passes sinks[head].
__global__ void attn_softmax_sink(float *P, const float *scores, const float *sink_per_row,
                                  const int32_t *valid, float scale, uint32_t n_q, uint32_t n_kv) {
  uint32_t q = blockIdx.x;
  if (q >= n_q) return;
  const float *sr = scores + (uint64_t)q * n_kv;
  float *pr = P + (uint64_t)q * n_kv;
  uint32_t lim = valid ? (uint32_t)valid[q] : n_kv;
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
  attn_softmax_sink<<<n_q,256>>>(dP,dScores,dSink,nullptr,scale,n_q,n_kv);
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
  attn_softmax_sink<<<n_head*n_q,256>>>(dP,dScores,dSinkRow,nullptr,scale,n_head*n_q,n_kv);
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
  printf("attn cutlass gemm: %s\n", rc==0?"OK":"FAIL");
  return rc;
}
#endif
