// ds4_mxfp4_cutlass.cu — CUTLASS MXFP4 tensor-core expert FFN for the ds4 MoE (sm_120f).
// Weights arrive pre-packed in CUTLASS B layout (from the offline converter); activations are
// packed to MXFP4 on-device at runtime. Path: pack(x) -> gate/up GEMM -> SwiGLU -> pack(mid) -> down GEMM.
// Build (standalone test):  nvcc -std=c++17 -arch=sm_120f --expt-relaxed-constexpr --expt-extended-lambda
//                           -DDS4_MXFP4_STANDALONE -I cutlass/include -I cutlass/tools/util/include ...
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "ds4_gpu.h"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;

// ---- GEMM: MXFP4 (A=activation RowMajor) x MXFP4 (B=weight ColumnMajor) -> f32 D. ----
using ElementA   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementD   = float;
using ElementC   = float;
using ElementAcc = float;
using LayoutA    = cutlass::layout::RowMajor;
using LayoutB    = cutlass::layout::ColumnMajor;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
constexpr int AlignA = 32, AlignB = 32;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
using TileShape    = Shape<_128,_128,_128>;
using ClusterShape = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = cutlass::float_ue8m0_t;

// ---- device activation packer (validated): f32 [M,K] RowMajor -> A data (packed E2M1) + SF (swizzled) ----
__device__ __constant__ float d_kE2M1[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f, 0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
__device__ __forceinline__ uint8_t d_to_e2m1(float v){ float best=1e30f; uint8_t bn=0; for(uint8_t n=0;n<16;n++){ float d=fabsf(v-d_kE2M1[n]); if(d<best){best=d;bn=n;} } return bn; }
template<class TSFA>
__global__ void pack_act_rowmajor(uint8_t *A_data, TSFA tSFA, const float *act, int M, int K){
  int nblk=K/32; long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)M*nblk) return;
  int m=(int)(idx/nblk), kb=(int)(idx%nblk);
  const float *x=act+(size_t)m*K+(size_t)kb*32;
  float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(x[i]));
  int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-30)e=-30; if(e>30)e=30;
  float scale=exp2f((float)e);
  uint8_t *outb=A_data+((size_t)m*K+(size_t)kb*32)/2;
  for(int i=0;i<16;i++){ uint8_t lo=d_to_e2m1(x[2*i]/scale), hi=d_to_e2m1(x[2*i+1]/scale); outb[i]=(uint8_t)(lo|(hi<<4)); }
  tSFA(m, kb*32, 0)=ElementSF::bitcast((uint8_t)(e+127));
}
// mid = silu(clamp(gate)) * clamp(up) * routing_weight  — matches engine ds4_cuda.cu:10827-10835
__global__ void swiglu_kernel(float *mid, const float *gate, const float *up, const float *w, float clamp, int mid_dim, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float g=gate[i], u=up[i];
  if(clamp>1.0e-6f){ if(g>clamp)g=clamp; if(u>clamp)u=clamp; if(u<-clamp)u=-clamp; }
  float s=g/(1.f+expf(-g));
  mid[i]=s*u*w[i/mid_dim];
}

static void pack_activation(uint8_t *A_data, ElementSF *A_sf, const float *x, int M, int K){
  auto layoutSF = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(A_sf), layoutSF);
  int nb=M*(K/32), t=128, b=(nb+t-1)/t;
  pack_act_rowmajor<<<b,t>>>(A_data, tSFA, x, M, K);
}

static typename Gemm::Arguments make_gemm_args(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                    const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K){
  auto strideA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M,K,1});
  auto strideB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto strideC = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M,N,1});
  auto strideD = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M,N,1});
  auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  return typename Gemm::Arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
    { reinterpret_cast<const ElementA::DataType*>(A_data), strideA,
      reinterpret_cast<const ElementB::DataType*>(B_data), strideB,
      A_sf, lSFA, B_sf, lSFB },
    { {1.0f, 0.0f}, D, strideC, D, strideD } };   // C=D ptr (beta=0, unused) to keep epilogue happy
}

// Workspace bytes CUTLASS needs for a GEMM of this shape (queried once per distinct
// (M,N,K) at scratch-sizing time; NOT malloc'd here -- caller folds this into one
// scratch allocation so the hot path never touches the CUDA allocator).
static size_t gemm_workspace_bytes(int M, int N, int K){
  auto args = make_gemm_args(nullptr,nullptr,nullptr,nullptr,nullptr,M,N,K);
  return Gemm::get_workspace_size(args);
}

// One MXFP4 GEMM: D[M,N] = A[M,K](act,packed) . B[N,K](weight,packed). A_sf/B_sf = swizzled SF
// buffers. workspace must be at least gemm_workspace_bytes(M,N,K) bytes, caller-owned.
static int run_gemm(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                    const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K,
                    void *workspace){
  auto args = make_gemm_args(D,A_data,A_sf,B_data,B_sf,M,N,K);
  Gemm gemm;
  if (gemm.can_implement(args)!=cutlass::Status::kSuccess) return 1;
  if (gemm.initialize(args, workspace)!=cutlass::Status::kSuccess) return 2;
  auto st = gemm.run();
  return st==cutlass::Status::kSuccess ? 0 : 3;
}

// ---- Scratch layout shared by the sizing and execution paths (must stay in lock-step). ----
// Everything the FFN needs beyond the weight/activation pointers it's called with: packed
// activation buffers, their SF tables, the three GEMMs' float outputs, and the three GEMMs'
// (usually zero-size, but not guaranteed) CUTLASS workspaces.
struct ds4_cutlass_ffn_scratch_layout {
  size_t xA_off, xSF_off, midA_off, midSF_off, gate_off, up_off, mid_off;
  size_t ws_gate_off, ws_up_off, ws_down_off;
  size_t xSF_n, midSF_n;
  size_t ws_gate_bytes, ws_up_bytes, ws_down_bytes;
  size_t total_bytes;
};

static size_t align_up_bytes(size_t n, size_t a){ return (n + a - 1) / a * a; }

static ds4_cutlass_ffn_scratch_layout cutlass_ffn_scratch_layout(int T, int in_dim, int mid_dim, int out_dim){
  ds4_cutlass_ffn_scratch_layout L{};
  const size_t align = 256;
  size_t off = 0;
  L.xA_off = off; off = align_up_bytes(off + (size_t)T*in_dim/2, align);
  L.xSF_n = (size_t)((T+127)/128*128)*((in_dim/32+3)/4*4);
  L.xSF_off = off; off = align_up_bytes(off + L.xSF_n*sizeof(ElementSF), align);
  L.midA_off = off; off = align_up_bytes(off + (size_t)T*mid_dim/2, align);
  L.midSF_n = (size_t)((T+127)/128*128)*((mid_dim/32+3)/4*4);
  L.midSF_off = off; off = align_up_bytes(off + L.midSF_n*sizeof(ElementSF), align);
  L.gate_off = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.up_off   = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.mid_off  = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.ws_gate_bytes = gemm_workspace_bytes(T, mid_dim, in_dim);
  L.ws_up_bytes   = L.ws_gate_bytes;   // identical (M,N,K) shape to gate
  L.ws_down_bytes = gemm_workspace_bytes(T, out_dim, mid_dim);
  L.ws_gate_off = off; off = align_up_bytes(off + L.ws_gate_bytes, align);
  L.ws_up_off   = off; off = align_up_bytes(off + L.ws_up_bytes, align);
  L.ws_down_off = off; off = align_up_bytes(off + L.ws_down_bytes, align);
  L.total_bytes = off;
  return L;
}

// Scratch bytes ds4_cutlass_expert_ffn_scratch() needs for the worst case of T tokens routed
// to a single expert. Callers size one buffer for their layer's (in_dim,mid_dim,out_dim) shape
// at T=max_tokens_per_expert once (e.g. via cuda_tmp_alloc) and reuse it across every expert
// and every layer that shares that shape -- this is deliberately NOT malloc'd per call.
extern "C" size_t ds4_cutlass_expert_ffn_scratch_bytes(int T, int in_dim, int mid_dim, int out_dim){
  return cutlass_ffn_scratch_layout(T, in_dim, mid_dim, out_dim).total_bytes;
}

// Core FFN, no allocation and no synchronization: out[T,out_dim] = down(swiglu(x.Wg^T, x.Wu^T)).Wd^T.
// Weights pre-packed (data+sf) in B layout; x is f32 [T,in_dim]. `scratch` must be at least
// ds4_cutlass_expert_ffn_scratch_bytes(T,in_dim,mid_dim,out_dim) bytes, 256-byte aligned.
// Launches into the caller's current stream (legacy default stream); the caller is responsible
// for ordering/synchronizing against its own subsequent work, same as every other kernel in
// this engine's decode/prefill graphs.
// SF weight pointers are typed as raw bytes at this extern "C" boundary (not ElementSF*) so
// callers outside this TU -- e.g. ds4_cuda_moe.cu, which has no CUTLASS header visibility --
// can call this without depending on CUTLASS types. ElementSF (cutlass::float_ue8m0_t) is a
// 1-byte POD; the reinterpret below is exact.
extern "C" int ds4_cutlass_expert_ffn_scratch(
    float *out, const float *x,
    const uint8_t *Wg_d, const uint8_t *Wg_sf,
    const uint8_t *Wu_d, const uint8_t *Wu_sf,
    const uint8_t *Wd_d, const uint8_t *Wd_sf,
    const float *weights, float clamp,
    int T, int in_dim, int mid_dim, int out_dim,
    uint8_t *scratch, size_t scratch_bytes){
  ds4_cutlass_ffn_scratch_layout L = cutlass_ffn_scratch_layout(T, in_dim, mid_dim, out_dim);
  if (!scratch || scratch_bytes < L.total_bytes) return -1;
  const ElementSF *Wg_sf_e = reinterpret_cast<const ElementSF*>(Wg_sf);
  const ElementSF *Wu_sf_e = reinterpret_cast<const ElementSF*>(Wu_sf);
  const ElementSF *Wd_sf_e = reinterpret_cast<const ElementSF*>(Wd_sf);
  uint8_t *xA = scratch + L.xA_off;
  ElementSF *xSF = reinterpret_cast<ElementSF*>(scratch + L.xSF_off);
  uint8_t *midA = scratch + L.midA_off;
  ElementSF *midSF = reinterpret_cast<ElementSF*>(scratch + L.midSF_off);
  float *gate = reinterpret_cast<float*>(scratch + L.gate_off);
  float *up   = reinterpret_cast<float*>(scratch + L.up_off);
  float *mid  = reinterpret_cast<float*>(scratch + L.mid_off);
  void *ws_gate = L.ws_gate_bytes ? scratch + L.ws_gate_off : nullptr;
  void *ws_up   = L.ws_up_bytes   ? scratch + L.ws_up_off   : nullptr;
  void *ws_down = L.ws_down_bytes ? scratch + L.ws_down_off : nullptr;
  if (L.xSF_n) cudaMemset(xSF,0,L.xSF_n*sizeof(ElementSF));
  if (L.midSF_n) cudaMemset(midSF,0,L.midSF_n*sizeof(ElementSF));
  int rc=0;
  pack_activation(xA,xSF,x,T,in_dim);
  rc|=run_gemm(gate,xA,xSF,Wg_d,Wg_sf_e,T,mid_dim,in_dim,ws_gate);
  rc|=run_gemm(up,  xA,xSF,Wu_d,Wu_sf_e,T,mid_dim,in_dim,ws_up);
  { long n=(long)T*mid_dim; int t=256,b=(int)((n+t-1)/t); swiglu_kernel<<<b,t>>>(mid,gate,up,weights,clamp,mid_dim,n); }
  pack_activation(midA,midSF,mid,T,mid_dim);
  rc|=run_gemm(out, midA,midSF,Wd_d,Wd_sf_e,T,out_dim,mid_dim,ws_down);
  return rc;
}

// ---- extern-C expert FFN (standalone/test convenience): allocates+frees its own scratch and
// synchronizes before returning. The engine never calls this -- it calls the scratch variant
// above with a pre-sized, reused buffer, since a per-call cudaMalloc/cudaFree/cudaDeviceSynchronize
// here is exactly the hot-path cost the engine path exists to avoid. ----
extern "C" int ds4_cutlass_expert_ffn(
    float *out, const float *x,
    const uint8_t *Wg_d, const ElementSF *Wg_sf,
    const uint8_t *Wu_d, const ElementSF *Wu_sf,
    const uint8_t *Wd_d, const ElementSF *Wd_sf,
    const float *weights, float clamp,
    int T, int in_dim, int mid_dim, int out_dim){
  size_t scratch_bytes = ds4_cutlass_expert_ffn_scratch_bytes(T, in_dim, mid_dim, out_dim);
  uint8_t *scratch=nullptr;
  cudaMalloc(&scratch, scratch_bytes);
  int rc = ds4_cutlass_expert_ffn_scratch(out,x,Wg_d,reinterpret_cast<const uint8_t*>(Wg_sf),
                                          Wu_d,reinterpret_cast<const uint8_t*>(Wu_sf),
                                          Wd_d,reinterpret_cast<const uint8_t*>(Wd_sf),
                                          weights,clamp,
                                          T,in_dim,mid_dim,out_dim,scratch,scratch_bytes);
  cudaDeviceSynchronize();
  cudaFree(scratch);
  return rc;
}

// Pack SOURCE-format MXFP4 (separate E2M1 [N,K/2] row-major + E8M0 [N,K/32]) — exactly as the
// DeepSeek-V4-Flash source stores rich experts — into CUTLASS B layout (ColumnMajor packed E2M1
// data + swizzled SFB). Host-side, lossless (copies nibbles+scale verbatim). This is the permanent
// source->CUTLASS packer; nothing consumes ds4's 17-byte format.
extern "C" void ds4_cutlass_pack_source(uint8_t *Bd, ElementSF *Bsf, const uint8_t *e2m1, const uint8_t *e8m0, int N, int K){
  auto lB   = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto layB = make_layout(make_shape(N,K,1), lB);
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  auto tB   = make_tensor(recast_ptr<cutlass::float_e2m1_t>(Bd), layB);
  auto tSFB = make_tensor(Bsf, lSFB);
  int nblk=K/32, rowbytes=K/2;
  for(int n=0;n<N;n++) for(int kb=0;kb<nblk;kb++){
    tSFB(n, kb*32, 0) = ElementSF::bitcast(e8m0[(size_t)n*nblk + kb]);   // E8M0 scale
    const uint8_t *row = e2m1 + (size_t)n*rowbytes + kb*16;              // E2M1: 16 bytes = 32 nibbles
    for(int i=0;i<16;i++){ uint8_t byte=row[i]; int k=kb*32+i*2;
      tB(n,k,0)   = cutlass::float_e2m1_t::bitcast(byte & 0xF);
      tB(n,k+1,0) = cutlass::float_e2m1_t::bitcast(byte >> 4); }
  }
}

// Physical element count of the swizzled SF tensor for a weight of shape (N=out, K=in).
extern "C" size_t ds4_cutlass_weight_sf_count(int N, int K){
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  return cute::size(cute::filter_zeros(lSFB));
}

#ifdef DS4_MXFP4_REPACK_CLI
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
// Offline converter CLI. Two modes:
//   per-expert: e2m1.bin e8m0.bin N K out_data.bin out_sf.bin
//   stacked   : --stacked e2m1_stacked.bin e8m0_stacked.bin N K n_expert out_blob.bin
//               (in: [ne,N,K/2] + [ne,N,K/32]; out: per expert data||sf, expert-major)
int main(int argc, char **argv){
  if(argc>=8 && strcmp(argv[1],"--stacked")==0){
    int N=atoi(argv[4]), K=atoi(argv[5]), ne=atoi(argv[6]);
    size_t e2each=(size_t)N*(K/2), e8each=(size_t)N*(K/32);
    std::vector<uint8_t> e2(e2each*ne), e8(e8each*ne);
    FILE*f1=fopen(argv[2],"rb"); if(!f1||fread(e2.data(),1,e2.size(),f1)!=e2.size()){ fprintf(stderr,"e2m1 read fail\n"); return 1; } fclose(f1);
    FILE*f2=fopen(argv[3],"rb"); if(!f2||fread(e8.data(),1,e8.size(),f2)!=e8.size()){ fprintf(stderr,"e8m0 read fail\n"); return 1; } fclose(f2);
    size_t sfn=ds4_cutlass_weight_sf_count(N,K);
    FILE*fo=fopen(argv[7],"wb"); if(!fo){ fprintf(stderr,"out open fail\n"); return 1; }
    std::vector<uint8_t> Bd((size_t)N*K/2); std::vector<ElementSF> Bsf(sfn);
    for(int i=0;i<ne;i++){
      std::fill(Bd.begin(),Bd.end(),(uint8_t)0);
      std::fill(Bsf.begin(),Bsf.end(),ElementSF::bitcast(127));
      ds4_cutlass_pack_source(Bd.data(), Bsf.data(), e2.data()+(size_t)i*e2each, e8.data()+(size_t)i*e8each, N, K);
      fwrite(Bd.data(),1,Bd.size(),fo); fwrite(Bsf.data(),sizeof(ElementSF),sfn,fo);
    }
    fclose(fo);
    printf("stacked N=%d K=%d ne=%d -> per-expert data=%zuB sf=%zuB total=%zuB\n",
           N,K,ne, Bd.size(), sfn*sizeof(ElementSF), (size_t)ne*(Bd.size()+sfn*sizeof(ElementSF)));
    return 0;
  }
  if(argc<7){ fprintf(stderr,"usage: %s e2m1.bin e8m0.bin N K out_data.bin out_sf.bin\n       %s --stacked e2m1_stacked.bin e8m0_stacked.bin N K n_expert out_blob.bin\n",argv[0],argv[0]); return 1; }
  int N=atoi(argv[3]), K=atoi(argv[4]);
  size_t e2n=(size_t)N*(K/2), e8n=(size_t)N*(K/32);
  std::vector<uint8_t> e2(e2n), e8(e8n);
  FILE*f1=fopen(argv[1],"rb"); if(!f1||fread(e2.data(),1,e2n,f1)!=e2n){ fprintf(stderr,"e2m1 read fail %s\n",argv[1]); return 1; } fclose(f1);
  FILE*f2=fopen(argv[2],"rb"); if(!f2||fread(e8.data(),1,e8n,f2)!=e8n){ fprintf(stderr,"e8m0 read fail %s\n",argv[2]); return 1; } fclose(f2);
  std::vector<uint8_t> Bd((size_t)N*K/2,0);
  size_t sfn=ds4_cutlass_weight_sf_count(N,K);
  std::vector<ElementSF> Bsf(sfn, ElementSF::bitcast(127));
  ds4_cutlass_pack_source(Bd.data(), Bsf.data(), e2.data(), e8.data(), N, K);
  FILE*fd=fopen(argv[5],"wb"); fwrite(Bd.data(),1,Bd.size(),fd); fclose(fd);
  FILE*fs=fopen(argv[6],"wb"); fwrite(Bsf.data(),sizeof(ElementSF),sfn,fs); fclose(fs);
  printf("packed N=%d K=%d -> data=%zuB sf=%zuB\n",N,K,Bd.size(),sfn*sizeof(ElementSF));
  return 0;
}
#endif

#ifdef DS4_MXFP4_STANDALONE
#include <vector>
#include <random>
#include <cmath>
// host MXFP4 quant (matches device): returns dequantized value array + packs into B ColumnMajor layout.
static const float hE2M1[16]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f, 0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static uint8_t h_nib(float v){ float best=1e30f; uint8_t bn=0; for(uint8_t n=0;n<16;n++){float d=fabsf(v-hE2M1[n]); if(d<best){best=d;bn=n;}} return bn; }
// Pack weight W[out,in] (row-major float) -> B data+SF in CUTLASS ColumnMajor(B) layout for shape (N=out,K=in). Fill dq.
static void host_pack_weight(std::vector<uint8_t>& Bd, std::vector<ElementSF>& Bsf, std::vector<float>& dq,
                             const std::vector<float>& W, int N, int K){
  auto lB  = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto layB = make_layout(make_shape(N,K,1), lB);
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1)); // M unused for SFB shape? use N,K
  Bd.assign((size_t)N*K/2,0); dq.assign((size_t)N*K,0.f);
  Bsf.assign(cute::size(cute::filter_zeros(lSFB)), ElementSF::bitcast(127));
  auto tB  = make_tensor(recast_ptr<cutlass::float_e2m1_t>((uint8_t*)Bd.data()), layB);
  auto tSFB= make_tensor(Bsf.data(), lSFB);
  for(int n=0;n<N;n++) for(int kb=0;kb<K/32;kb++){
    float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(W[(size_t)n*K+kb*32+i]));
    int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-30)e=-30; if(e>30)e=30; float sc=exp2f((float)e);
    tSFB(n, kb*32, 0)=ElementSF::bitcast((uint8_t)(e+127));
    for(int i=0;i<32;i++){ int k=kb*32+i; uint8_t nb=h_nib(W[(size_t)n*K+k]/sc); tB(n,k,0)=cutlass::float_e2m1_t::bitcast(nb); dq[(size_t)n*K+k]=hE2M1[nb]*sc; }
  }
}
static void host_quant_act(const std::vector<float>& X, std::vector<float>& dq, int M, int K){
  dq.assign((size_t)M*K,0.f);
  for(int m=0;m<M;m++) for(int kb=0;kb<K/32;kb++){
    float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(X[(size_t)m*K+kb*32+i]));
    int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-30)e=-30; if(e>30)e=30; float sc=exp2f((float)e);
    for(int i=0;i<32;i++){ int k=kb*32+i; uint8_t nb=h_nib(X[(size_t)m*K+k]/sc); dq[(size_t)m*K+k]=hE2M1[nb]*sc; }
  }
}
// host: float weight [N,K] -> SOURCE-format arrays (E2M1 [N,K/2] + E8M0 [N,K/32]), same quant as host_pack_weight
static void host_to_source(const std::vector<float>& W, std::vector<uint8_t>& e2, std::vector<uint8_t>& e8, int N, int K){
  int nblk=K/32; e2.assign((size_t)N*(K/2),0); e8.assign((size_t)N*nblk,0);
  for(int n=0;n<N;n++) for(int kb=0;kb<nblk;kb++){
    float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(W[(size_t)n*K+kb*32+i]));
    int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-30)e=-30; if(e>30)e=30; float sc=exp2f((float)e);
    e8[(size_t)n*nblk+kb]=(uint8_t)(e+127);
    uint8_t* row=e2.data()+(size_t)n*(K/2)+kb*16;
    for(int i=0;i<16;i++){ int k=kb*32+i*2; uint8_t lo=h_nib(W[(size_t)n*K+k]/sc), hi=h_nib(W[(size_t)n*K+k+1]/sc); row[i]=(uint8_t)(lo|(hi<<4)); }
  }
}
int main(int argc,char**argv){
  int T=256,in_dim=2048,mid_dim=1408,out_dim=2048;
  if(argc>=5){T=atoi(argv[1]);in_dim=atoi(argv[2]);mid_dim=atoi(argv[3]);out_dim=atoi(argv[4]);}
  std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,1.f);
  std::vector<float> X((size_t)T*in_dim), Wg((size_t)mid_dim*in_dim), Wu((size_t)mid_dim*in_dim), Wd((size_t)out_dim*mid_dim);
  for(auto&v:X)v=nd(rng); for(auto&v:Wg)v=nd(rng)*0.3f; for(auto&v:Wu)v=nd(rng)*0.3f; for(auto&v:Wd)v=nd(rng)*0.3f;
  // host-pack weights + dequant refs
  std::vector<uint8_t> Wgd,Wud,Wdd; std::vector<ElementSF> Wgsf,Wusf,Wdsf; std::vector<float> dqWg,dqWu,dqWd,dqX;
  host_pack_weight(Wgd,Wgsf,dqWg,Wg,mid_dim,in_dim);
  host_pack_weight(Wud,Wusf,dqWu,Wu,mid_dim,in_dim);
  host_pack_weight(Wdd,Wdsf,dqWd,Wd,out_dim,mid_dim);
  host_quant_act(X,dqX,T,in_dim);
  // validate source packer: float->(E2M1,E8M0)->CUTLASS must equal float->CUTLASS (byte-identical)
  { std::vector<uint8_t> e2,e8; host_to_source(Wg,e2,e8,mid_dim,in_dim);
    std::vector<uint8_t> Bd2(Wgd.size(),0); std::vector<ElementSF> Bsf2(Wgsf.size(),ElementSF::bitcast(127));
    ds4_cutlass_pack_source(Bd2.data(),Bsf2.data(),e2.data(),e8.data(),mid_dim,in_dim);
    int dd=memcmp(Bd2.data(),Wgd.data(),Wgd.size());
    int ds=memcmp(Bsf2.data(),Wgsf.data(),Wgsf.size()*sizeof(ElementSF));
    printf("pack_source(E2M1+E8M0->CUTLASS) check: data=%s sf=%s\n", dd==0?"MATCH":"DIFFER", ds==0?"MATCH":"DIFFER"); }
  // reference FFN using the same quantized values
  std::vector<float> gate((size_t)T*mid_dim),up((size_t)T*mid_dim),mid((size_t)T*mid_dim),dqMid,ref((size_t)T*out_dim,0.f);
  std::vector<float> W_route(T); for(auto&v:W_route) v=0.5f+std::abs(nd(rng))*0.5f;  // per-token routing weight
  float clamp = 4.0f;
  for(int t=0;t<T;t++)for(int j=0;j<mid_dim;j++){ double g=0,u=0; for(int k=0;k<in_dim;k++){ g+=(double)dqX[(size_t)t*in_dim+k]*dqWg[(size_t)j*in_dim+k]; u+=(double)dqX[(size_t)t*in_dim+k]*dqWu[(size_t)j*in_dim+k]; } gate[(size_t)t*mid_dim+j]=(float)g; up[(size_t)t*mid_dim+j]=(float)u; }
  for(int t=0;t<T;t++)for(int j=0;j<mid_dim;j++){ size_t i=(size_t)t*mid_dim+j; float g=gate[i],u=up[i]; if(clamp>1e-6f){if(g>clamp)g=clamp;if(u>clamp)u=clamp;if(u<-clamp)u=-clamp;} float s=g/(1.f+expf(-g)); mid[i]=s*u*W_route[t]; }
  host_quant_act(mid,dqMid,T,mid_dim);
  for(int t=0;t<T;t++)for(int o=0;o<out_dim;o++){ double a=0; for(int j=0;j<mid_dim;j++) a+=(double)dqMid[(size_t)t*mid_dim+j]*dqWd[(size_t)o*mid_dim+j]; ref[(size_t)t*out_dim+o]=(float)a; }
  // device buffers
  float *dX,*dOut,*dWr; cudaMalloc(&dX,X.size()*4); cudaMalloc(&dOut,(size_t)T*out_dim*4); cudaMalloc(&dWr,(size_t)T*4);
  cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dWr,W_route.data(),(size_t)T*4,cudaMemcpyHostToDevice);
  uint8_t *dWg,*dWu,*dWd; ElementSF *dWgs,*dWus,*dWds;
  cudaMalloc(&dWg,Wgd.size()); cudaMalloc(&dWu,Wud.size()); cudaMalloc(&dWd,Wdd.size());
  cudaMalloc(&dWgs,Wgsf.size()*sizeof(ElementSF)); cudaMalloc(&dWus,Wusf.size()*sizeof(ElementSF)); cudaMalloc(&dWds,Wdsf.size()*sizeof(ElementSF));
  cudaMemcpy(dWg,Wgd.data(),Wgd.size(),cudaMemcpyHostToDevice); cudaMemcpy(dWu,Wud.data(),Wud.size(),cudaMemcpyHostToDevice); cudaMemcpy(dWd,Wdd.data(),Wdd.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dWgs,Wgsf.data(),Wgsf.size()*sizeof(ElementSF),cudaMemcpyHostToDevice); cudaMemcpy(dWus,Wusf.data(),Wusf.size()*sizeof(ElementSF),cudaMemcpyHostToDevice); cudaMemcpy(dWds,Wdsf.data(),Wdsf.size()*sizeof(ElementSF),cudaMemcpyHostToDevice);
  int rc=ds4_cutlass_expert_ffn(dOut,dX,dWg,dWgs,dWu,dWus,dWd,dWds,dWr,clamp,T,in_dim,mid_dim,out_dim);
  std::vector<float> got((size_t)T*out_dim); cudaMemcpy(got.data(),dOut,(size_t)T*out_dim*4,cudaMemcpyDeviceToHost);
  double maxrel=0,maxabs=0; int bad=0;
  for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-ref[i]); double r=a/(fabs(ref[i])+1e-3); if(r>maxrel)maxrel=r; if(a>maxabs)maxabs=a; if(r>0.05&&a>0.1)bad++; }
  printf("rc=%d  max_rel=%.5f max_abs=%.4f bad=%d/%zu  -> %s\n", rc,maxrel,maxabs,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  return (rc==0&&bad==0)?0:1;
}
#endif
