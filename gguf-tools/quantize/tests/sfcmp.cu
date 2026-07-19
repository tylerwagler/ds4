// Compare weight-SF (SFB) and activation-SF (SFA) CUTLASS block-scaled layouts between
// the current all-fp4 (W4A4, mxf4nvf4) config and the mixed W4A8 (f8f6f4) config, to decide
// whether the type-40 weight repack (and shipped blk 0-2) stay valid under W4A8.
#include <cstdio>
#include <cstdint>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
using namespace cute;

using ElementD=float; using ElementC=float; using ElementAcc=float;
using LayoutA=cutlass::layout::RowMajor; using LayoutB=cutlass::layout::ColumnMajor;
using LayoutC=cutlass::layout::RowMajor; using LayoutD=cutlass::layout::RowMajor;
using TileShape=Shape<_128,_128,_128>; using ClusterShape=Shape<_1,_1,_1>;
constexpr int AlignC=128/cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignD=128/cutlass::sizeof_bits<ElementD>::value;

template<class EA,class EB,int AA,int AB>
struct Cfg {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape,ClusterShape,cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc,ElementAcc,ElementC,LayoutC,AlignC,ElementD,LayoutD,AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    EA,LayoutA,AA, EB,LayoutB,AB, ElementAcc, TileShape,ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Main,Epi,void>;
  using Cfg2 = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

// all-fp4 (current type-40 path)
using A4 = Cfg<cutlass::mx_float4_t<cutlass::float_e2m1_t>, cutlass::mx_float4_t<cutlass::float_e2m1_t>, 32,32>;
// mixed W4A8: A=fp8 e4m3, B=fp4 e2m1
using A8 = Cfg<cutlass::mx_float8_t<cutlass::float_e4m3_t>, cutlass::mx_float4_t<cutlass::float_e2m1_t>, 32,32>;

template<class C> void dumpB(const char*tag,int N,int K){
  auto l = C::Cfg2::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  printf("%-6s SFB shape=", tag); print(shape(l)); printf(" stride="); print(stride(l));
  printf(" filtered_size=%ld\n",(long)cute::size(cute::filter_zeros(l)));
}
template<class C> void dumpA(const char*tag,int M,int K){
  auto l = C::Cfg2::tile_atom_to_shape_SFA(make_shape(M,1,K,1));
  printf("%-6s SFA shape=", tag); print(shape(l)); printf(" stride="); print(stride(l));
  printf(" filtered_size=%ld\n",(long)cute::size(cute::filter_zeros(l)));
}

int main(){
  int N=2048,K=4096; // blk gate shape (N=out=dim1, K=in=dim0)
  printf("== weight SFB (N=%d K=%d) ==\n",N,K);
  dumpB<A4>("A4",N,K); dumpB<A8>("A8",N,K);
  int N2=4096,K2=2048; // down shape
  printf("== weight SFB down (N=%d K=%d) ==\n",N2,K2);
  dumpB<A4>("A4",N2,K2); dumpB<A8>("A8",N2,K2);
  printf("== activation SFA (M=128 K=%d) ==\n",K);
  dumpA<A4>("A4",128,K); dumpA<A8>("A8",128,K);
  printf("sizeof SFB elem A4=%zu A8=%zu\n",
    sizeof(typename A4::Kernel::CollectiveMainloop::ElementSF),
    sizeof(typename A8::Kernel::CollectiveMainloop::ElementSF));
  return 0;
}
