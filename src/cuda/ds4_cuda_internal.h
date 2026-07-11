/* ds4_cuda_internal.h — internal shared declarations for the cuda/ translation units.
 * Produced by the multi-TU split of ds4_cuda.cu; edit freely (the
 * generator is not part of the build).
 *
 * No -rdc: __global__/__device__/__constant__ symbols never cross TU
 * boundaries. The shared __device__ helpers below are static
 * __forceinline__, so each TU gets its own copy. */
#ifndef DS4_CUDA_INTERNAL_H
#define DS4_CUDA_INTERNAL_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context.  11712 fits under the
     * GB10 48 KB shared-memory limit. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 11712u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u
};

#define DS4_FP8_KV_BLOCK 64u
#define DS4_FP8_KV_NBLK(HD) (((HD) + DS4_FP8_KV_BLOCK - 1u) / DS4_FP8_KV_BLOCK)
#define DS4_FP8_KV_ROWBYTES(HD) ((HD) + DS4_FP8_KV_NBLK(HD) * sizeof(float))

/*
 * Microscaling (MX / OCP) compressed-KV storage.  One E8M0 (power-of-two)
 * scale byte per BLOCK=32 elements, laid out per row as [data ...][scales ...]:
 *   MXFP8 (E4M3 data): HD data bytes  + NBLK scale bytes  (HD=512 -> 528 B/row)
 *   MXFP4 (E2M1 data): HD/2 nibble bytes + NBLK scale bytes (HD=512 -> 272 B/row)
 * This is the CUTLASS-consumable layout (float_ue8m0_t scales, block 32); the
 * GEMM path re-tiles the scales into CUTLASS's swizzled SF layout at use time.
 */
#define DS4_MXKV_BLOCK 32u
#define DS4_MXKV_NBLK(HD) (((HD) + DS4_MXKV_BLOCK - 1u) / DS4_MXKV_BLOCK)
#define DS4_MXKV_FP8_ROWBYTES(HD) ((HD) + DS4_MXKV_NBLK(HD))
#define DS4_MXKV_FP4_ROWBYTES(HD) (((HD) + 1u) / 2u + DS4_MXKV_NBLK(HD))
/* KV cache storage format selector (compile/runtime). */
#define DS4_MXKV_FMT_NONE 0u
#define DS4_MXKV_FMT_FP8  1u
#define DS4_MXKV_FMT_FP4  2u
#define DS4_MXKV_ROWBYTES(FMT, HD) \
    ((FMT) == DS4_MXKV_FMT_FP4 ? DS4_MXKV_FP4_ROWBYTES(HD) \
   : (FMT) == DS4_MXKV_FMT_FP8 ? DS4_MXKV_FP8_ROWBYTES(HD) \
   : (HD) * sizeof(float))

/*
 * DS4_ATTN_PACK compressed-KV storage (value-preserving).  A comp row today is
 * f32 with the nope dims already fp8-roundtripped in place (fp8_kv_quantize:
 * per-64 block scale = exp2f(ceilf(log2f(fmaxf(amax,1e-4)/448))), e4m3
 * clamp-roundtrip) and the n_rot rope tail untouched f32.  The packed row
 * stores exactly those values:
 *   [n_nope e4m3 bytes][n_nope/64 E8M0 scale bytes][pad to 4B][n_rot f32]
 * head_dim 512 / n_rot 64 -> 448 + 7 + 1 + 256 = 712 B/row (vs 2048 f32).
 * The E8M0 byte is the scale exponent + 127 (power-of-two by construction),
 * so decode (e4m3 value * 2^(e8-127); rope read f32) is bit-identical to the
 * f32 cache.  Must stay in sync with DS4_ENGINE_ATTN_PACK_ROWBYTES.
 */
#define DS4_ATTN_PACK_NROT 64u
#define DS4_ATTN_PACK_NOPE(HD) ((HD) - DS4_ATTN_PACK_NROT)
#define DS4_ATTN_PACK_SCALES_PAD(HD) \
    ((DS4_ATTN_PACK_NOPE(HD) / DS4_FP8_KV_BLOCK + 3u) & ~3u)
#define DS4_ATTN_PACK_ROWBYTES(HD) \
    ((uint64_t)DS4_ATTN_PACK_NOPE(HD) + DS4_ATTN_PACK_SCALES_PAD(HD) + \
     (uint64_t)DS4_ATTN_PACK_NROT * 4u)

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

/* ---- shared types ---- */

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct fp8_mx_weight { const void *host_base; uint64_t offset, in_dim, out_dim; __nv_fp8_e4m3 *data; unsigned char *scale; };

/* ---- shared host globals ---- */

extern cublasHandle_t g_cublas;
extern int g_cublas_ready;
extern int g_quality_mode;
extern cublasLtHandle_t g_cublaslt;
extern std::unordered_set<uint64_t> g_fp8_offsets;

/* ---- shared host functions ---- */

void *cuda_tmp_alloc(uint64_t bytes, const char *what);
int cuda_attention_score_buffer_fits(uint32_t n_comp);
const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what);
int cuda_ok(cudaError_t err, const char *what);
int cublas_ok(cublasStatus_t st, const char *what);
int cuda_matmul_fp8_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label);

/* ---- shared __device__ inline helpers (per-TU copies; no -rdc) ---- */

__device__ static __forceinline__ float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static __forceinline__ float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static __forceinline__ float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

#endif /* DS4_CUDA_INTERNAL_H */
