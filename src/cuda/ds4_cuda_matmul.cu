#include "ds4_cuda_internal.h"



__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_vocab, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    uint32_t tok = token < n_vocab ? token : n_vocab - 1; /* clamp: an OOB token id is a wild global read */
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)tok * n_embd + e]);
}



__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}



__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}


/* Small-batch (2..4 token) f16 GEMV: one weight-row read serves all NT tokens,
 * replacing the cuBLAS GemmEx path which is latency-bound at these shapes and
 * needs an f32->f16 activation convert + tmp alloc per call. Per-token loop
 * structure and shared-memory reduction match matmul_f16_kernel exactly, so each
 * token's output is bit-identical to the n=1 kernel run on that token alone. */
template <int NT>
__global__ static void matmul_f16_nt_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out_dim) return;

    float sum[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) sum[t] = 0.0f;
    const __half *wr = w + row * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const float wv = __half2float(wr[i]);
        #pragma unroll
        for (int t = 0; t < NT; t++) sum[t] += wv * x[t * in_dim + i];
    }

    __shared__ float partial[256];
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        partial[threadIdx.x] = sum[t];
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[t * out_dim + row] = partial[0];
        __syncthreads();
    }
}



__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}



__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}



__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}



__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}



__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}



__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}



/* BF16 is the high 16 bits of f32, so conversions are pure bit ops (no header). */
__device__ __forceinline__ static float bf16_to_f32(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}


__global__ static void f32_to_bf16_kernel(uint16_t *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t u = __float_as_uint(x[i]);
        out[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  /* round-to-nearest-even */
    }
}


__global__ static void matmul_bf16_kernel(
        float *out,
        const uint16_t *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    float sum = 0.0f;
    const uint16_t *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += bf16_to_f32(wr[i]) * xr[i];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}



/* MXFP8 weight block (33B: [E8M0][32 e4m3]) . raw f32 activation block.
 * Decode-side attn-output dot: the e4m3 weight is expanded to float and MAC'd
 * against the unquantized activation, so the only quantization error left on
 * this path is the weight's own (e4m3-packing the activations pushed the
 * perplexity gate; int8 requant is what this change removes). The contiguous
 * 32-element inner loop is what nvcc vectorizes — a lane-per-element
 * (mmvq-style) mapping profiled 2.2x slower. Returns wscale*sum(e4m3*x). */
__device__ __forceinline__ int mx_sfoff(int row, int kb, int KBp);   /* defined below */


/* De-interleaved MXFP8 block dot: contiguous 32-E4M3 block (aligned) + separate
 * E8M0 scale byte. Reads 8 aligned uint32 (4 fp8 each) -> vectorized, vs the raw
 * 33B interleaved kernels' misaligned byte-wise weight reads. */
__device__ __forceinline__ static float dev_dot_fp8mx_deint_block(
        const __nv_fp8_e4m3 *blk, unsigned char scale_byte, const float *xb) {
    const float wscale = __int_as_float((uint32_t)scale_byte << 23);   /* E8M0 */
    float s = 0.0f;
#pragma unroll
    for (int g = 0; g < 8; g++) {
        const uint32_t p = ((const uint32_t *)blk)[g];
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&p;
#pragma unroll
        for (int j = 0; j < 4; j++) s += __half2float((__half)q[j]) * xb[g * 4 + j];
    }
    return wscale * s;
}


__device__ __forceinline__ static float dev_dot_fp8mx_f32_block(
        const unsigned char *wblk, const float *x, uint64_t bn) {
    const float wscale = __int_as_float((uint32_t)wblk[0] << 23);   /* E8M0 */
    float s = 0.0f;
    for (uint64_t i = 0; i < bn; i++) {
        const __half wh = (__half)(*(const __nv_fp8_e4m3 *)&wblk[1 + i]);
        s += __half2float(wh) * x[i];
    }
    return wscale * s;
}



/* Same dot against an activation block already staged in registers (full
 * 32-element block only). */
__device__ __forceinline__ static float dev_dot_fp8mx_xreg_block(
        const unsigned char *wblk, const float *xb) {
    const float wscale = __int_as_float((uint32_t)wblk[0] << 23);   /* E8M0 */
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const __half wh = (__half)(*(const __nv_fp8_e4m3 *)&wblk[1 + i]);
        s += __half2float(wh) * xb[i];
    }
    return wscale * s;
}



/* Fused attn-output kernels: MXFP8 weight rows dotted against RAW f32
 * activations. Each warp covers DS4_FP8MX_ROWS consecutive output rows and
 * loads every 32-float activation block into registers ONCE for all of them,
 * so per-row activation traffic drops to int8-path parity (per-row f32 reads
 * straight from global measured ~13% slower end-to-end at decode; a shared-
 * memory staging variant measured worse still). */
enum { DS4_FP8MX_ROWS = 4 };



template<bool DEINT>
__global__ static void matmul_fp8mx_hc_expand_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * DS4_FP8MX_ROWS;
    const uint32_t lane = threadIdx.x & 31u;
    if (row0 >= out_dim) return;
    const uint32_t nr = out_dim - row0 < DS4_FP8MX_ROWS ? (uint32_t)(out_dim - row0)
                                                        : (uint32_t)DS4_FP8MX_ROWS;
    const unsigned char *wr = w + row0 * blocks * 33u;
    float acc[DS4_FP8MX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        if (bn == 32u) {
            float xb[32];
#pragma unroll
            for (int k = 0; k < 8; k++) {
                *(float4 *)&xb[k * 4] = *(const float4 *)&x[i0 + (uint32_t)k * 4u];
            }
#pragma unroll
            for (uint32_t r = 0; r < DS4_FP8MX_ROWS; r++) {
                if (r >= nr) continue;
                if (DEINT) {
                    const uint32_t rw = (uint32_t)(row0 + r);
                    acc[r] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * in_dim + i0,
                                                        wscale[mx_sfoff((int)rw, (int)b, KBp)], xb);
                } else {
                    acc[r] += dev_dot_fp8mx_xreg_block(wr + (r * blocks + b) * 33u, xb);
                }
            }
        } else {
            for (uint32_t r = 0; r < nr; r++) {
                acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, x + i0, bn);
            }
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
        const float red = warp_sum_f32(acc[r]);
        if (lane != 0) continue;
        const uint32_t d = (uint32_t)(row0 + r);
        block_out[d] = red;
        float block_v = red;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}



template<bool DEINT>
__global__ static void grouped_fp8mx_a_warp8_kernel(
        float *low,
        const unsigned char *w,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const float *x,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * DS4_FP8MX_ROWS;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim || tok >= n_tokens) return;
    const uint32_t nr = low_dim - row0 < DS4_FP8MX_ROWS ? (uint32_t)(low_dim - row0)
                                                        : (uint32_t)DS4_FP8MX_ROWS;
    const unsigned char *wr = w + row0 * blocks * 33u;
    const uint64_t group = row0 / rank;
    float acc[DS4_FP8MX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    if ((row0 + nr - 1) / rank == group) {
        /* Common case (rank % DS4_FP8MX_ROWS == 0): all rows share the
         * group's activation row, so its blocks are loaded once per warp. */
        const float *xr = x + (tok * (uint64_t)n_groups + group) * group_dim;
        for (uint64_t b = lane; b < blocks; b += 32u) {
            const uint64_t i0 = b * 32;
            const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
            if (bn == 32u) {
                float xb[32];
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    *(float4 *)&xb[k * 4] = *(const float4 *)&xr[i0 + (uint32_t)k * 4u];
                }
#pragma unroll
                for (uint32_t r = 0; r < DS4_FP8MX_ROWS; r++) {
                    if (r >= nr) continue;
                    if (DEINT) {
                        const uint32_t rw = (uint32_t)(row0 + r);
                        acc[r] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * group_dim + i0,
                                                            wscale[mx_sfoff((int)rw, (int)b, KBp)], xb);
                    } else {
                        acc[r] += dev_dot_fp8mx_xreg_block(wr + (r * blocks + b) * 33u, xb);
                    }
                }
            } else {
                for (uint32_t r = 0; r < nr; r++) {
                    acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, xr + i0, bn);
                }
            }
        }
    } else {
        for (uint32_t r = 0; r < nr; r++) {
            const float *xr = x + (tok * (uint64_t)n_groups + (row0 + r) / rank) * group_dim;
            for (uint64_t b = lane; b < blocks; b += 32u) {
                const uint64_t i0 = b * 32;
                const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
                acc[r] += dev_dot_fp8mx_f32_block(wr + (r * blocks + b) * 33u, xr + i0, bn);
            }
        }
    }
    for (uint32_t r = 0; r < nr; r++) {
        const float red = warp_sum_f32(acc[r]);
        if (lane == 0) low[tok * low_dim + row0 + r] = red;
    }
}


/* Small-batch (2..4 token) variant of the grouped o_a GEMV: one launch whose
 * weight blocks are read once and served from L1 across the NT tokens' dots,
 * replacing the per-group block-scaled tensor-core GEMMs that dominated the
 * spec-verify launch storm (8 GEMMs/layer at 2-4 rows each). Per-(row,token)
 * block order and dot helper match grouped_fp8mx_a_warp8_kernel's DEINT fast
 * path exactly, so each token's output is bit-identical to the n=1 kernel.
 * Caller guarantees: deint weight available, rank % DS4_FP8MX_ROWS == 0 (row
 * quads never straddle a group), group_dim % 32 == 0 (whole blocks). */
template <int NT>
__global__ static void grouped_fp8mx_a_nt_kernel(
        float *low,
        const __nv_fp8_e4m3 *wdata,
        const unsigned char *wscale,
        int KBp,
        const float *x,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t blocks) {
    const uint64_t row0 = ((uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u)) * DS4_FP8MX_ROWS;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row0 >= low_dim) return;
    const uint64_t group = row0 / rank;
    float acc[DS4_FP8MX_ROWS][NT];
#pragma unroll
    for (uint32_t r = 0; r < DS4_FP8MX_ROWS; r++)
#pragma unroll
        for (int t = 0; t < NT; t++) acc[r][t] = 0.0f;
    const float *xg = x + group * group_dim;
    const uint64_t tok_stride = (uint64_t)n_groups * group_dim;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float xb[32];
            const float *xr = xg + (uint64_t)t * tok_stride + i0;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                *(float4 *)&xb[k * 4] = *(const float4 *)&xr[(uint32_t)k * 4u];
            }
#pragma unroll
            for (uint32_t r = 0; r < DS4_FP8MX_ROWS; r++) {
                const uint32_t rw = (uint32_t)(row0 + r);
                acc[r][t] += dev_dot_fp8mx_deint_block(wdata + (uint64_t)rw * group_dim + i0,
                                                       wscale[mx_sfoff((int)rw, (int)b, KBp)], xb);
            }
        }
    }
#pragma unroll
    for (uint32_t r = 0; r < DS4_FP8MX_ROWS; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float red = warp_sum_f32(acc[r][t]);
            if (lane == 0) low[(uint64_t)t * low_dim + row0 + r] = red;
        }
    }
}



extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || weight_offset >= model_size || n_vocab == 0) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_vocab, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}



extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}


static int g_cublaslt_ready = 0;


static int cublaslt_ensure(void) {
    if (g_cublaslt_ready) return 1;
    if (cublasLtCreate(&g_cublaslt) != CUBLAS_STATUS_SUCCESS) return 0;
    g_cublaslt_ready = 1; return 1;
}


static inline int mx_rup(int x, int n) { return (x + n - 1) / n * n; }


__device__ __forceinline__ int mx_sfoff(int row, int kb, int KBp) {
    return ((row / 128) * (KBp / 4) + (kb / 4)) * 512
           + (row % 32) * 16 + ((row % 128) / 32) * 4 + (kb % 4);
}


/* X[rows,K] f32 -> E4M3 data[K,rows]col + swizzled E8M0 scale. one warp per (row,kb). */
__global__ static void mxfp8_quant_act_kernel(const float *X, int rows, int K, int KBp,
                                              __nv_fp8_e4m3 *data, unsigned char *scale) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = K / 32; if (warp >= rows * KB) return;
    int row = warp / KB, kb = warp % KB;
    float v = X[(size_t)row * K + kb * 32 + lane], a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int se = -127; if (a > 0.f) { int e = (int)floorf(log2f(a)); se = e - 7; }
    if (se < -127) se = -127; if (se > 127) se = 127;
    data[(size_t)(kb * 32 + lane) + (size_t)row * K] = (__nv_fp8_e4m3)(v * exp2f((float)-se));
    if (lane == 0) scale[mx_sfoff(row, kb, KBp)] = (unsigned char)(se + 127);
}


/* Grouped variant for the attn-output "a" projection. The activation tensor is
 * heads[tok][group][K] but each group is an independent GEMM, so the E4M3 data
 * is regrouped as n_groups slabs of [K, n_tokens] col-major and the E8M0 scale
 * as n_groups independent swizzles of scale_slab bytes (token = scale row). */
__global__ static void mxfp8_quant_act_grouped_kernel(const float *X, int n_tokens, int n_groups,
                                                      int K, int KBp, __nv_fp8_e4m3 *data,
                                                      unsigned char *scale, size_t scale_slab) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = K / 32; if (warp >= n_tokens * n_groups * KB) return;
    int row = warp / KB, kb = warp % KB;        /* row = tok * n_groups + g */
    int g = row % n_groups, tok = row / n_groups;
    float v = X[(size_t)row * K + kb * 32 + lane], a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int se = -127; if (a > 0.f) { int e = (int)floorf(log2f(a)); se = e - 7; }
    if (se < -127) se = -127; if (se > 127) se = 127;
    data[((size_t)g * n_tokens + tok) * K + kb * 32 + lane] = (__nv_fp8_e4m3)(v * exp2f((float)-se));
    if (lane == 0) scale[(size_t)g * scale_slab + mx_sfoff(tok, kb, KBp)] = (unsigned char)(se + 127);
}


/* GGUF MXFP8 weight (row-major [out,in], 33B blocks: [E8M0][32xE4M3]) ->
 * E4M3 data[in,out]col + swizzled E8M0 scale. one warp per (out,kb). */
__global__ static void mxfp8_weight_convert_kernel(const unsigned char *src, int out_dim, int in_dim,
                                                   int KBp, __nv_fp8_e4m3 *data, unsigned char *scale) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    int KB = in_dim / 32; if (warp >= out_dim * KB) return;
    int o = warp / KB, kb = warp % KB;
    const unsigned char *blk = src + ((size_t)o * KB + kb) * 33;
    data[(size_t)(kb * 32 + lane) + (size_t)o * in_dim] = *(const __nv_fp8_e4m3 *)&blk[1 + lane];
    if (lane == 0) scale[mx_sfoff(o, kb, KBp)] = blk[0];
}


static std::unordered_map<uint64_t, fp8_mx_weight> g_fp8_mx_by_offset;


/* lazily de-interleave + swizzle an MXFP8 weight into device buffers, cached by offset. */
static const fp8_mx_weight *cuda_fp8_mx_weight(const void *model_map, uint64_t offset, uint64_t weight_bytes,
                                               uint64_t in_dim, uint64_t out_dim, const char *label) {
    /* The same ~300 weight offsets are resolved once per layer every token on
     * the launch-serializing host thread. A tiny direct-mapped cache in front
     * of the unordered_map skips the probe on the hot repeat; a miss or hash
     * collision just falls through (benign), and the cached pointer is
     * re-validated (map references are stable across inserts). */
    constexpr uint32_t FC = 2048u;
    static uint64_t fc_off[FC];              /* zero-init; real offsets are never 0 */
    static const fp8_mx_weight *fc_ptr[FC];
    const uint32_t slot = (uint32_t)(((offset >> 5) ^ (offset >> 17)) & (FC - 1u));
    if (offset != 0 && fc_off[slot] == offset) {
        const fp8_mx_weight *p = fc_ptr[slot];
        if (p && p->host_base == model_map && p->in_dim == in_dim && p->out_dim == out_dim) return p;
    }
    auto it = g_fp8_mx_by_offset.find(offset);
    if (it != g_fp8_mx_by_offset.end() && it->second.host_base == model_map &&
        it->second.in_dim == in_dim && it->second.out_dim == out_dim) {
        fc_off[slot] = offset; fc_ptr[slot] = &it->second;
        return &it->second;
    }
    const unsigned char *src = (const unsigned char *)cuda_model_range_ptr(model_map, offset, weight_bytes, "fp8_mx");
    if (!src) return NULL;
    int KB = (int)(in_dim / 32), KBp = mx_rup(KB, 4);
    size_t data_bytes = in_dim * out_dim;
    size_t scale_bytes = (size_t)mx_rup((int)out_dim, 128) * KBp;
    __nv_fp8_e4m3 *data = NULL; unsigned char *scale = NULL;
    if (cudaMalloc(&data, data_bytes) != cudaSuccess) return NULL;
    if (cudaMalloc(&scale, scale_bytes) != cudaSuccess) { cudaFree(data); return NULL; }
    cudaMemset(scale, 0, scale_bytes);
    int warps = (int)out_dim * KB;
    mxfp8_weight_convert_kernel<<<(warps * 32 + 255) / 256, 256>>>(src, (int)out_dim, (int)in_dim, KBp, data, scale);
    if (!cuda_ok(cudaGetLastError(), "fp8_mx weight convert")) { cudaFree(data); cudaFree(scale); return NULL; }
    fp8_mx_weight w = { model_map, offset, in_dim, out_dim, data, scale };
    g_fp8_mx_by_offset[offset] = w;
    (void)label;
    const fp8_mx_weight *wp = &g_fp8_mx_by_offset[offset];
    fc_off[slot] = offset; fc_ptr[slot] = wp;
    return wp;
}


static int cuda_matmul_fp8_mx_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map || in_dim % 32 != 0 || !cublaslt_ensure()) return 0;
    uint64_t KB = in_dim / 32, weight_bytes = out_dim * KB * 33;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) || out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
    if (!w) return 0;
    int ntok = (int)n_tok, KBp = mx_rup((int)KB, 4);
    size_t sx_bytes = (size_t)mx_rup(ntok, 128) * KBp;
    size_t wz = 32u << 20;
    /* xq, sx and the cuBLASLt workspace must be DISTINCT buffers, but
     * cuda_tmp_alloc hands out one shared scratch region (later calls alias or
     * realloc/free earlier ones). Carve three non-overlapping, 256-aligned
     * regions from a single allocation instead. */
    size_t off_xq = 0;
    size_t off_sx = (in_dim * (size_t)ntok + 255) & ~(size_t)255;
    size_t off_ws = (off_sx + sx_bytes + 255) & ~(size_t)255;
    char *scratch = (char *)cuda_tmp_alloc(off_ws + wz, "fp8_mx scratch");
    if (!scratch) return 0;
    __nv_fp8_e4m3 *xq = (__nv_fp8_e4m3 *)(scratch + off_xq);
    unsigned char *sx = (unsigned char *)(scratch + off_sx);
    void *ws = scratch + off_ws;
    cudaMemset(sx, 0, sx_bytes);
    int warps = ntok * (int)KB;
    mxfp8_quant_act_kernel<<<(warps * 32 + 255) / 256, 256>>>((const float *)x->ptr, ntok, (int)in_dim, KBp, xq, sx);
    if (!cuda_ok(cudaGetLastError(), "fp8_mx act quant")) return 0;
    cublasLtMatmulDesc_t op; if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 0;
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t mo = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mo, sizeof(mo));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mo, sizeof(mo));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, in_dim, out_dim, in_dim);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, in_dim, ntok, in_dim);
    cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, out_dim, ntok, out_dim);
    cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
    cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
    cublasLtMatmulHeuristicResult_t h; int got = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, op, la, lb, ld, ld, pf, 1, &h, &got);
    int ok = 0;
    if (hs == CUBLAS_STATUS_SUCCESS && got && ws) {
        float al = 1.f, be = 0.f;
        cublasStatus_t st = cublasLtMatmul(g_cublaslt, op, &al, w->data, la, xq, lb, &be,
                                           out->ptr, ld, out->ptr, ld, &h.algo, ws, wz, 0);
        ok = (st == CUBLAS_STATUS_SUCCESS);
        if (!ok) fprintf(stderr, "ds4: cuBLASLt MXFP8 matmul failed: status %d\n", (int)st);
    }
    cublasLtMatmulPreferenceDestroy(pf);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
    return ok;
}


extern "C" int ds4_gpu_matmul_fp8_mx_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_fp8_mx_tensor_labeled(out, model_map, model_size, weight_offset,
                                             in_dim, out_dim, x, n_tok, "fp8_mx");
}



/* Prefill attn-output "a" projection as n_groups block-scaled MXFP8xMXFP8 GEMMs.
 *
 * The projection is block-diagonal: group g's [rank, group_dim] weight slice only
 * sees activation rows (tok, g). cuBLASLt's VEC32_UE8M0 scale layout tiles rows in
 * 128-row blocks, so a batched/strided formulation can't share one swizzled scale
 * buffer across groups; instead the weight cache is sliced per group (exact when
 * rank % 128 == 0: both data columns and scale tiles are contiguous per group)
 * and the activations are quantized into per-group data + scale slabs. Each of
 * the n_groups (8/16) GEMMs writes straight into low[tok][g*rank + r] via ldd,
 * so no epilogue pass is needed. */
static int cuda_attention_output_a_mx_gemm(
        ds4_gpu_tensor *low,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    if (group_dim % 32 != 0 || rank % 128 != 0 || !cublaslt_ensure()) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t KB = group_dim / 32;
    const uint64_t weight_bytes = low_dim * KB * 33;
    if (out_a_offset > model_size || weight_bytes > model_size - out_a_offset) return 0;
    const fp8_mx_weight *w = cuda_fp8_mx_weight(model_map, out_a_offset, weight_bytes,
                                                group_dim, low_dim, "attn_out_a");
    if (!w) return 0;
    const int KBp = mx_rup((int)KB, 4);
    const size_t x_scale_slab = (size_t)mx_rup((int)n_tokens, 128) * KBp;
    const size_t w_scale_slab = ((size_t)rank / 128) * (size_t)KBp * 128;
    const size_t data_bytes = (size_t)n_tokens * n_groups * group_dim;
    const size_t scale_bytes = (size_t)n_groups * x_scale_slab;
    size_t wz = 32u << 20;
    size_t off_sx = (data_bytes + 255) & ~(size_t)255;
    size_t off_ws = (off_sx + scale_bytes + 255) & ~(size_t)255;
    char *scratch = (char *)cuda_tmp_alloc(off_ws + wz, "attn_out_a mx scratch");
    if (!scratch) return 0;
    __nv_fp8_e4m3 *xq = (__nv_fp8_e4m3 *)scratch;
    unsigned char *sx = (unsigned char *)(scratch + off_sx);
    void *ws = scratch + off_ws;
    cudaMemset(sx, 0, scale_bytes);
    int warps = (int)n_tokens * (int)n_groups * (int)KB;
    mxfp8_quant_act_grouped_kernel<<<(warps * 32 + 255) / 256, 256>>>(
            (const float *)heads->ptr, (int)n_tokens, (int)n_groups,
            (int)group_dim, KBp, xq, sx, x_scale_slab);
    if (!cuda_ok(cudaGetLastError(), "attn_out_a act quant")) return 0;
    cublasLtMatmulDesc_t op; if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return 0;
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t mo = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mo, sizeof(mo));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mo, sizeof(mo));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w->scale, sizeof(w->scale));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, group_dim, rank, group_dim);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, group_dim, n_tokens, group_dim);
    cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, rank, n_tokens, low_dim);
    cublasLtMatmulPreference_t pf; cublasLtMatmulPreferenceCreate(&pf);
    cublasLtMatmulPreferenceSetAttribute(pf, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wz, sizeof(wz));
    cublasLtMatmulHeuristicResult_t h; int got = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(g_cublaslt, op, la, lb, ld, ld, pf, 1, &h, &got);
    int ok = 0;
    if (hs == CUBLAS_STATUS_SUCCESS && got && ws) {
        ok = 1;
        for (uint32_t g = 0; g < n_groups && ok; g++) {
            const __nv_fp8_e4m3 *ag = w->data + (size_t)g * rank * group_dim;
            const unsigned char *as = w->scale + (size_t)g * w_scale_slab;
            const __nv_fp8_e4m3 *bg = xq + (size_t)g * n_tokens * group_dim;
            const unsigned char *bs = sx + (size_t)g * x_scale_slab;
            float *dg = (float *)low->ptr + (size_t)g * rank;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &as, sizeof(as));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bs, sizeof(bs));
            float al = 1.f, be = 0.f;
            cublasStatus_t st = cublasLtMatmul(g_cublaslt, op, &al, ag, la, bg, lb, &be,
                                               dg, ld, dg, ld, &h.algo, ws, wz, 0);
            ok = (st == CUBLAS_STATUS_SUCCESS);
            if (!ok) fprintf(stderr, "ds4: cuBLASLt attn_out_a MXFP8 matmul failed: status %d\n", (int)st);
        }
    }
    cublasLtMatmulPreferenceDestroy(pf);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(op);
    return ok;
}



/* Native FP8 decode (mmvq): read the MXFP8 weight 8-bit and dot with the f32
 * activation (single token). One warp per output row; lane L covers in-dim
 * positions {L, 32+L, ...}. No f16 expansion -> no decode bandwidth regression
 * vs the removed Q8_0 path. Used when MXFP8 tensor-cores don't apply (n_tok==1). */
__global__ static void mxfp8_mmvq_kernel(float *out, const unsigned char *w, const float *x,
                                         int in_dim, int out_dim) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    int KB = in_dim / 32;
    const unsigned char *row = w + (size_t)o * KB * 33;
    float acc = 0.f;
    for (int kb = 0; kb < KB; kb++) {
        const unsigned char *blk = row + (size_t)kb * 33;
        float scale = exp2f((float)blk[0] - 127.f);
        __half wh = (__half)(*(const __nv_fp8_e4m3 *)&blk[1 + lane]);
        acc += __half2float(wh) * scale * x[kb * 32 + lane];
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[o] = acc;
}


/* Decode mmvq over the DE-INTERLEAVED cached weight (contiguous E4M3 data[out,in]
 * + swizzled E8M0 scale), the same buffers the prefill tensor-core path builds via
 * cuda_fp8_mx_weight(). Reading contiguous fp8 lets each lane pull a uint32 (4 E4M3)
 * per step -> 128-wide coalesced loads vs the raw 33B-interleaved kernel's misaligned
 * 1-byte/thread reads. Numerically identical (same fp8 bytes, same raw E8M0 scale byte).
 * Requires in_dim % 128 == 0 (all MLA/shared/head dims qualify); else fall back to raw. */
__global__ static void mxfp8_mmvq_deint_kernel(float *out, const __nv_fp8_e4m3 *data,
                                               const unsigned char *scale, const float *x,
                                               int in_dim, int out_dim, int KBp) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;                       /* this lane's 4 in-positions */
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;                               /* 32-elem block for these 4 */
        float sc = __int_as_float((uint32_t)scale[mx_sfoff(o, kb, KBp)] << 23);  /* 2^(e-127), no SFU */
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) acc += __half2float((__half)q[j]) * sc * xk[j];
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[o] = acc;
}


/* Fused pair of the de-interleaved mmvq: two weights (out0,out1) sharing one
 * activation x and in_dim, computed in a single launch. Each warp owns one
 * global output row -- rows [0,out0_dim) go to weight0/out0, the rest to
 * weight1/out1. Bit-identical to launching mxfp8_mmvq_deint_kernel twice (same
 * per-row math); the win is one launch instead of two on the overhead-bound
 * decode path (q_a+kv from attn_norm, shared gate+up from ffn_norm). */
__global__ static void mxfp8_mmvq_deint_pair_kernel(
        float *out0, float *out1,
        const __nv_fp8_e4m3 *data0, const unsigned char *scale0, int out0_dim,
        const __nv_fp8_e4m3 *data1, const unsigned char *scale1, int out1_dim,
        const float *x, int in_dim, int KBp) {
    int warp = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    const __nv_fp8_e4m3 *data;
    const unsigned char *scale;
    float *out;
    int o;
    if (warp < out0_dim) { o = warp;            data = data0; scale = scale0; out = out0; }
    else                 { o = warp - out0_dim; if (o >= out1_dim) return;
                           data = data1; scale = scale1; out = out1; }
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;
        float sc = __int_as_float((uint32_t)scale[mx_sfoff(o, kb, KBp)] << 23);
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) acc += __half2float((__half)q[j]) * sc * xk[j];
    }
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
    if (lane == 0) out[o] = acc;
}


/* Small-batch (2..4 token) variant of the de-interleaved mmvq for the spec-decode
 * verify forward. One weight-row read serves all NT tokens (per-token accumulators),
 * vs the tensor-core tile path (latency-bound at 2-4 rows) or NT GEMV relaunches
 * (NT x weight traffic). Per-token multiply/accumulate order matches
 * mxfp8_mmvq_deint_kernel exactly, so each token's output is bit-identical to the
 * n=1 kernel run on that token alone. */
template <int NT>
__global__ static void mxfp8_mmvq_deint_nt_kernel(float *out, const __nv_fp8_e4m3 *data,
                                                  const unsigned char *scale, const float *x,
                                                  int in_dim, int out_dim, int KBp) {
    int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const __nv_fp8_e4m3 *row = data + (size_t)o * in_dim;
    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    for (int base = 0; base < in_dim; base += 128) {
        int k = base + lane * 4;
        uint32_t packed = *(const uint32_t *)(row + k);
        int kb = k >> 5;
        float sc = __int_as_float((uint32_t)scale[mx_sfoff(o, kb, KBp)] << 23);
        const __nv_fp8_e4m3 *q = (const __nv_fp8_e4m3 *)&packed;
        const float *xk = x + k;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float wj = __half2float((__half)q[j]) * sc;
            #pragma unroll
            for (int t = 0; t < NT; t++) acc[t] += wj * xk[(size_t)t * in_dim + j];
        }
    }
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        float a = acc[t];
        for (int s = 16; s > 0; s >>= 1) a += __shfl_xor_sync(0xffffffffu, a, s);
        if (lane == 0) out[(size_t)t * out_dim + o] = a;
    }
}


/* ============================================================================
 * PROTOTYPE: MXFP4 attention path (DS4_ATTN_MXFP4). Requant a workhorse MXFP8
 * weight (raw 33B blocks) -> MXFP4 (per 32-block: E8M0 byte + 16 e2m1 bytes),
 * cached by offset, and run a batch-1 mmvq off it. Halves the weight read (4-bit
 * vs 8-bit) for the big attention projections. Lossy requant -> measures the
 * quality/speed tradeoff before committing to a converter change.
 * ============================================================================ */
__device__ __constant__ float d_kE2M1_m[16] =
    {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f, 0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
__device__ __forceinline__ static uint8_t d_to_e2m1_m(float v){
    float best=1e30f; uint8_t bn=0;
    #pragma unroll
    for(uint8_t n=0;n<16;n++){ float d=fabsf(v-d_kE2M1_m[n]); if(d<best){best=d;bn=n;} }
    return bn;
}

/* raw MXFP8 [out,in] 33B blocks -> DE-INTERLEAVED MXFP4: nibbles packed contiguous in
 * data[out,in/2] + separate E8M0 scale[out,in/32]. Contiguous nibbles let the mmvq read
 * uint32 (8 nibbles) coalesced. one thread per (row,32-block). */
__global__ static void mxfp8_to_mxfp4_kernel(uint8_t *data, uint8_t *scale, const uint8_t *in8, int in_dim, int out_dim){
    const int nb = in_dim / 32;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)out_dim * nb) return;
    const int o = (int)(idx / nb), sb = (int)(idx % nb);
    const uint8_t *ib = in8 + ((size_t)o * nb + sb) * 33;
    const float ms = __int_as_float((uint32_t)ib[0] << 23);
    float v[32], mx = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) { v[i] = __half2float((__half)*(const __nv_fp8_e4m3 *)&ib[1 + i]) * ms; mx = fmaxf(mx, fabsf(v[i])); }
    int e = (mx > 0.f) ? (int)ceilf(log2f(mx / 6.f)) : 0; if (e < -30) e = -30; if (e > 30) e = 30;
    const float sc = exp2f((float)e);
    scale[(size_t)o * nb + sb] = (uint8_t)(e + 127);
    uint8_t *ob = data + ((size_t)o * in_dim + sb * 32) / 2;   /* 16 bytes = 32 nibbles */
    #pragma unroll
    for (int i = 0; i < 16; i++) { uint8_t lo = d_to_e2m1_m(v[2*i]/sc), hi = d_to_e2m1_m(v[2*i+1]/sc); ob[i] = (uint8_t)(lo | (hi << 4)); }
}

/* Arithmetic e2m1 decode (NO LUT). Returns 2x the actual value (fold 0.5 into scale). */
__device__ __forceinline__ static float e2m1_x2f(uint8_t nib){
    const unsigned e = (nib >> 1) & 3u, m = nib & 1u;
    const int mag = e ? (int)((1u << e) | (m << (e - 1u))) : (int)m;
    return (nib & 8u) ? -(float)mag : (float)mag;
}

/* batch-1 de-interleaved MXFP4 mmvq: warp-per-row (8 rows/block), each lane reads a uint32
 * (8 nibbles) per step -> coalesced. Mirrors the efficient MXFP8 mmvq. */
__global__ static void mxfp4_mmvq_kernel(float *out, const uint8_t *data, const uint8_t *scale,
                                         const float *x, int in_dim, int out_dim){
    const int o = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (o >= out_dim) return;
    const int nb = in_dim / 32;
    const uint8_t *drow = data + (size_t)o * (in_dim / 2);
    const uint8_t *srow = scale + (size_t)o * nb;
    float acc = 0.f;
    for (int base = 0; base < in_dim; base += 256) {          /* 32 lanes * 8 elems */
        const int k = base + lane * 8;
        const uint32_t packed = *(const uint32_t *)(drow + (k >> 1));   /* 8 nibbles */
        const float sc = __int_as_float((uint32_t)srow[k >> 5] << 23);  /* one 32-block */
        const float *xk = x + k;
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) s += e2m1_x2f((packed >> (4 * i)) & 0xF) * xk[i];
        acc += sc * s;
    }
    acc *= 0.5f;
    for (int st = 16; st > 0; st >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, st);
    if (lane == 0) out[o] = acc;
}

struct mxfp4_weight { const void *host_base; uint64_t offset; uint64_t in_dim, out_dim; uint8_t *data; uint8_t *scale; };
static std::unordered_map<uint64_t, mxfp4_weight> g_mxfp4_by_offset;

/* Requant a raw MXFP8 weight [out,in] to de-interleaved MXFP4, cached by offset. */
static const mxfp4_weight *cuda_mxfp4_from_mxfp8(const void *model_map, uint64_t offset, uint64_t mxfp8_bytes,
                                                 uint64_t in_dim, uint64_t out_dim){
    auto it = g_mxfp4_by_offset.find(offset);
    if (it != g_mxfp4_by_offset.end() && it->second.host_base == model_map &&
        it->second.in_dim == in_dim && it->second.out_dim == out_dim) return &it->second;
    const unsigned char *src = (const unsigned char *)cuda_model_range_ptr(model_map, offset, mxfp8_bytes, "mxfp4_req");
    if (!src) return NULL;
    const uint64_t nb = in_dim / 32;
    uint8_t *data = NULL, *scale = NULL;
    if (cudaMalloc(&data, (size_t)out_dim * in_dim / 2) != cudaSuccess) return NULL;
    if (cudaMalloc(&scale, (size_t)out_dim * nb) != cudaSuccess) { cudaFree(data); return NULL; }
    const long total = (long)out_dim * nb;
    mxfp8_to_mxfp4_kernel<<<(unsigned)((total + 255) / 256), 256>>>(data, scale, src, (int)in_dim, (int)out_dim);
    if (!cuda_ok(cudaGetLastError(), "mxfp8->mxfp4 requant")) { cudaFree(data); cudaFree(scale); return NULL; }
    mxfp4_weight w = { model_map, offset, in_dim, out_dim, data, scale };
    g_mxfp4_by_offset[offset] = w;
    return &g_mxfp4_by_offset[offset];
}


/* PER-TENSOR routing: every offset registered at load (the MXFP8 workhorse
 * weights: attn_kv/q_a/q_b, attn_output_a/b, shared experts, output head)
 * takes the FP8 path; anything unregistered is rejected below. */
std::unordered_set<uint64_t> g_fp8_offsets;


extern "C" void ds4_gpu_register_fp8_weight(uint64_t weight_offset) { g_fp8_offsets.insert(weight_offset); }



static int cuda_matmul_mxfp8_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    if (g_fp8_offsets.count(weight_offset)) {
        const uint64_t fblocks = (in_dim + 31) / 32;
        const uint64_t fbytes = out_dim * fblocks * 33;
        if (weight_offset > model_size || fbytes > model_size - weight_offset ||
            x->bytes < n_tok * in_dim * sizeof(float) ||
            out->bytes < n_tok * out_dim * sizeof(float)) return 0;
        /* Small batches (spec-decode verify, n_tok 2..4): batched GEMV over the
         * de-interleaved weight. One weight-row read serves all tokens, vs the
         * tensor-core tile path which is latency-bound at these shapes (the
         * measured "CUTLASS launch storm" that made verify(3) cost ~2x a decode
         * token). Bit-identical per token to the n=1 deint mmvq, so verify logits
         * match the decode path's numerics. DS4_FP8_GEMV_MAX_N=1 restores the
         * tensor-core dispatch for all n_tok>1. */
        static const int gemv_max_n = []{ const char *e = getenv("DS4_FP8_GEMV_MAX_N");
                return e ? atoi(e) : 4; }();
        static const int nt_fp8_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
        static const int nt_mxfp4 = getenv("DS4_ATTN_MXFP4") != NULL;
        if (n_tok >= 2 && n_tok <= 4 && n_tok <= (uint64_t)gemv_max_n &&
            in_dim % 128 == 0 && !nt_fp8_raw && !nt_mxfp4) {
            const fp8_mx_weight *bw = cuda_fp8_mx_weight(model_map, weight_offset, fbytes,
                                                         in_dim, out_dim, label);
            if (bw) {
                const int KBp = mx_rup((int)(in_dim / 32), 4);
                const unsigned wpb = 8;
                dim3 grid(((unsigned)out_dim + wpb - 1) / wpb);
                switch (n_tok) {
                case 2: mxfp8_mmvq_deint_nt_kernel<2><<<grid, wpb * 32>>>((float *)out->ptr,
                        bw->data, bw->scale, (const float *)x->ptr, (int)in_dim, (int)out_dim, KBp); break;
                case 3: mxfp8_mmvq_deint_nt_kernel<3><<<grid, wpb * 32>>>((float *)out->ptr,
                        bw->data, bw->scale, (const float *)x->ptr, (int)in_dim, (int)out_dim, KBp); break;
                default: mxfp8_mmvq_deint_nt_kernel<4><<<grid, wpb * 32>>>((float *)out->ptr,
                        bw->data, bw->scale, (const float *)x->ptr, (int)in_dim, (int)out_dim, KBp); break;
                }
                return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint nt");
            }
        }
        /* Prefill (n_tok>1) uses the cuBLASLt MX tensor-core GEMM; decode falls
         * through to the per-token mmvq kernel below. DS4_FP8_NO_MXCORE forces
         * the mmvq path everywhere as an operational fallback. */
        if (n_tok > 1 && getenv("DS4_FP8_NO_MXCORE") == NULL &&
                cuda_matmul_fp8_mx_tensor_labeled(out, model_map, model_size,
                weight_offset, in_dim, out_dim, x, n_tok, label)) return 1;
        /* PROTOTYPE: decode MXFP4 (4-bit) instead of MXFP8 (8-bit) for the workhorse
         * projections -> halves the weight read. Requant-from-MXFP8 cached per offset. */
        static const int attn_mxfp4 = getenv("DS4_ATTN_MXFP4") != NULL;
        /* DS4_ATTN_MXFP4=big -> only the two big output-side projections (q_b in=1024,
         * output_b in=8192), sparing the sensitive latent q_a/kv + shared expert. */
        static const int attn_mxfp4_big = []{ const char *e = getenv("DS4_ATTN_MXFP4"); return e && !strcmp(e, "big"); }();
        const bool mxfp4_pick = attn_mxfp4 && (!attn_mxfp4_big || in_dim == 1024 || in_dim == 8192);
        if (mxfp4_pick && in_dim % 256 == 0) {
            const mxfp4_weight *m4 = cuda_mxfp4_from_mxfp8(model_map, weight_offset, fbytes, in_dim, out_dim);
            if (m4) {
                /* per-token loop (prefill n_tok>1 uses this too, so frontier logits reflect
                 * MXFP4 attention -> a valid quality signal; slow but only a correctness path). */
                for (uint64_t t = 0; t < n_tok; t++)
                    mxfp4_mmvq_kernel<<<((unsigned)out_dim + 7) / 8, 256>>>((float *)out->ptr + t * out_dim,
                            m4->data, m4->scale, (const float *)x->ptr + t * in_dim, (int)in_dim, (int)out_dim);
                return cuda_ok(cudaGetLastError(), "mxfp4 attn mmvq");
            }
        }
        const unsigned wpb = 8;  /* output rows per block */
        dim3 grid(((unsigned)out_dim + wpb - 1) / wpb);
        /* Prefer the de-interleaved cached weight (contiguous E4M3 -> coalesced 128-wide
         * loads, vs the raw 33B-interleaved kernel's misaligned 1-byte/thread reads).
         * Bit-exact vs the raw kernel (verified: rel 0 across all workhorse shapes);
         * ~+8% decode. DS4_FP8_MMVQ_RAW forces the raw path as an operational fallback. */
        static const int fp8_mmvq_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
        const fp8_mx_weight *w = (in_dim % 128 == 0 && !fp8_mmvq_raw)
                ? cuda_fp8_mx_weight(model_map, weight_offset, fbytes, in_dim, out_dim, label)
                : NULL;
        if (w) {
            const int KBp = mx_rup((int)(in_dim / 32), 4);
            for (uint64_t t = 0; t < n_tok; t++)
                mxfp8_mmvq_deint_kernel<<<grid, wpb * 32>>>((float *)out->ptr + t * out_dim,
                        w->data, w->scale, (const float *)x->ptr + t * in_dim,
                        (int)in_dim, (int)out_dim, KBp);
            return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint");
        }
        const unsigned char *wfp8 = (const unsigned char *)cuda_model_range_ptr(
                model_map, weight_offset, fbytes, "fp8_mx_mmvq");
        if (!wfp8) return 0;
        for (uint64_t t = 0; t < n_tok; t++)
            mxfp8_mmvq_kernel<<<grid, wpb * 32>>>((float *)out->ptr + t * out_dim, wfp8,
                                                  (const float *)x->ptr + t * in_dim,
                                                  (int)in_dim, (int)out_dim);
        return cuda_ok(cudaGetLastError(), "fp8_mx mmvq");
    }
    fprintf(stderr,
            "ds4: matmul %s at offset %llu is not a registered MXFP8 weight "
            "(legacy q8_0 weights are no longer supported)\n",
            label ? label : "mxfp8",
            (unsigned long long)weight_offset);
    return 0;
}



extern "C" int ds4_gpu_matmul_mxfp8_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_mxfp8_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "mxfp8");
}



extern "C" int ds4_gpu_matmul_mxfp8_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    /* Decode fast path: both weights share x and in_dim, so fuse the two
     * de-interleaved mmvq launches into one. Only when both are registered
     * de-interleavable MXFP8 weights and neither the MXFP4 prototype nor the raw
     * mmvq fallback is forced; otherwise defer to the per-weight path below. */
    static const int fp8_mmvq_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
    static const int attn_mxfp4 = getenv("DS4_ATTN_MXFP4") != NULL;
    static const int no_pair_fuse = getenv("DS4_CUDA_NO_MXFP8_PAIR_FUSE") != NULL;
    if (n_tok == 1 && in_dim % 128 == 0 && !fp8_mmvq_raw && !attn_mxfp4 && !no_pair_fuse &&
        g_fp8_offsets.count(weight0_offset) && g_fp8_offsets.count(weight1_offset)) {
        const uint64_t fblocks = (in_dim + 31) / 32;
        const uint64_t fbytes0 = out0_dim * fblocks * 33;
        const uint64_t fbytes1 = out1_dim * fblocks * 33;
        const bool bounds_ok =
            weight0_offset <= model_size && fbytes0 <= model_size - weight0_offset &&
            weight1_offset <= model_size && fbytes1 <= model_size - weight1_offset &&
            x->bytes >= in_dim * sizeof(float) &&
            out0->bytes >= out0_dim * sizeof(float) &&
            out1->bytes >= out1_dim * sizeof(float);
        if (bounds_ok) {
            const fp8_mx_weight *w0 = cuda_fp8_mx_weight(model_map, weight0_offset, fbytes0,
                                                         in_dim, out0_dim, "mxfp8_pair0");
            const fp8_mx_weight *w1 = cuda_fp8_mx_weight(model_map, weight1_offset, fbytes1,
                                                         in_dim, out1_dim, "mxfp8_pair1");
            if (w0 && w1) {
                const int KBp = mx_rup((int)(in_dim / 32), 4);
                const unsigned wpb = 8;  /* output rows (warps) per 256-thread block */
                dim3 grid(((unsigned)(out0_dim + out1_dim) + wpb - 1) / wpb);
                mxfp8_mmvq_deint_pair_kernel<<<grid, wpb * 32>>>(
                        (float *)out0->ptr, (float *)out1->ptr,
                        w0->data, w0->scale, (int)out0_dim,
                        w1->data, w1->scale, (int)out1_dim,
                        (const float *)x->ptr, (int)in_dim, KBp);
                return cuda_ok(cudaGetLastError(), "fp8_mx mmvq deint pair");
            }
        }
    }
    return cuda_matmul_mxfp8_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                           in_dim, out0_dim, x, n_tok, "mxfp8_pair0") &&
           cuda_matmul_mxfp8_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                           in_dim, out1_dim, x, n_tok, "mxfp8_pair1");
}



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
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    if (!g_fp8_offsets.count(weight_offset)) return 0;
    const uint64_t wstride = 33u;
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * wstride)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * wstride;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "fp8_hc_expand");
    if (!wptr) return 0;

    /* Decode uses the de-interleaved cached weight (coalesced vectorized loads); raw
     * 33B path retained as DS4_FP8_MMVQ_RAW fallback. Same weight buffers as mmvq. */
    static const int fp8_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
    const fp8_mx_weight *dw = (in_dim % 32 == 0 && !fp8_raw)
            ? cuda_fp8_mx_weight(model_map, weight_offset, weight_bytes, in_dim, out_dim,
                                 label ? label : "fp8_hc_expand")
            : NULL;
    const int KBp = mx_rup((int)(in_dim / 32), 4);
    const dim3 hg(((unsigned)out_dim + 31u) / 32u);
    const float *ba = block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr;
    if (dw) {
        matmul_fp8mx_hc_expand_warp8_kernel<true><<<hg, 256>>>(
                (float *)out_hc->ptr, (float *)block_out->ptr, ba,
                (const float *)residual_hc->ptr, (const float *)split->ptr,
                (const unsigned char *)wptr, dw->data, dw->scale, KBp, (const float *)x->ptr,
                in_dim, out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
    } else {
        matmul_fp8mx_hc_expand_warp8_kernel<false><<<hg, 256>>>(
                (float *)out_hc->ptr, (float *)block_out->ptr, ba,
                (const float *)residual_hc->ptr, (const float *)split->ptr,
                (const unsigned char *)wptr, (const __nv_fp8_e4m3 *)NULL, (const unsigned char *)NULL, 0,
                (const float *)x->ptr, in_dim, out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
    }
    return cuda_ok(cudaGetLastError(), "matmul_fp8_hc_expand launch");
}



extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    static const int serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    static const int env_serial_router = getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    static const int env_ordered_f16 = getenv("DS4_CUDA_ORDERED_F16_MATMUL") != NULL;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        env_serial_router;
    /* Decode f16 matmuls default to the coalesced 256-thread matmul_f16_kernel (the
     * fall-through below): ~+6% decode vs the uncoalesced 32-thread ordered_chunks
     * kernel. Both are deterministic; the reduction order differs by ~1 ULP, which is
     * far below this model's run-to-run nondeterminism (float atomics elsewhere). The
     * exact old order is opt-in via DS4_CUDA_ORDERED_F16_MATMUL. */
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        n_tok == 1u &&
        env_ordered_f16;
    /* Small batches (spec-decode verify, n_tok 2..4): batched f16 GEMV. One
     * weight-row read serves all tokens and there is no f32->f16 activation
     * convert or tmp alloc; per-token output is bit-identical to the n=1
     * matmul_f16_kernel. DS4_F16_GEMV_MAX_N=1 restores the cuBLAS dispatch. */
    static const int f16_gemv_max_n = []{ const char *e = getenv("DS4_F16_GEMV_MAX_N");
            return e ? atoi(e) : 4; }();
    if (!serial_f16 && n_tok >= 2 && n_tok <= 4 && n_tok <= (uint64_t)f16_gemv_max_n) {
        dim3 g((unsigned)out_dim);
        switch (n_tok) {
        case 2: matmul_f16_nt_kernel<2><<<g, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim); break;
        case 3: matmul_f16_nt_kernel<3><<<g, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim); break;
        default: matmul_f16_nt_kernel<4><<<g, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim); break;
        }
        return cuda_ok(cudaGetLastError(), "matmul_f16 nt launch");
    }
    if (!serial_f16 && g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_router_serial launch" : "matmul_f16_serial launch");
    }
    if (ordered_router) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}



extern "C" int ds4_gpu_matmul_bf16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "bf16");
    if (!wptr) return 0;
    const uint16_t *w = (const uint16_t *)wptr;
    if (g_cublas_ready && n_tok > 1 && getenv("DS4_CUDA_SERIAL_BF16_MATMUL") == NULL) {
        const uint64_t xb_count = n_tok * in_dim;
        uint16_t *xb = (uint16_t *)cuda_tmp_alloc(xb_count * sizeof(uint16_t), "bf16 gemm activations");
        if (!xb) return 0;
        f32_to_bf16_kernel<<<(xb_count + 255) / 256, 256>>>(xb, (const float *)x->ptr, xb_count);
        if (!cuda_ok(cudaGetLastError(), "bf16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         (int)out_dim, (int)n_tok, (int)in_dim,
                                         &alpha,
                                         w, CUDA_R_16BF, (int)in_dim,
                                         xb, CUDA_R_16BF, (int)in_dim,
                                         &beta,
                                         out->ptr, CUDA_R_32F, (int)out_dim,
                                         CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "bf16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_bf16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_bf16 launch");
}



extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    /* Default: two separate coalesced matmuls (each fast via matmul_f16_kernel). The
     * fused pair_ordered_chunks kernel is opt-in with the exact old order. */
    static const int env_no_pair = getenv("DS4_CUDA_NO_F16_PAIR_MATMUL") != NULL;
    static const int env_serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    static const int env_serial_router = getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    static const int env_ordered_f16 = getenv("DS4_CUDA_ORDERED_F16_MATMUL") != NULL;
    if (n_tok != 1 ||
        env_no_pair ||
        env_serial_f16 ||
        env_serial_router ||
        !env_ordered_f16) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}



extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}



extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}


/* Decode grouped "a" projection: prefer the de-interleaved cached weight (vectorized
 * coalesced loads) over the raw 33B kernel. Bit-exact; DS4_FP8_MMVQ_RAW forces raw. */
static int launch_grouped_fp8mx_a(float *low, const void *model_map, uint64_t out_a_offset,
        uint64_t out_a_bytes, const unsigned char *out_a, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, uint32_t n_tokens, uint64_t blocks_a, uint64_t low_dim,
        const float *heads, const char *label) {
    const dim3 grid_a(((unsigned)low_dim + 31u) / 32u, (unsigned)n_tokens, 1);
    static const int fp8_raw = getenv("DS4_FP8_MMVQ_RAW") != NULL;
    const fp8_mx_weight *dw = (group_dim % 32 == 0 && !fp8_raw)
            ? cuda_fp8_mx_weight(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, label) : NULL;
    const int KBp = mx_rup((int)(group_dim / 32), 4);
    /* Small verify batches: one nt launch, weight blocks L1-shared across tokens;
     * per-token bit-identical to the n=1 DEINT kernel below. */
    if (dw && n_tokens >= 2u && n_tokens <= 4u &&
        rank % DS4_FP8MX_ROWS == 0 && low_dim % DS4_FP8MX_ROWS == 0) {
        const dim3 g(((unsigned)low_dim + 31u) / 32u);
        switch (n_tokens) {
        case 2: grouped_fp8mx_a_nt_kernel<2><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        case 3: grouped_fp8mx_a_nt_kernel<3><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        default: grouped_fp8mx_a_nt_kernel<4><<<g, 256>>>(low, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, blocks_a); break;
        }
        return cuda_ok(cudaGetLastError(), "attention_output_a nt launch");
    }
    if (dw) {
        grouped_fp8mx_a_warp8_kernel<true><<<grid_a, 256>>>(low, out_a, dw->data, dw->scale, KBp,
                heads, group_dim, rank, n_groups, n_tokens, blocks_a);
    } else {
        grouped_fp8mx_a_warp8_kernel<false><<<grid_a, 256>>>(low, out_a,
                (const __nv_fp8_e4m3 *)NULL, (const unsigned char *)NULL, 0,
                heads, group_dim, rank, n_groups, n_tokens, blocks_a);
    }
    return cuda_ok(cudaGetLastError(), "attention_output_a launch");
}


extern "C" int ds4_gpu_attention_output_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    if (!g_fp8_offsets.count(out_a_offset) || !g_fp8_offsets.count(out_b_offset)) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 33u;
    const uint64_t out_b_bytes = out_dim * blocks_b * 33u;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }

    /* "a" projection: prefill takes the block-scaled MXFP8xMXFP8 tensor-core
     * GEMMs; decode and small verify batches (n_tokens<=4) take the
     * register-blocked GEMV path (launch_grouped_fp8mx_a dispatches the nt
     * variant at 2..4 -- one launch vs 8 per-group GEMMs, bit-identical per
     * token to decode's kernel). DS4_FP8_GEMV_MAX_N=1 restores the tensor-core
     * dispatch for all n_tokens>1, same as the dense-matmul gate. */
    static const int a_gemv_max_n = []{ const char *e = getenv("DS4_FP8_GEMV_MAX_N");
            return e ? atoi(e) : 4; }();
    int a_done = 0;
    if (n_tokens > 1 && (int)n_tokens > a_gemv_max_n && getenv("DS4_FP8_NO_MXCORE") == NULL) {
        a_done = cuda_attention_output_a_mx_gemm(low, model_map, model_size, out_a_offset,
                                                 group_dim, rank, n_groups, heads, n_tokens);
    }
    if (!a_done) {
        const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
                cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
        if (!out_a) return 0;
        if (!launch_grouped_fp8mx_a((float *)low->ptr, model_map, out_a_offset, out_a_bytes, out_a,
                                    group_dim, rank, n_groups, n_tokens, blocks_a, low_dim,
                                    (const float *)heads->ptr, "attn_out_a")) return 0;
    }

    return cuda_matmul_mxfp8_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}



extern "C" int ds4_gpu_attention_output_low_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    if (!g_fp8_offsets.count(out_a_offset)) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 33u;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    return launch_grouped_fp8mx_a((float *)low->ptr, model_map, out_a_offset, out_a_bytes, out_a,
                                  group_dim, rank, n_groups, 1, blocks_a, low_dim,
                                  (const float *)heads->ptr, "attn_out_a");
}

