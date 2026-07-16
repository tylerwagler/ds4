#include "ds4_cuda_internal.h"

#include "ds4_iq2_tables_cuda.inc"



__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}



__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}



__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}



__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}



__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}



__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}



__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}



__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}



__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}



__device__ __forceinline__ static int dev_e2m1_x2(unsigned nib) {
    const unsigned e = (nib >> 1) & 3u;
    const unsigned m = nib & 1u;
    const int mag = e ? (int)((1u << e) | (m << (e - 1u))) : (int)m;
    return (nib & 8u) ? -mag : mag;
}



__device__ static float dev_dot_mxfp4_q8_K_block(const unsigned char *x, const cuda_block_q8_K *y) {
    const int8_t *q8 = y->qs;
    float acc = 0.0f;
    #pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        const unsigned char *blk = x + (size_t)sb * 17;
        /* E8M0 -> 2^(b-127): a bare float exponent field. Bitcast (b<<23) beats
         * exp2f (transcendental) — valid for b in [1,254]; real scales never hit
         * 0 (2^-127, ~0) or 255 (E8M0 NaN). */
        const float scale = __int_as_float((uint32_t)blk[0] << 23);
        int32_t sumi = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++) {                         /* 8 groups of 4 -> dp4a */
            const unsigned char b0 = blk[1 + g * 2];
            const unsigned char b1 = blk[1 + g * 2 + 1];
            const int32_t wpack =
                  ((uint32_t)(uint8_t)dev_e2m1_x2(b0 & 0xF))
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b0 >> 4)  << 8)
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b1 & 0xF) << 16)
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b1 >> 4)  << 24);
            sumi = __dp4a(wpack, *(const int32_t *)(q8 + sb * 32 + g * 4), sumi);
        }
        acc += scale * (float)sumi;
    }
    return 0.5f * y->d * acc;
}



/* MXFP4 (type-39) weight block vs up to 8 q8_K activation blocks, decoding each
 * weight nibble-group ONCE (into wpack) and dp4a-ing it against all 8 tokens --
 * the cross-token weight reuse the per-pair qwarp32 kernel lacks. Accumulates
 * into acc[8] with the SAME per-subblock (scale*sumi) then 0.5*y->d order as
 * dev_dot_mxfp4_q8_K_block, so a tile of 8 is bit-identical to 8 single dots. */
__device__ static void dev_dot_mxfp4_q8_K_block8(
        const unsigned char *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    float facc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        const unsigned char *blk = x + (size_t)sb * 17;
        const float scale = __int_as_float((uint32_t)blk[0] << 23);
        int32_t wpack[8];
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const unsigned char b0 = blk[1 + g * 2];
            const unsigned char b1 = blk[1 + g * 2 + 1];
            wpack[g] =
                  ((uint32_t)(uint8_t)dev_e2m1_x2(b0 & 0xF))
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b0 >> 4)  << 8)
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b1 & 0xF) << 16)
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b1 >> 4)  << 24);
        }
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q8 = ys[p]->qs + sb * 32;
            int32_t sumi = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++)
                sumi = __dp4a(wpack[g], *(const int32_t *)(q8 + g * 4), sumi);
            facc[p] += scale * (float)sumi;
        }
    }
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.5f * ys[p]->d * facc[p];
}



__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}



__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}






__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}



__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        /* v already holds xr[tid] (loaded above; the reduction only touches the
         * shared arrays), so reuse it instead of a second global load. */
        int qv = (int)lrintf(iscale_s * v);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}












__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}



/* MXFP4 routed gate/up (type-39 experts). Generic correctness-first variant: one
 * (token,expert) pair per blockIdx.y, 128 rows per block via the rr loop. Row stride
 * inside an expert weight is gate_row_bytes; each Q8_K input block maps to 8*17 bytes. */
__global__ static void moe_gate_up_mid_mxfp4_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const unsigned char *gr = (const unsigned char *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const unsigned char *ur = (const unsigned char *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_mxfp4_q8_K_block(gr + (uint64_t)b * 8u * 17u, xqb + b);
            up += dev_dot_mxfp4_q8_K_block(ur + (uint64_t)b * 8u * 17u, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}



__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}



__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}



__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t expert_count) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[expert_count] = sum;
    }
}



__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}



__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[expert_count] = sum;
        *tile_total = sum;
    }
}



__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= expert_count) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}



__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}


/* Dynamic-SMEM bytes the MXFP4 N-tile gate/up kernel stages for NT tokens of
 * xq_blocks q8_K activation blocks each -- or 0, meaning "read the activations
 * straight from global".  HOST AND DEVICE CALL THIS SAME FUNCTION: the kernel
 * stages iff it returns non-zero, and the launch's dynamic-SMEM argument is
 * exactly what it returns.  One predicate, two callers -- two copies could
 * drift apart into a kernel that indexes SMEM the host never allocated. */
__host__ __device__ __forceinline__ static uint32_t moe_mxfp4_gate_up_smem_bytes(
        uint32_t nt, uint32_t xq_blocks) {
    return (xq_blocks <= 16u) ? (uint32_t)(nt * xq_blocks * (uint32_t)sizeof(cuda_block_q8_K)) : 0u;
}

/* The most the NT=16 kernel can ever stage, given the xq_blocks <= 16 bound
 * above: 16 tokens * 16 blocks * 292 B = 74,752 B.  That is over the 48 KB
 * static __shared__ cap -- which is the whole reason this tile was pinned to 8
 * -- so NT=16 is reachable only through the one-time dynamic-SMEM opt-in in
 * moe_mxfp4_gate_up_tile_width(). */
#define MOE_MXFP4_GATE_UP_NT16_SMEM_MAX (16u * 16u * (uint32_t)sizeof(cuda_block_q8_K))

/* MXFP4 (type-39) weight block vs up to NT q8_K activation blocks: the N-tile
 * generalisation of dev_dot_mxfp4_q8_K_block8, decoding each weight
 * nibble-group ONCE (into wpack) and dp4a-ing it against all NT tokens.
 *
 * BIT-EXACTNESS: the reduction is over K, and NT tiles TOKENS.  The per-subblock
 * (scale*sumi) order and the trailing 0.5*y->d fold are identical at every NT,
 * and tokens are independent output columns, so a tile of NT is bit-identical to
 * NT single dev_dot_mxfp4_q8_K_block calls -- for any NT.  Widening N moves zero
 * bits; `make cuda-prefill-gate` is the referee.
 *
 * The loops over the token arrays are unrolled over the COMPILE-TIME bound NT
 * with a uniform `break`, never over runtime `n`.  A runtime bound makes ptxas
 * spill ys[]/facc[] to LOCAL memory, which trades the tile win for local-memory
 * traffic.  np is block-uniform, so the predicate is a uniform branch, not
 * divergence.  (dev_dot_mxfp4_q8_K_block8 above keeps its runtime loop and its
 * 8 scalar parameters: it is the down kernel's dot, which this increment does
 * not touch -- down is already past its tile knee, measured 1.00x at NT=32.
 * Any change to the decode arithmetic here must be mirrored there.) */
template <uint32_t NT>
__device__ static void dev_dot_mxfp4_q8_K_blockN(
        const unsigned char *x,
        const cuda_block_q8_K *const *ys,
        uint32_t n,
        float acc[NT]) {
    float facc[NT];
    #pragma unroll
    for (uint32_t p = 0; p < NT; p++) facc[p] = 0.0f;
    #pragma unroll
    for (int sb = 0; sb < 8; sb++) {
        const unsigned char *blk = x + (size_t)sb * 17;
        const float scale = __int_as_float((uint32_t)blk[0] << 23);
        int32_t wpack[8];
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const unsigned char b0 = blk[1 + g * 2];
            const unsigned char b1 = blk[1 + g * 2 + 1];
            wpack[g] =
                  ((uint32_t)(uint8_t)dev_e2m1_x2(b0 & 0xF))
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b0 >> 4)  << 8)
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b1 & 0xF) << 16)
                | ((uint32_t)(uint8_t)dev_e2m1_x2(b1 >> 4)  << 24);
        }
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) {
            if (p >= n) break;
            const int8_t *q8 = ys[p]->qs + sb * 32;
            int32_t sumi = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++)
                sumi = __dp4a(wpack[g], *(const int32_t *)(q8 + g * 4), sumi);
            facc[p] += scale * (float)sumi;
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < NT; p++) { if (p >= n) break; acc[p] += 0.5f * ys[p]->d * facc[p]; }
}


/* MXFP4 (type-39) gate/up big-batch expert-tiled kernel: mirror of
 * moe_gate_up_mid_expert_tile8_rowspan_kernel but with the MXFP4 dot (no
 * iq2 grid/signs tables needed). One NT-token tile reuses each weight row's
 * decode across all NT tokens -- the prefill weight-reuse the per-pair
 * qwarp32 kernel lacks. Writes only mid_out (swiglu fused), same as the
 * iq2/q2k tiled kernels. Bit-identical to the qwarp32 path per token.
 *
 * NT was 8 until this increment, inherited from the iq2 gate (7c92951 "mirrors
 * the iq2 gate") despite MXFP4 carrying 2.06x the weight bytes -- and 8 was
 * never a register or occupancy choice, it was the largest tile whose staging
 * fit the 48 KB STATIC __shared__ cap (8*16*292 = 37,376 B; 16 tokens needs
 * 74,752 and ptxas refuses it outright).  Staging in DYNAMIC SMEM lifts that
 * ceiling, and NT=16 measures 2.30x on the GB10, bit-exact.
 *
 * Two counter-intuitive results, recorded so they are not "optimised" away:
 *   - NT=16 runs at 1 block/SM (16.7% occupancy), HALF the NT=8 kernel's, and is
 *     still 2.30x faster.  Occupancy is not the constraint; decode amortisation
 *     (per-weight, flat in NT) and the halved weight re-reads are.
 *   - Dropping the staging at NT=16 looks better on every traffic metric (4.2 GB
 *     of L2 vs 14.6) and is 2x SLOWER.  The kernel is issue/latency-bound, not
 *     bandwidth-bound.  STAGE AT 16.
 * NT=32 cannot stage at all (149,504 B > the 101,376 B opt-in cap) and unstaged
 * hits 255 registers + spill (1.43x, worse than NT=16 staged); NT=64 spills
 * 1536 B and is SLOWER than shipped.  16 is the peak, not a waypoint. */
template <uint32_t ROW_SPAN, uint32_t NT>
__global__ static void moe_gate_up_mid_mxfp4_expert_ntile_rowspan_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    /* [NT][xq_blocks], packed to the real xq_blocks stride (the static array
     * this replaces was [8][16] and wasted the tail when xq_blocks < 16).
     * Sized by the host from moe_mxfp4_gate_up_smem_bytes(); untouched when
     * that returns 0. */
    extern __shared__ cuda_block_q8_K sxq_dyn[];
    uint32_t pair[NT];
    uint32_t tok[NT];
    uint32_t slot[NT];
    const cuda_block_q8_K *xqb[NT];
    #pragma unroll
    for (uint32_t i = 0; i < NT; i++) { pair[i] = 0; tok[i] = 0; slot[i] = 0; xqb[i] = NULL; }
    const uint32_t cnt = counts[expert];
    const uint32_t avail = (local_start < cnt) ? (cnt - local_start) : 0u;
    const uint32_t np = avail < NT ? avail : NT;
    #pragma unroll
    for (uint32_t i = 0; i < NT; i++) {
        if (i >= np) break;
        pair[i] = sorted_pairs[offsets[expert] + local_start + i];
        tok[i] = pair[i] / n_expert;
        slot[i] = pair[i] - tok[i] * n_expert;
        xqb[i] = xq + (uint64_t)tok[i] * xq_blocks;
    }
    if (moe_mxfp4_gate_up_smem_bytes(NT, xq_blocks)) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq_dyn[p * xq_blocks + b] = xqb[p][b];
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) { if (p >= np) break; xqb[p] = sxq_dyn + p * xq_blocks; }
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const unsigned char *gr = (const unsigned char *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const unsigned char *ur = (const unsigned char *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[NT], up[NT];
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) { gate[p] = 0.0f; up[p] = 0.0f; }
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            const cuda_block_q8_K *yb[NT];
            #pragma unroll
            for (uint32_t p = 0; p < NT; p++) { yb[p] = (p < np) ? xqb[p] + b : NULL; }
            dev_dot_mxfp4_q8_K_blockN<NT>(gr + (uint64_t)b * 8u * 17u, yb, np, gate);
            dev_dot_mxfp4_q8_K_blockN<NT>(ur + (uint64_t)b * 8u * 17u, yb, np, up);
        }
        #pragma unroll
        for (uint32_t p = 0; p < NT; p++) {
            if (p >= np) break;
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}


/* The MXFP4 gate/up token-tile width, resolved ONCE from a DEVICE QUERY plus
 * the opt-in call itself -- deliberately NOT an env switch (project rule: no
 * getenv/flag branching on per-token/per-layer paths).  NT=16 stages 74,752 B,
 * over the 48 KB default cap; a device whose opt-in cap cannot cover that, or a
 * driver that refuses the opt-in, gets NT=8 -- which needs no opt-in (37,376 B)
 * and is the shipped behaviour -- rather than a launch that fails at 16.
 *
 * The cudaGetLastError() bracketing is not superstition.  A failed
 * cudaFuncSetAttribute leaves a STICKY last-error that the next unrelated
 * cudaGetLastError() reads as its own; in the microbench that exact mechanism
 * reported NT=32/64 as "launch failures" they never were and inverted the
 * finding.  So: clear before, trust the call's OWN return value, clear after --
 * never infer this call's outcome from a later cudaGetLastError().
 *
 * ROW_SPAN is a template parameter because the attribute is per-INSTANTIATION:
 * it must be set on exactly the specialisation that gets launched. */
/* Does this device grant the NT=16 tile?  Split out from the caching wrapper so
 * that the cache is published EXACTLY ONCE, with its final value, rather than
 * staging an 8 up front as an "unresolved" marker that a concurrent caller could
 * observe as a resolved 8 and take the narrow tile on.  (That would only ever
 * have been a lost speedup, never a wrong result -- each caller re-derives its
 * nt, its smem size and its tile list together, so any launch is self-consistent
 * and bit-exact -- and engine access is single-threaded today.  But the
 * multi-session work would make it reachable, and a once-resolver that can hand
 * out a provisional answer is not worth keeping.) */
template <uint32_t ROW_SPAN>
static uint32_t moe_mxfp4_gate_up_resolve_tile_width(void) {
    int dev = 0;
    int optin_cap = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin_cap, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: MoE MXFP4 gate/up: device SMEM query failed; using the NT=8 tile\n");
        return 8u;
    }
    if ((uint32_t)optin_cap < MOE_MXFP4_GATE_UP_NT16_SMEM_MAX) {
        fprintf(stderr,
                "ds4: MoE MXFP4 gate/up: device dynamic-SMEM opt-in cap %d B < %u B needed for the "
                "NT=16 tile; using the NT=8 tile (~2.3x slower on this stage)\n",
                optin_cap, MOE_MXFP4_GATE_UP_NT16_SMEM_MAX);
        return 8u;
    }
    (void)cudaGetLastError();       /* clear: a stale error must not be misread as ours */
    const cudaError_t at = cudaFuncSetAttribute(
        (const void *)moe_gate_up_mid_mxfp4_expert_ntile_rowspan_kernel<ROW_SPAN, 16u>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)MOE_MXFP4_GATE_UP_NT16_SMEM_MAX);
    (void)cudaGetLastError();       /* clear: a FAILED setattr is sticky and would surface as the next launch's error */
    if (at != cudaSuccess) {
        fprintf(stderr,
                "ds4: MoE MXFP4 gate/up: dynamic-SMEM opt-in for %u B refused (%s); "
                "using the NT=8 tile\n",
                MOE_MXFP4_GATE_UP_NT16_SMEM_MAX, cudaGetErrorString(at));
        return 8u;
    }
    return 16u;
}

/* NOTE: the attribute cudaFuncSetAttribute sets is per-device/per-context, while
 * this cache is per-process.  Tearing down and recreating the primary context, or
 * selecting a second device with a smaller opt-in cap, would leave a cached 16
 * launching against a kernel whose attribute is unset -- which surfaces as a loud
 * cudaErrorInvalidValue on the launch check, not as bad output.  No
 * cudaDeviceReset exists in the tree and the target is single-GPU, so this is
 * latent; it needs re-resolving per device if that ever changes. */
template <uint32_t ROW_SPAN>
static uint32_t moe_mxfp4_gate_up_tile_width(void) {
    static uint32_t width = 0;      /* 0 = unresolved; resolved once, then cached */
    if (!width) width = moe_mxfp4_gate_up_resolve_tile_width<ROW_SPAN>();
    return width;
}


/* Q2_K gate/up variants of the three gate kernels above (mid preset:
 * all-Q2_K experts). Same structure and q8_K activations; the dot is the
 * production-validated dev_dot_q2_K_q8_K_block/block8 the down kernels use. */
__global__ static void moe_gate_up_mid_q2k_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q2_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q2_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}


__global__ static void moe_gate_up_mid_q2k_expert_tile8_row32_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
        dev_dot_q2_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}


template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q2k_expert_tile8_rowspan_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
            dev_dot_q2_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}



__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}



/* MXFP4 routed down (type-39 experts). Generic correctness-first variant. */
__global__ static void moe_down_mxfp4_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const unsigned char *wr = (const unsigned char *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_mxfp4_q8_K_block(wr + (uint64_t)b * 8u * 17u, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}



/* Fused decode down: the 6 selected experts' down-projections summed straight into
 * the final output row (skips the separate down buffer + moe_sum pass). n_expert==6. */
__global__ static void moe_down_mxfp4_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const unsigned char *wr = (const unsigned char *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_mxfp4_q8_K_block(wr + (uint64_t)b * 8u * 17u, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}



__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}



__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
}



__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                /* per-pair store; the fixed-order moe_sum pass accumulates.
                 * (was atomicAdd per token: float order varied with tile
                 * scheduling AND with the sorted-pair scatter order -- the
                 * prefill nondeterminism behind the #17 flake) */
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}



/* IQ2_XXS down variants of the down kernels above (v5mx mixed per-layer
 * expert combos: iq2_xxs down paired with any gate/up format). Same
 * structure and q8_K mid activations; the dots are the production-validated
 * dev_dot_iq2_xxs_q8_K_block / _block_lut / _block8_deq_lut helpers the
 * gate/up kernels use. Prefill tiles keep the per-pair store + fixed-order
 * moe_sum accumulation (no atomicAdd) so numerics stay run-to-run
 * deterministic, matching the Q2_K down family. */
__global__ static void moe_down_iq2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_iq2_xxs_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}



/* Fused decode down for iq2_xxs experts: the 6 selected experts' down
 * projections summed straight into the final output row (skips the separate
 * down buffer + moe_sum pass). n_expert==6. Uses the smem-LUT iq2 dot like
 * the decode gate kernel. */
__global__ static void moe_down_iq2_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    __syncthreads();
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_iq2_xxs_q8_K_block_lut(wr + b, xq + b, s_iq2_grid, s_iq2_signs);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}



__global__ static void moe_down_iq2_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    __syncthreads();
    if (midq_blocks <= 8u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
}



__global__ static void moe_down_iq2_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    __syncthreads();
    if (midq_blocks <= 8u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc,
                                                s_iq2_grid, s_iq2_signs);
            if (np > 8u) {
                dev_dot_iq2_xxs_q8_K_block8_deq_lut(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                                    xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                                    xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                                    xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8,
                                                    s_iq2_grid, s_iq2_signs);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                /* per-pair store; the fixed-order moe_sum pass accumulates
                 * (deterministic -- see the Q2_K tile16 kernel above). */
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}



/* MXFP4 (type-39) down big-batch expert-tiled kernel: mirror of
 * moe_down_iq2_expert_tile16_row2048_kernel with the MXFP4 dot. Per-pair stores
 * into the flat down buffer + fixed-order moe_sum (NO atomicAdd) -> deterministic,
 * same as the iq2/base tile16 kernels. Bit-identical to the qwarp32 down per pair. */
__global__ static void moe_down_mxfp4_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const unsigned char *wr = (const unsigned char *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_mxfp4_q8_K_block8(wr + (uint64_t)b * 8u * 17u,
                                      xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_mxfp4_q8_K_block8(wr + (uint64_t)b * 8u * 17u,
                                          xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                          xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                          xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                          xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
}



__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}



__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}



__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}



__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}



/* ---- CUTLASS MXFP4 (type 40) grouped-expert dispatch. --------------------------------------
 * Unlike the qwarp32/dp4a paths above, which process each (token,expert) pair independently on
 * device, CUTLASS's dense GemmUniversal interface needs one contiguous [T_e,in_dim] activation
 * matrix per expert with T_e known host-side at launch time. So this path: (1) sorts tokens by
 * expert exactly like use_sorted_pairs above, (2) reads the resulting per-expert offsets back to
 * host (the one sync point this path has that the others don't -- n_total_expert is small, so
 * this is a bounded single readback per CUTLASS layer per forward pass, not per expert), then
 * (3) for each active expert: gather its rows into a contiguous scratch buffer, run the CUTLASS
 * FFN, scatter the result into a [n_tokens*n_expert,out_dim] flat buffer at the pair's slot --
 * reusing `down` as that flat buffer and moe_sum_kernel for the final per-token reduction is
 * exactly the existing convention other paths use, just fed from CUTLASS instead of dp4a. */

__global__ static void moe_cutlass_gather_kernel(
        float *x_gathered,
        float *w_gathered,
        const float *x,
        const float *weights,
        const uint32_t *sorted_pairs,
        uint32_t pair_offset,
        uint32_t count,
        uint32_t n_expert,
        uint32_t in_dim) {
    uint32_t i = blockIdx.x;
    if (i >= count) return;
    uint32_t pair = sorted_pairs[pair_offset + i];
    uint32_t tok = pair / n_expert;
    const float *src = x + (uint64_t)tok * in_dim;
    float *dst = x_gathered + (uint64_t)i * in_dim;
    for (uint32_t k = threadIdx.x; k < in_dim; k += blockDim.x) dst[k] = src[k];
    if (threadIdx.x == 0) w_gathered[i] = weights[pair];
}



__global__ static void moe_cutlass_scatter_kernel(
        float *down_flat,
        const float *ffn_out,
        const uint32_t *sorted_pairs,
        uint32_t pair_offset,
        uint32_t count,
        uint32_t out_dim) {
    uint32_t i = blockIdx.x;
    if (i >= count) return;
    uint32_t pair = sorted_pairs[pair_offset + i];
    const float *src = ffn_out + (uint64_t)i * out_dim;
    float *dst = down_flat + (uint64_t)pair * out_dim;
    for (uint32_t k = threadIdx.x; k < out_dim; k += blockDim.x) dst[k] = src[k];
}



static uint64_t cutlass_moe_align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

/* gate_stride/gate_data_bytes and down_stride/down_data_bytes come from
 * routed_expert_gate_down_layout()'s CUTLASS_MXFP4 branch: *_stride is the full per-expert
 * [data+SF] block size, *_data_bytes is where the SF blob starts within that block (the
 * "row_bytes" parameter slot, repurposed -- see that function's comment in weights.c). */
static int routed_moe_launch_cutlass(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_stride,
        uint64_t gate_data_bytes,
        uint64_t down_stride,
        uint64_t down_data_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!out || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }

    const uint64_t gate_total_bytes = (uint64_t)n_total_expert * gate_stride;
    const uint64_t down_total_bytes = (uint64_t)n_total_expert * down_stride;
    if (gate_total_bytes > model_size - gate_offset ||
        gate_total_bytes > model_size - up_offset ||
        down_total_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total_bytes, "moe_cutlass_gate");
    const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total_bytes, "moe_cutlass_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total_bytes, "moe_cutlass_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t pair_count = n_tokens * n_expert;
    const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
    const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1) * sizeof(uint32_t);
    const uint64_t cursors_bytes = counts_bytes;
    const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);

    /* Safe upper bound for the per-expert FFN scratch: one expert could receive every token
     * in this batch. Sizing at n_tokens (rather than the true, smaller max observed count)
     * means everything fits in ONE cuda_tmp_alloc call -- cuda_tmp_alloc is a single global
     * buffer that may cudaFree-and-regrow on a size increase, so a second call after reading
     * back real per-expert counts could invalidate the sorted_pairs region we still need. */
    const uint32_t T_max = n_tokens;
    const uint64_t gather_x_bytes = (uint64_t)T_max * expert_in_dim * sizeof(float);
    const uint64_t gather_w_bytes = (uint64_t)T_max * sizeof(float);
    const uint64_t ffn_out_bytes = (uint64_t)T_max * out_dim * sizeof(float);
    const uint64_t ffn_scratch_bytes = ds4_cutlass_expert_ffn_scratch_bytes(
            (int)T_max, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim);

    const uint64_t align = 256;
    uint64_t off = 0;
    const uint64_t counts_off = off;  off = cutlass_moe_align_up(off + counts_bytes, align);
    const uint64_t offsets_off = off; off = cutlass_moe_align_up(off + offsets_bytes, align);
    const uint64_t cursors_off = off; off = cutlass_moe_align_up(off + cursors_bytes, align);
    const uint64_t sorted_off = off;  off = cutlass_moe_align_up(off + sorted_bytes, align);
    const uint64_t gx_off = off;      off = cutlass_moe_align_up(off + gather_x_bytes, align);
    const uint64_t gw_off = off;      off = cutlass_moe_align_up(off + gather_w_bytes, align);
    const uint64_t fo_off = off;      off = cutlass_moe_align_up(off + ffn_out_bytes, align);
    const uint64_t fs_off = off;      off = cutlass_moe_align_up(off + ffn_scratch_bytes, align);
    const uint64_t total_scratch = off;

    uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(total_scratch, "routed_moe cutlass");
    if (!scratch) return 0;

    uint32_t *counts = (uint32_t *)(scratch + counts_off);
    uint32_t *offsets = (uint32_t *)(scratch + offsets_off);
    uint32_t *cursors = (uint32_t *)(scratch + cursors_off);
    uint32_t *sorted_pairs = (uint32_t *)(scratch + sorted_off);
    float *x_gathered = (float *)(scratch + gx_off);
    float *w_gathered = (float *)(scratch + gw_off);
    float *ffn_out = (float *)(scratch + fo_off);
    uint8_t *ffn_scratch = scratch + fs_off;

    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    int ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe_cutlass counts clear");
    if (ok) {
        moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(counts, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "routed_moe_cutlass count launch");
    }
    if (ok) {
        moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, n_total_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe_cutlass prefix launch");
    }
    if (ok) {
        moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(sorted_pairs, cursors, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "routed_moe_cutlass scatter launch");
    }
    if (!ok) return 0;

    std::vector<uint32_t> h_offsets((size_t)n_total_expert + 1);
    if (!cuda_ok(cudaMemcpy(h_offsets.data(), offsets, offsets_bytes, cudaMemcpyDeviceToHost),
                "routed_moe_cutlass offsets readback")) {
        return 0;
    }

    for (uint32_t e = 0; e < n_total_expert; e++) {
        const uint32_t pair_offset = h_offsets[e];
        const uint32_t count = h_offsets[e + 1] - pair_offset;
        if (count == 0) continue;

        moe_cutlass_gather_kernel<<<count, 256>>>(x_gathered, w_gathered,
                (const float *)x->ptr, (const float *)weights->ptr,
                sorted_pairs, pair_offset, count, n_expert, expert_in_dim);
        if (!cuda_ok(cudaGetLastError(), "routed_moe_cutlass gather launch")) return 0;

        const uint8_t *Wg_d = (const uint8_t *)gate_w + (uint64_t)e * gate_stride;
        const uint8_t *Wg_sf = Wg_d + gate_data_bytes;
        const uint8_t *Wu_d = (const uint8_t *)up_w + (uint64_t)e * gate_stride;
        const uint8_t *Wu_sf = Wu_d + gate_data_bytes;
        const uint8_t *Wd_d = (const uint8_t *)down_w + (uint64_t)e * down_stride;
        const uint8_t *Wd_sf = Wd_d + down_data_bytes;

        const int rc = ds4_cutlass_expert_ffn_scratch(ffn_out, x_gathered,
                Wg_d, Wg_sf, Wu_d, Wu_sf, Wd_d, Wd_sf,
                w_gathered, clamp,
                (int)count, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim,
                ffn_scratch, ffn_scratch_bytes);
        if (rc != 0) return 0;

        moe_cutlass_scatter_kernel<<<count, 256>>>((float *)down->ptr, ffn_out,
                sorted_pairs, pair_offset, count, out_dim);
        if (!cuda_ok(cudaGetLastError(), "routed_moe_cutlass scatter launch")) return 0;
    }

    const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
    moe_sum_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
            (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    return cuda_ok(cudaGetLastError(), "routed_moe_cutlass sum launch");
}



/* ---- Grouped MXFP4 prefill path (single ptr-array grouped GEMM per gate/up/down; no host
 * readback, no host sync). Mirrors routed_moe_launch_cutlass but replaces the per-expert loop
 * with ds4_cutlass_grouped_moe. Experts' tokens are gathered to 128-ROW-PADDED offsets so each
 * group's block-scaled SF starts on a 128-row atom boundary (see ds4_cutlass_grouped_moe). ---- */

/* padded_offsets[e] = sum_{e'<e} ceil(counts[e']/128)*128 (per-group 128-row-padded row base). */
__global__ static void moe_padded_offsets_kernel(uint32_t *padded_off, const uint32_t *counts, uint32_t n_total) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t run = 0;
    for (uint32_t e = 0; e < n_total; e++) {
        padded_off[e] = run;
        run += (counts[e] + 127u) / 128u * 128u;
    }
}

/* One block per sorted-pair slot s: locate its expert e (offsets[e] <= s < offsets[e+1]), place
 * the token's activation row at padded row padded_off[e] + (s - offsets[e]), record the routing
 * weight and the originating pair index (padded_pair[R]) for the later scatter. */
__global__ static void moe_padded_gather_kernel(
        float *x_gathered, float *w_gathered, int32_t *padded_pair,
        const float *x, const float *weights,
        const uint32_t *sorted_pairs, const uint32_t *offsets, const uint32_t *padded_off,
        uint32_t pair_count, uint32_t n_total, uint32_t n_expert, uint32_t in_dim) {
    uint32_t s = blockIdx.x;
    if (s >= pair_count) return;
    /* upper_bound(offsets, s) - 1 over [0,n_total) */
    uint32_t lo = 0, hi = n_total;
    while (lo < hi) {
        uint32_t midx = (lo + hi) >> 1;
        if (offsets[midx] <= s) lo = midx + 1; else hi = midx;
    }
    uint32_t e = lo - 1u;
    uint32_t i = s - offsets[e];
    uint32_t R = padded_off[e] + i;
    uint32_t pair = sorted_pairs[s];
    uint32_t tok = pair / n_expert;
    const float *src = x + (uint64_t)tok * in_dim;
    float *dst = x_gathered + (uint64_t)R * in_dim;
    for (uint32_t k = threadIdx.x; k < in_dim; k += blockDim.x) dst[k] = src[k];
    if (threadIdx.x == 0) { w_gathered[R] = weights[pair]; padded_pair[R] = (int32_t)pair; }
}

/* One block per padded row R: scatter the pre-weighted FFN result back to the flat down buffer at
 * its originating pair (padding rows carry padded_pair<0 and are skipped). */
__global__ static void moe_padded_scatter_kernel(
        float *down_flat, const float *ffn_out, const int32_t *padded_pair,
        uint32_t padded_total, uint32_t out_dim) {
    uint32_t R = blockIdx.x;
    if (R >= padded_total) return;
    int32_t pair = padded_pair[R];
    if (pair < 0) return;
    const float *src = ffn_out + (uint64_t)R * out_dim;
    float *dst = down_flat + (uint64_t)pair * out_dim;
    for (uint32_t k = threadIdx.x; k < out_dim; k += blockDim.x) dst[k] = src[k];
}

static int routed_moe_launch_cutlass_grouped(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_stride,
        uint64_t gate_data_bytes,
        uint64_t down_stride,
        uint64_t down_data_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!out || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }

    const uint64_t gate_total_bytes = (uint64_t)n_total_expert * gate_stride;
    const uint64_t down_total_bytes = (uint64_t)n_total_expert * down_stride;
    if (gate_total_bytes > model_size - gate_offset ||
        gate_total_bytes > model_size - up_offset ||
        down_total_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total_bytes, "moe_grouped_gate");
    const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total_bytes, "moe_grouped_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total_bytes, "moe_grouped_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t pair_count = n_tokens * n_expert;
    /* Host upper bound on padded rows: every active expert adds <=127 padding; round to 128. */
    const uint64_t padded_upper = (((uint64_t)pair_count + 128ull * n_total_expert + 127ull) / 128ull) * 128ull;

    const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
    const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1) * sizeof(uint32_t);
    const uint64_t cursors_bytes = counts_bytes;
    const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
    const uint64_t padoff_bytes = counts_bytes;
    const uint64_t xg_bytes = padded_upper * expert_in_dim * sizeof(float);
    const uint64_t wg_bytes = padded_upper * sizeof(float);
    const uint64_t ppair_bytes = padded_upper * sizeof(int32_t);
    const uint64_t ffn_bytes = padded_upper * out_dim * sizeof(float);
    const uint64_t grp_bytes = ds4_cutlass_grouped_moe_scratch_bytes(
            (int)padded_upper, (int)n_total_expert, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim);

    const uint64_t align = 256;
    uint64_t off = 0;
    const uint64_t counts_off = off;  off = cutlass_moe_align_up(off + counts_bytes, align);
    const uint64_t offsets_off = off; off = cutlass_moe_align_up(off + offsets_bytes, align);
    const uint64_t cursors_off = off; off = cutlass_moe_align_up(off + cursors_bytes, align);
    const uint64_t sorted_off = off;  off = cutlass_moe_align_up(off + sorted_bytes, align);
    const uint64_t padoff_off = off;  off = cutlass_moe_align_up(off + padoff_bytes, align);
    const uint64_t xg_off = off;      off = cutlass_moe_align_up(off + xg_bytes, align);
    const uint64_t wg_off = off;      off = cutlass_moe_align_up(off + wg_bytes, align);
    const uint64_t ppair_off = off;   off = cutlass_moe_align_up(off + ppair_bytes, align);
    const uint64_t ffn_off = off;     off = cutlass_moe_align_up(off + ffn_bytes, align);
    const uint64_t grp_off = off;     off = cutlass_moe_align_up(off + grp_bytes, align);
    const uint64_t total_scratch = off;

    uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(total_scratch, "routed_moe grouped");
    if (!scratch) return 0;

    uint32_t *counts = (uint32_t *)(scratch + counts_off);
    uint32_t *offsets = (uint32_t *)(scratch + offsets_off);
    uint32_t *cursors = (uint32_t *)(scratch + cursors_off);
    uint32_t *sorted_pairs = (uint32_t *)(scratch + sorted_off);
    uint32_t *padded_off = (uint32_t *)(scratch + padoff_off);
    float *x_gathered = (float *)(scratch + xg_off);
    float *w_gathered = (float *)(scratch + wg_off);
    int32_t *padded_pair = (int32_t *)(scratch + ppair_off);
    float *ffn_out = (float *)(scratch + ffn_off);
    uint8_t *grp_scratch = scratch + grp_off;

    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    int ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "moe_grouped counts clear");
    /* padding rows must be zeroed (pack sees clean data) and unmapped (padded_pair = -1). */
    if (ok) ok = cuda_ok(cudaMemsetAsync(x_gathered, 0, xg_bytes), "moe_grouped xg clear");
    if (ok) ok = cuda_ok(cudaMemsetAsync(w_gathered, 0, wg_bytes), "moe_grouped wg clear");
    if (ok) ok = cuda_ok(cudaMemsetAsync(padded_pair, 0xFF, ppair_bytes), "moe_grouped ppair clear");
    if (ok) {
        moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(counts, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped count launch");
    }
    if (ok) {
        moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, n_total_expert);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped prefix launch");
    }
    if (ok) {
        moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(sorted_pairs, cursors, selected_ptr, pair_count);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped scatter launch");
    }
    if (ok) {
        moe_padded_offsets_kernel<<<1, 1>>>(padded_off, counts, n_total_expert);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped padded offsets launch");
    }
    if (ok) {
        moe_padded_gather_kernel<<<pair_count, 256>>>(x_gathered, w_gathered, padded_pair,
                (const float *)x->ptr, (const float *)weights->ptr,
                sorted_pairs, offsets, padded_off, pair_count, n_total_expert, n_expert, expert_in_dim);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped gather launch");
    }
    if (ok) {
        int rc = ds4_cutlass_grouped_moe(ffn_out, x_gathered, w_gathered,
                (const uint8_t *)gate_w, (const uint8_t *)up_w, (const uint8_t *)down_w,
                gate_stride, gate_data_bytes, down_stride, down_data_bytes,
                clamp, (int)n_total_expert, (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim,
                counts, padded_off, (int)padded_upper, grp_scratch, grp_bytes);
        if (rc != 0) return 0;
    }
    if (ok) {
        moe_padded_scatter_kernel<<<(uint32_t)padded_upper, 256>>>((float *)down->ptr, ffn_out,
                padded_pair, (uint32_t)padded_upper, out_dim);
        ok = cuda_ok(cudaGetLastError(), "moe_grouped padded scatter launch");
    }
    if (!ok) return 0;

    const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
    moe_sum_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
            (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    return cuda_ok(cudaGetLastError(), "moe_grouped sum launch");
}

/* A/B dispatcher: DS4_MOE_FP4_GROUPED=0 forces the legacy per-expert loop (routed_moe_launch_cutlass);
 * default (unset or !=0) uses the grouped single-launch path. Same result, bit-exact. */
static int routed_moe_launch_cutlass_dispatch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_stride, uint64_t gate_data_bytes, uint64_t down_stride, uint64_t down_data_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t n_tokens) {
    static int grouped = -1;
    if (grouped < 0) {
        const char *e = getenv("DS4_MOE_FP4_GROUPED");
        grouped = !(e && e[0] == '0');
    }
    if (grouped) {
        int rc = routed_moe_launch_cutlass_grouped(out, down, model_map, model_size,
                gate_offset, up_offset, down_offset, gate_stride, gate_data_bytes,
                down_stride, down_data_bytes, expert_in_dim, expert_mid_dim, out_dim,
                selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
        {
            static int glog = -1;
            if (glog < 0) glog = getenv("DS4_MOE_GROUPED_LOG") != NULL ? 1 : 0;
            static int logged = 0;
            if (glog && !logged) { logged = 1;
                fprintf(stderr, "ds4: moe grouped path rc=%d n_tok=%u n_total=%u n_exp=%u -> %s\n",
                        rc, n_tokens, n_total_expert, n_expert, rc ? "USED grouped" : "FELL BACK to per-expert"); }
        }
        if (rc) return rc;   /* any failure falls through to the legacy loop (safety net) */
    }
    return routed_moe_launch_cutlass(out, down, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_stride, gate_data_bytes,
            down_stride, down_data_bytes, expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
}



static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    /* Per-role, per-layer kernel selection (v5mx mixed combos): gate/up (a
     * matching pair, enforced at load) in {iq2_xxs, q2_k, mxfp4}, down in
     * {iq2_xxs, q2_k, mxfp4}, any pairing; combos vary layer to layer. */
    const int gate_mxfp4 = gate_type == 39u;
    const int gate_q2k = gate_type == 10u;
    const int down_mxfp4 = down_type == 39u;
    const int down_iq2 = down_type == 16u;
    /* MXFP4 (type-39) big-batch expert-tiled prefill path (DS4_MOE_FP4_TILED, default on;
     * =0 restores the per-pair qwarp32 kernels). Bit-identical, ~10x faster at prefill. */
    static int fp4_tiled = -1;
    if (fp4_tiled < 0) { const char *e = getenv("DS4_MOE_FP4_TILED"); fp4_tiled = !(e && e[0] == '0'); }
    if (gate_type != 16u && !gate_q2k && !gate_mxfp4) return 0;
    if (down_type != 10u && !down_iq2 && !down_mxfp4) return 0;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        /* One release configuration (the tuned production values from the old
         * DS4_CUDA_MOE_* env matrix): expert-major sorted pair tiles, 16-row
         * down tiles + row-span kernels for big prefill batches, LUT gate for
         * decode, direct down-sum for the 6-expert decode case.  (This said
         * "atomic down accumulation" until this commit; there is no atomicAdd
         * on any down path -- every output element is written by exactly one
         * thread, per-pair stores plus a fixed-order moe_sum_kernel, which is
         * precisely why these stages are bit-exact and re-tileable.) */
        const uint32_t pair_count = n_tokens * n_expert;
        /* Sorted expert tiles exist for the iq2_xxs/q2_k kernels only; mxfp4
         * stages take the per-pair qwarp32 kernels. Build the sorted
         * structures whenever at least one stage consumes them. */
        /* With the tiled MXFP4 path enabled, both-mxfp4 layers also build the sorted
         * structures (the tiled kernels consume them). Without it, keep the legacy
         * behaviour (both-mxfp4 skips sorting and takes qwarp32). */
        const uint32_t use_sorted_pairs =
            n_tokens > 1u && (fp4_tiled || !(gate_mxfp4 && down_mxfp4));
        const uint32_t use_big_batch = use_sorted_pairs && n_tokens >= 128u;
        const uint32_t use_decode_lut_gate =
            gate_type == 16u && n_tokens == 1u && xq_blocks <= 16u;
        const uint32_t use_direct_down_sum6 = n_tokens == 1u && n_expert == 6u;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint32_t sort_expert_count = n_total_expert;
            if (sort_expert_count == 0) ok = 0;
            const uint64_t counts_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + 7u) / 8u + sort_expert_count;
            tile16_capacity = use_big_batch ? ((pair_count + 15u) / 16u + sort_expert_count) : 0u;
            const uint64_t tile_offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_big_batch ? ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_big_batch ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_big_batch ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_big_batch ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_big_batch ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_big_batch ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, sort_expert_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        sorted_pairs,
                        cursors,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, sort_expert_count, 8u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile_experts, tile_starts, tile_offsets, counts, sort_expert_count, 8u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_big_batch) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_big_batch) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok) {
            if (sorted_pairs && !gate_mxfp4) {
                if (use_big_batch) {
                    dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                    if (gate_q2k) {
                        moe_gate_up_mid_q2k_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            clamp);
                    } else {
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            clamp);
                    }
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    if (gate_q2k) {
                        moe_gate_up_mid_q2k_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            clamp);
                    } else {
                        moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            clamp);
                    }
                }
            } else if (sorted_pairs && gate_mxfp4 && fp4_tiled && use_big_batch) {
                /* NT=16 where the device allows the dynamic-SMEM opt-in (measured
                 * 2.30x, bit-exact), else the NT=8 fallback.  The two widths must
                 * read tile lists of their OWN width: a 16-token tile walking the
                 * tile-8 list would re-process pairs and skip others.  Both lists
                 * are already built above under use_big_batch, which this branch
                 * requires; the mxfp4 down stage has consumed the tile16 list since
                 * 7c92951. */
                /* NT=16 ONLY WHERE IT CAN STAGE.  The width and the staging are not
                 * independent knobs -- measured at the v5mx shape: staged, 8 ->
                 * 171.67 ms and 16 -> 74.64 (2.30x); UNSTAGED, 8 -> 122.53 but 16 ->
                 * 152.71, i.e. unstaged the WIDER tile is ~25% SLOWER than the
                 * narrower one.  So when xq_blocks > 16 puts staging out of reach
                 * (moe_mxfp4_gate_up_smem_bytes -> 0), widening would be a
                 * regression, not a win: keep NT=8 there. */
                const uint32_t nt = moe_mxfp4_gate_up_smem_bytes(16u, xq_blocks)
                                        ? moe_mxfp4_gate_up_tile_width<1024u>()
                                        : 8u;
                const uint32_t smem = moe_mxfp4_gate_up_smem_bytes(nt, xq_blocks);
                if (nt == 16u) {
                    dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile16_capacity, 1);
                    moe_gate_up_mid_mxfp4_expert_ntile_rowspan_kernel<1024u, 16u><<<tgrid, 256, smem>>>(
                        (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile16_total, tile16_experts, tile16_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                    moe_gate_up_mid_mxfp4_expert_ntile_rowspan_kernel<1024u, 8u><<<tgrid, 256, smem>>>(
                        (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        clamp);
                }
            } else {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (gate_mxfp4) {
                    moe_gate_up_mid_mxfp4_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else if (gate_q2k) {
                    moe_gate_up_mid_q2k_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (down_mxfp4) {
                    moe_down_mxfp4_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else if (down_iq2) {
                    moe_down_iq2_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                }
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && down_mxfp4 && fp4_tiled && use_big_batch) {
                dim3 tgrid((out_dim + 2047u) / 2048u, tile16_capacity, 1);
                moe_down_mxfp4_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                    (float *)down->ptr,
                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                    tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                    midq_blocks, out_dim, n_expert);
            } else if (sorted_pairs && !down_mxfp4 && use_big_batch) {
                dim3 tgrid((out_dim + 2047u) / 2048u, tile16_capacity, 1);
                if (down_iq2) {
                    moe_down_iq2_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                        (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert);
                } else {
                    moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                        (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert);
                }
            } else if (sorted_pairs && !down_mxfp4) {
                dim3 tgrid((out_dim + 31u) / 32u, tile_capacity, 1);
                if (down_iq2) {
                    moe_down_iq2_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert);
                } else {
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert);
                }
            } else {
                if (down_mxfp4) {
                    moe_down_mxfp4_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else if (down_iq2) {
                    moe_down_iq2_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_direct_down_sum6) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        "ds4: CUDA MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            selected_ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            selected_ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}



extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index) {
    if (gate_type == 40u && down_type == 40u) {
        /* Decode (n=1) takes the same direct fp4 GEMV as small verify batches:
         * 4 launches with no host round-trip, vs the grouped path's BLOCKING
         * per-layer offsets readback -- required for CUDA graph capture of the
         * decode tape, and computes the exact same function (see the batch
         * path's comment; bit-exact oracle in temp/fp4gemv_test.cu covers
         * n_tokens>=1). DS4_MOE_FP4_GEMV=0 restores the grouped dispatch. */
        static int fp4_gemv = -1;
        if (fp4_gemv < 0) {
            const char *e = getenv("DS4_MOE_FP4_GEMV");
            fp4_gemv = !(e && e[0] == '0');
        }
        if (fp4_gemv &&
            mid && mid->ptr && down && down->ptr && out && out->ptr &&
            selected && selected->ptr && weights && weights->ptr && x && x->ptr &&
            mid->bytes >= (uint64_t)n_expert * expert_mid_dim * sizeof(float) &&
            down->bytes >= (uint64_t)n_expert * out_dim * sizeof(float) &&
            out->bytes >= (uint64_t)out_dim * sizeof(float) &&
            selected->bytes >= (uint64_t)n_expert * sizeof(int32_t) &&
            weights->bytes >= (uint64_t)n_expert * sizeof(float) &&
            x->bytes >= (uint64_t)expert_in_dim * sizeof(float)) {
            const uint64_t gate_total = (uint64_t)n_total_expert * gate_expert_bytes;
            const uint64_t down_total = (uint64_t)n_total_expert * down_expert_bytes;
            const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total, "moe_fp4_gemv_gate");
            const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total, "moe_fp4_gemv_up");
            const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total, "moe_fp4_gemv_down");
            if (gate_w && up_w && down_w &&
                ds4_cutlass_expert_ffn_gemv_small(
                        (float *)down->ptr, (float *)mid->ptr, (const float *)x->ptr,
                        (const int32_t *)selected->ptr, (const float *)weights->ptr,
                        (const uint8_t *)gate_w, (const uint8_t *)up_w, (const uint8_t *)down_w,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        clamp, 1, (int)n_expert, n_total_expert,
                        (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim) == 0) {
                moe_sum_kernel<<<(uint32_t)((out_dim + 255u) / 256u), 256>>>(
                        (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, 1u);
                if (cuda_ok(cudaGetLastError(), "moe fp4 gemv sum")) return 1;
            }
            /* any failure: fall through to the grouped CUTLASS path */
        }
        return routed_moe_launch_cutlass(out, down, model_map, model_size,
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, gate_row_bytes,
                                         down_expert_bytes, down_row_bytes,
                                         expert_in_dim, expert_mid_dim, out_dim,
                                         selected, weights, n_total_expert, n_expert, clamp, x, 1);
    }
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, 1);
}


static int routed_moe_batch_impl(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    {
        static int entry_log = -1;
        if (entry_log < 0) entry_log = getenv("DS4_MOE_PATH_LOG") != NULL ? 200 : 0;
        if (entry_log > 0) { entry_log--;
            fprintf(stderr, "ds4: moe_batch ENTRY layer=%u gate_type=%u down_type=%u n_tok=%u\n",
                    layer_index, gate_type, down_type, n_tokens); }
    }
    if (gate_type == 40u && down_type == 40u) {
        /* Small batches (spec-decode verify, n_tokens 2..4): direct fp4 GEMV over the
         * packed expert weights -- 4 launches per layer (act-roundtrip + gate/up+swiglu
         * + act-roundtrip + down) vs the grouped path's BLOCKING per-layer offsets
         * readback + ~5 launches per active expert. Computes the exact same function
         * as the grouped GEMMs (weights read in the verified row-major CUTLASS B
         * layout; activations round-tripped through the same fp4 quantizer) -- proven
         * bit-exact vs a quant-exact CPU oracle (temp/fp4gemv_test.cu) and clean-text
         * on-model. Measured verify(3) step 118.2 -> 116.5 ms; the bigger win is
         * removing the per-rich-layer host sync from the verify path (CUDA-graph
         * prerequisite). n_tokens==1 (decode) keeps the grouped path unchanged.
         * DS4_MOE_FP4_GEMV=0 restores the grouped dispatch. */
        static int fp4_gemv = -1;
        if (fp4_gemv < 0) {
            const char *e = getenv("DS4_MOE_FP4_GEMV");
            fp4_gemv = !(e && e[0] == '0');
        }
        static int path_log = -1;
        if (path_log < 0) path_log = getenv("DS4_MOE_PATH_LOG") != NULL ? 400 : 0;
        if (path_log > 0) {
            path_log--;
            fprintf(stderr,
                    "ds4: moe40 layer=%u n_tok=%u n_total=%u stride=%llu split=%llu mid=%d down=%d midb=%llu need=%llu\n",
                    layer_index, n_tokens, n_total_expert,
                    (unsigned long long)gate_expert_bytes, (unsigned long long)gate_row_bytes,
                    mid && mid->ptr ? 1 : 0, down && down->ptr ? 1 : 0,
                    (unsigned long long)(mid ? mid->bytes : 0),
                    (unsigned long long)((uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float)));
        }
        if (fp4_gemv && n_tokens >= 2u && n_tokens <= 4u &&
            mid && mid->ptr && down && down->ptr && out && out->ptr &&
            selected && selected->ptr && weights && weights->ptr && x && x->ptr &&
            mid->bytes >= (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) &&
            down->bytes >= (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) &&
            out->bytes >= (uint64_t)n_tokens * out_dim * sizeof(float) &&
            selected->bytes >= (uint64_t)n_tokens * n_expert * sizeof(int32_t) &&
            weights->bytes >= (uint64_t)n_tokens * n_expert * sizeof(float) &&
            x->bytes >= (uint64_t)n_tokens * expert_in_dim * sizeof(float)) {
            const uint64_t gate_total = (uint64_t)n_total_expert * gate_expert_bytes;
            const uint64_t down_total = (uint64_t)n_total_expert * down_expert_bytes;
            const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total, "moe_fp4_gemv_gate");
            const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total, "moe_fp4_gemv_up");
            const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total, "moe_fp4_gemv_down");
            if (gate_w && up_w && down_w &&
                ds4_cutlass_expert_ffn_gemv_small(
                        (float *)down->ptr, (float *)mid->ptr, (const float *)x->ptr,
                        (const int32_t *)selected->ptr, (const float *)weights->ptr,
                        (const uint8_t *)gate_w, (const uint8_t *)up_w, (const uint8_t *)down_w,
                        gate_expert_bytes, gate_row_bytes,
                        down_expert_bytes, down_row_bytes,
                        clamp, (int)n_tokens, (int)n_expert, n_total_expert,
                        (int)expert_in_dim, (int)expert_mid_dim, (int)out_dim) == 0) {
                const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
                moe_sum_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
                        (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
                if (cuda_ok(cudaGetLastError(), "moe fp4 gemv sum")) {
                    if (path_log > 0) fprintf(stderr, "ds4: moe40 layer=%u -> gemv\n", layer_index);
                    return 1;
                }
            }
            /* any failure: fall through to the grouped CUTLASS path */
        }
        if (path_log > 0) fprintf(stderr, "ds4: moe40 layer=%u -> grouped\n", layer_index);
        return routed_moe_launch_cutlass_dispatch(out, down, model_map, model_size,
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, gate_row_bytes,
                                         down_expert_bytes, down_row_bytes,
                                         expert_in_dim, expert_mid_dim, out_dim,
                                         selected, weights, n_total_expert, n_expert, clamp, x,
                                         n_tokens);
    }
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, n_tokens);
}

/* ---- Per-format MoE prefill timing (DS4_MOE_TIME=1). Buckets each batch MoE call's GPU time
 * by (gate_type,down_type) via a CUDA-event pair around the impl, and dumps a table at exit.
 * The event sync perturbs host wall-time but the measured per-call GPU interval is accurate;
 * this is a diagnostic-only path (one getenv read, no per-token flag branching in production). */
struct moe_time_bucket { uint32_t gate_type, down_type; double ms; uint64_t calls; };
static moe_time_bucket g_moe_time[64];
static int g_moe_time_n = 0;
static void moe_time_dump(void){
    if (g_moe_time_n == 0) return;
    double tot = 0; for (int i = 0; i < g_moe_time_n; i++) tot += g_moe_time[i].ms;
    fprintf(stderr, "\n=== DS4_MOE_TIME per-format MoE batch GPU time (total %.2f ms over all calls) ===\n", tot);
    for (int i = 0; i < g_moe_time_n; i++) {
        moe_time_bucket *b = &g_moe_time[i];
        fprintf(stderr, "  gate_type=%2u down_type=%2u : %9.2f ms  (%5.1f%%)  calls=%llu  avg=%.3f ms\n",
                b->gate_type, b->down_type, b->ms, tot > 0 ? 100.0 * b->ms / tot : 0.0,
                (unsigned long long)b->calls, b->calls ? b->ms / (double)b->calls : 0.0);
    }
    fprintf(stderr, "=== end DS4_MOE_TIME ===\n");
}
static void moe_time_accum(uint32_t gt, uint32_t dt, double ms){
    for (int i = 0; i < g_moe_time_n; i++)
        if (g_moe_time[i].gate_type == gt && g_moe_time[i].down_type == dt) {
            g_moe_time[i].ms += ms; g_moe_time[i].calls++; return;
        }
    if (g_moe_time_n < 64) {
        static int registered = 0;
        if (!registered) { registered = 1; atexit(moe_time_dump); }
        g_moe_time[g_moe_time_n].gate_type = gt; g_moe_time[g_moe_time_n].down_type = dt;
        g_moe_time[g_moe_time_n].ms = ms; g_moe_time[g_moe_time_n].calls = 1; g_moe_time_n++;
    }
}

extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16) {
    static int time_moe = -1;
    if (time_moe < 0) time_moe = getenv("DS4_MOE_TIME") != NULL ? 1 : 0;
    if (!time_moe) {
        return routed_moe_batch_impl(out, gate, up, mid, down, model_map, model_size,
                gate_offset, up_offset, down_offset, gate_type, down_type,
                gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
                expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                n_total_expert, n_expert, clamp, x, layer_index, n_tokens, mid_is_f16);
    }
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s, 0);
    int r = routed_moe_batch_impl(out, gate, up, mid, down, model_map, model_size,
            gate_offset, up_offset, down_offset, gate_type, down_type,
            gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim, selected, weights,
            n_total_expert, n_expert, clamp, x, layer_index, n_tokens, mid_is_f16);
    cudaEventRecord(e, 0);
    if (cudaEventSynchronize(e) == cudaSuccess) {
        float ms = 0; cudaEventElapsedTime(&ms, s, e);
        moe_time_accum(gate_type, down_type, (double)ms);
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return r;
}

