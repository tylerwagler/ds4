#include "ds4_cuda_internal.h"

#include "ds4_iq2_tables_cuda.inc"



__global__ static void zero_kernel(float *out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}



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



__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}



__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
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
        float acc[8]) {
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
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
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



__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
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



__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
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



__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
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
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
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



__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
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
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
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



__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
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
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
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
        uint32_t write_aux,
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
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
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



__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
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



__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
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
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
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



__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
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
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
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
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
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
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}



__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
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
        uint32_t write_aux,
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
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}



__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
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
        uint32_t write_aux,
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
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
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
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
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
        uint32_t write_aux,
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
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}



__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
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



__global__ static DS4_CUDA_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}



__global__ static DS4_CUDA_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}



__global__ static DS4_CUDA_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
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



__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
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



__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
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
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}



__global__ static void moe_down_expert_tile4_row32_kernel(
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
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
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
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
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
        uint32_t n_expert,
        uint32_t atomic_out) {
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
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}



__global__ static void moe_down_expert_tile16_row32_kernel(
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
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
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
    if (row >= out_dim) return;
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
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
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
        uint32_t n_expert,
        uint32_t atomic_out) {
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
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
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
        uint32_t n_expert,
        uint32_t atomic_out) {
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
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
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
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}



__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
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

/* =====================================================================================
 * 2-bit -> fp4 prefill path: dequant IQ2_XXS gate/up and Q2_K down experts to f32, then
 * pack to CUTLASS MXFP4 (ds4_cutlass_pack_weight_f32) and run the proven CUTLASS grouped
 * FFN -- moving the 37 ordinary MoE layers off the dp4a CUDA-core path onto tensor cores
 * for large-batch prefill. LOSSY (2-bit -> fp4); gated by DS4_MOE_FP4_PREFILL + n_tokens.
 * ===================================================================================== */

/* Dequant one IQ2_XXS weight [N,K] (N rows of K, RowMajor) to f32, matching
 * dev_dot_iq2_xxs_q8_K_block numerics exactly: w = 0.125 * d * ls * signed_grid_int8.
 * One thread per (row, 256-block, ib32) -> 32 output weights. K must be a multiple of 256. */
__global__ static void dequant_iq2xxs_to_f32_kernel(float *W, const cuda_block_iq2_xxs *blocks,
                                                    int N, int K) {
    const int n256 = K / 256;
    const long total = (long)N * n256 * 8;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int ib32 = (int)(idx % 8);
    long t = idx / 8;
    const int b = (int)(t % n256);
    const int row = (int)(t / n256);
    const cuda_block_iq2_xxs *x = blocks + (size_t)row * n256 + b;
    const float d = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs + ib32 * 4;
    const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    const uint32_t ls = 2u * (aux1 >> 28) + 1u;
    const float sc = 0.125f * d * (float)ls;
    const uint8_t a[4] = { (uint8_t)(aux0 & 0xffu), (uint8_t)((aux0 >> 8) & 0xffu),
                           (uint8_t)((aux0 >> 16) & 0xffu), (uint8_t)((aux0 >> 24) & 0xffu) };
    const uint32_t si[4] = { (aux1 >> 0) & 127u, (aux1 >> 7) & 127u,
                             (aux1 >> 14) & 127u, (aux1 >> 21) & 127u };
    float *wrow = W + (size_t)row * K + (size_t)b * 256 + (size_t)ib32 * 32;
    for (int g = 0; g < 4; g++) {
        int32_t w0, w1;
        dev_iq2_i8x8_lut(cuda_iq2xxs_grid, cuda_ksigns_iq2xs, a[g], si[g], &w0, &w1);
        wrow[g * 8 + 0] = sc * (float)(int8_t)(w0 & 0xff);
        wrow[g * 8 + 1] = sc * (float)(int8_t)((w0 >> 8) & 0xff);
        wrow[g * 8 + 2] = sc * (float)(int8_t)((w0 >> 16) & 0xff);
        wrow[g * 8 + 3] = sc * (float)(int8_t)((w0 >> 24) & 0xff);
        wrow[g * 8 + 4] = sc * (float)(int8_t)(w1 & 0xff);
        wrow[g * 8 + 5] = sc * (float)(int8_t)((w1 >> 8) & 0xff);
        wrow[g * 8 + 6] = sc * (float)(int8_t)((w1 >> 16) & 0xff);
        wrow[g * 8 + 7] = sc * (float)(int8_t)((w1 >> 24) & 0xff);
    }
}

/* Dequant one Q2_K weight [N,K] to f32, matching dev_dot_q2_K_q8_K_block numerics:
 * w = d*(sc&0xf)*q - dmin*(sc>>4), q = 2-bit value. One thread per (row, 256-block, k, j)
 * -> 32 output weights (k in 0..1, j in 0..3). K must be a multiple of 256. */
__global__ static void dequant_q2k_to_f32_kernel(float *W, const cuda_block_q2_K *blocks,
                                                 int N, int K) {
    const int n256 = K / 256;
    const long total = (long)N * n256 * 8;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int kj = (int)(idx % 8);
    long t = idx / 8;
    const int b = (int)(t % n256);
    const int row = (int)(t / n256);
    const int k = kj / 4, j = kj % 4;
    const cuda_block_q2_K *x = blocks + (size_t)row * n256 + b;
    const float d = dev_f16_to_f32(x->d), dmin = dev_f16_to_f32(x->dmin);
    const uint8_t *qs = x->qs + (size_t)k * 32;
    const int shift = 2 * j;
    float *wrow = W + (size_t)row * K + (size_t)b * 256 + (size_t)k * 128 + (size_t)j * 32;
    for (int h = 0; h < 2; h++) {
        const uint8_t sc = x->scales[k * 8 + j * 2 + h];
        const float dl = d * (float)(sc & 0x0f), ml = dmin * (float)(sc >> 4);
        for (int i = 0; i < 16; i++) {
            const int q = (qs[h * 16 + i] >> shift) & 3;
            wrow[h * 16 + i] = dl * (float)q - ml;
        }
    }
}

static int ds4_moe_fp4_prefill_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DS4_MOE_FP4_PREFILL"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v;
}

/* IQ2_XXS(gate/up) + Q2_K(down) grouped-expert FFN via the CUTLASS fp4 path. Mirrors
 * routed_moe_launch_cutlass, but each expert's raw 2-bit weights are dequant->f32->fp4-packed
 * into a per-expert weight scratch (reused across experts) just before the FFN call. */
static int routed_moe_launch_2bit_cutlass(
        ds4_gpu_tensor *out, ds4_gpu_tensor *down,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_stride, uint64_t down_stride,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t n_tokens) {
    if (!out || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        (mid_dim & 1u) || (out_dim & 1u) || (in_dim % 256u) || (mid_dim % 256u)) {
        return 0;
    }
    const uint64_t gate_total_bytes = (uint64_t)n_total_expert * gate_stride;
    const uint64_t down_total_bytes = (uint64_t)n_total_expert * down_stride;
    if (gate_total_bytes > model_size - gate_offset ||
        gate_total_bytes > model_size - up_offset ||
        down_total_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total_bytes, "moe_2bit_gate");
    const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total_bytes, "moe_2bit_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total_bytes, "moe_2bit_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t pair_count = n_tokens * n_expert;
    const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
    const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1) * sizeof(uint32_t);
    const uint64_t cursors_bytes = counts_bytes;
    const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
    const uint32_t T_max = n_tokens;
    const uint64_t gather_x_bytes = (uint64_t)T_max * in_dim * sizeof(float);
    const uint64_t gather_w_bytes = (uint64_t)T_max * sizeof(float);
    const uint64_t ffn_out_bytes = (uint64_t)T_max * out_dim * sizeof(float);
    const uint64_t ffn_scratch_bytes = ds4_cutlass_expert_ffn_scratch_bytes(
            (int)T_max, (int)in_dim, (int)mid_dim, (int)out_dim);
    /* per-expert weight scratch (reused every expert): one f32 dequant buffer sized to the
     * larger of gate/up ([mid,in]) and down ([out,mid]), plus packed fp4 data+SF for all three. */
    const uint64_t wtmp_elems = (uint64_t)mid_dim * in_dim > (uint64_t)out_dim * mid_dim
                              ? (uint64_t)mid_dim * in_dim : (uint64_t)out_dim * mid_dim;
    const uint64_t wtmp_bytes = wtmp_elems * sizeof(float);
    const uint64_t gu_d_bytes = ds4_cutlass_weight_data_bytes((int)mid_dim, (int)in_dim);
    const uint64_t gu_sf_bytes = ds4_cutlass_weight_sf_count((int)mid_dim, (int)in_dim);
    const uint64_t dn_d_bytes = ds4_cutlass_weight_data_bytes((int)out_dim, (int)mid_dim);
    const uint64_t dn_sf_bytes = ds4_cutlass_weight_sf_count((int)out_dim, (int)mid_dim);

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
    const uint64_t wt_off = off;      off = cutlass_moe_align_up(off + wtmp_bytes, align);
    const uint64_t wgd_off = off;     off = cutlass_moe_align_up(off + gu_d_bytes, align);
    const uint64_t wgs_off = off;     off = cutlass_moe_align_up(off + gu_sf_bytes, align);
    const uint64_t wud_off = off;     off = cutlass_moe_align_up(off + gu_d_bytes, align);
    const uint64_t wus_off = off;     off = cutlass_moe_align_up(off + gu_sf_bytes, align);
    const uint64_t wdd_off = off;     off = cutlass_moe_align_up(off + dn_d_bytes, align);
    const uint64_t wds_off = off;     off = cutlass_moe_align_up(off + dn_sf_bytes, align);
    const uint64_t total_scratch = off;

    uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(total_scratch, "routed_moe 2bit cutlass");
    if (!scratch) return 0;
    uint32_t *counts = (uint32_t *)(scratch + counts_off);
    uint32_t *offsets = (uint32_t *)(scratch + offsets_off);
    uint32_t *cursors = (uint32_t *)(scratch + cursors_off);
    uint32_t *sorted_pairs = (uint32_t *)(scratch + sorted_off);
    float *x_gathered = (float *)(scratch + gx_off);
    float *w_gathered = (float *)(scratch + gw_off);
    float *ffn_out = (float *)(scratch + fo_off);
    uint8_t *ffn_scratch = scratch + fs_off;
    float *Wtmp = (float *)(scratch + wt_off);
    uint8_t *Wg_d = scratch + wgd_off, *Wg_sf = scratch + wgs_off;
    uint8_t *Wu_d = scratch + wud_off, *Wu_sf = scratch + wus_off;
    uint8_t *Wd_d = scratch + wdd_off, *Wd_sf = scratch + wds_off;

    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    int ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "moe_2bit counts clear");
    if (ok) { moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(counts, selected_ptr, pair_count);
              ok = cuda_ok(cudaGetLastError(), "moe_2bit count"); }
    if (ok) { moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, n_total_expert);
              ok = cuda_ok(cudaGetLastError(), "moe_2bit prefix"); }
    if (ok) { moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(sorted_pairs, cursors, selected_ptr, pair_count);
              ok = cuda_ok(cudaGetLastError(), "moe_2bit scatter"); }
    if (!ok) return 0;

    std::vector<uint32_t> h_offsets((size_t)n_total_expert + 1);
    if (!cuda_ok(cudaMemcpy(h_offsets.data(), offsets, offsets_bytes, cudaMemcpyDeviceToHost),
                 "moe_2bit offsets readback")) return 0;

    const int gu_thr = 256, dn_thr = 256;
    const int gu_blocks = (int)(((uint64_t)mid_dim * (in_dim / 256) * 8 + gu_thr - 1) / gu_thr);
    const int dn_blocks = (int)(((uint64_t)out_dim * (mid_dim / 256) * 8 + dn_thr - 1) / dn_thr);

    for (uint32_t e = 0; e < n_total_expert; e++) {
        const uint32_t pair_offset = h_offsets[e];
        const uint32_t count = h_offsets[e + 1] - pair_offset;
        if (count == 0) continue;

        /* dequant + fp4-pack this expert's gate, up (IQ2_XXS) and down (Q2_K). Same legacy
         * default stream serializes each dequant before its pack, so Wtmp reuse is safe. */
        dequant_iq2xxs_to_f32_kernel<<<gu_blocks, gu_thr>>>(
                Wtmp, (const cuda_block_iq2_xxs *)(gate_w + (uint64_t)e * gate_stride), (int)mid_dim, (int)in_dim);
        ds4_cutlass_pack_weight_f32(Wg_d, Wg_sf, Wtmp, (int)mid_dim, (int)in_dim);
        dequant_iq2xxs_to_f32_kernel<<<gu_blocks, gu_thr>>>(
                Wtmp, (const cuda_block_iq2_xxs *)(up_w + (uint64_t)e * gate_stride), (int)mid_dim, (int)in_dim);
        ds4_cutlass_pack_weight_f32(Wu_d, Wu_sf, Wtmp, (int)mid_dim, (int)in_dim);
        dequant_q2k_to_f32_kernel<<<dn_blocks, dn_thr>>>(
                Wtmp, (const cuda_block_q2_K *)(down_w + (uint64_t)e * down_stride), (int)out_dim, (int)mid_dim);
        ds4_cutlass_pack_weight_f32(Wd_d, Wd_sf, Wtmp, (int)out_dim, (int)mid_dim);
        if (!cuda_ok(cudaGetLastError(), "moe_2bit dequant/pack")) return 0;

        moe_cutlass_gather_kernel<<<count, 256>>>(x_gathered, w_gathered,
                (const float *)x->ptr, (const float *)weights->ptr,
                sorted_pairs, pair_offset, count, n_expert, in_dim);
        if (!cuda_ok(cudaGetLastError(), "moe_2bit gather")) return 0;

        const int rc = ds4_cutlass_expert_ffn_scratch(ffn_out, x_gathered,
                Wg_d, Wg_sf, Wu_d, Wu_sf, Wd_d, Wd_sf, w_gathered, clamp,
                (int)count, (int)in_dim, (int)mid_dim, (int)out_dim, ffn_scratch, ffn_scratch_bytes);
        if (rc != 0) return 0;

        moe_cutlass_scatter_kernel<<<count, 256>>>((float *)down->ptr, ffn_out,
                sorted_pairs, pair_offset, count, out_dim);
        if (!cuda_ok(cudaGetLastError(), "moe_2bit scatter")) return 0;
    }

    const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
    moe_sum_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
            (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    return cuda_ok(cudaGetLastError(), "moe_2bit sum");
}

/* =====================================================================================
 * Fused-dequant int8 tensor-core (IMMA) MoE prefill: read 2-bit weights directly, dequant
 * in registers, mma.sync against the already-int8 Q8_K activations -- no global repack.
 * IQ2_XXS gate/up (2.81x dp4a) + Q2_K down (3.68x dp4a), both bit-exact vs dp4a (validated
 * standalone). Tiling: 16 M-rows x 64 N-cols/block, 8 warps x 8 cols. Requires N%64==0,
 * K%256==0; M (token count) arbitrary (guarded). A operand = cuda_block_q8_K (qs/d/bsums).
 * ===================================================================================== */

/* IQ2_XXS gate/up, GROUPED: one launch covers all experts. blockIdx.y indexes a global
 * M-tile; tile arrays give its expert, row base (into the expert-contiguous gathered acts),
 * and row count. C/Xq are indexed by absolute gathered row. One m16n8k32 MMA per 32-sub-block. */
__global__ static void moe_iq2_mma_grouped(const cuda_block_q8_K *__restrict__ Xq,
                                           const uint8_t *__restrict__ W_all, uint64_t W_stride,
                                           const int *__restrict__ tE, const int *__restrict__ tRow,
                                           const int *__restrict__ tCnt,
                                           float *__restrict__ C, int N, int K) {
    const int NB = K / 256;
    const int mtidx = blockIdx.y;
    const int M = tRow[mtidx] + tCnt[mtidx];          /* absolute row upper bound */
    const int mt = tRow[mtidx];
    const cuda_block_iq2_xxs *W = (const cuda_block_iq2_xxs *)(W_all + (uint64_t)tE[mtidx] * W_stride);
    const int nb = blockIdx.x * 64;
    const int warp = threadIdx.y, lane = threadIdx.x;
    const int gid = lane >> 2, tig = lane & 3;
    const int ncol0 = nb + warp * 8;
    __shared__ int8_t sA[16 * 256];
    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    const int row_a = mt + gid, row_b = mt + gid + 8;
    const int col0 = ncol0 + tig * 2, col1 = ncol0 + tig * 2 + 1;
    const int oa = tig >> 1, ob = oa + 2, boff = (tig & 1) * 4;
    const int wn = ncol0 + gid;
    for (int kb = 0; kb < NB; ++kb) {
        const int tid = threadIdx.y * 32 + threadIdx.x, nthreads = 32 * 8;
        for (int i = tid; i < 16 * 256; i += nthreads) {
            const int r = i >> 8, c = i & 255;
            sA[i] = (mt + r < M) ? Xq[(size_t)(mt + r) * NB + kb].qs[c] : (int8_t)0;
        }
        __syncthreads();
        const cuda_block_iq2_xxs wblk = W[(size_t)wn * NB + kb];
        const float wd_own = dev_f16_to_f32(wblk.d);
        int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const uint32_t aux0 = (uint32_t)wblk.qs[s * 4 + 0] | ((uint32_t)wblk.qs[s * 4 + 1] << 16);
            const uint32_t aux1 = (uint32_t)wblk.qs[s * 4 + 2] | ((uint32_t)wblk.qs[s * 4 + 3] << 16);
            const int ls_own = 2 * (int)(aux1 >> 28) + 1;
            int32_t wa0, wa1, wb0, wb1;
            dev_iq2_i8x8_lut(cuda_iq2xxs_grid, cuda_ksigns_iq2xs,
                             (uint8_t)((aux0 >> (8 * oa)) & 0xff), (aux1 >> (7 * oa)) & 127u, &wa0, &wa1);
            dev_iq2_i8x8_lut(cuda_iq2xxs_grid, cuda_ksigns_iq2xs,
                             (uint8_t)((aux0 >> (8 * ob)) & 0xff), (aux1 >> (7 * ob)) & 127u, &wb0, &wb1);
            const int32_t b0 = (boff == 0) ? wa0 : wa1;
            const int32_t b1 = (boff == 0) ? wb0 : wb1;
            const int base = s * 32 + tig * 4;
            const int32_t a0 = *(const int32_t *)&sA[gid * 256 + base];
            const int32_t a1 = *(const int32_t *)&sA[(gid + 8) * 256 + base];
            const int32_t a2 = *(const int32_t *)&sA[gid * 256 + base + 16];
            const int32_t a3 = *(const int32_t *)&sA[(gid + 8) * 256 + base + 16];
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "r"(0), "r"(0), "r"(0), "r"(0));
            const int ls0 = __shfl_sync(0xffffffffu, ls_own, (tig * 2) * 4);
            const int ls1 = __shfl_sync(0xffffffffu, ls_own, (tig * 2 + 1) * 4);
            acc0 += d0 * ls0; acc1 += d1 * ls1; acc2 += d2 * ls0; acc3 += d3 * ls1;
        }
        const float wd0 = __shfl_sync(0xffffffffu, wd_own, (tig * 2) * 4);
        const float wd1 = __shfl_sync(0xffffffffu, wd_own, (tig * 2 + 1) * 4);
        const float ada = (row_a < M) ? Xq[(size_t)row_a * NB + kb].d : 0.f;
        const float adb = (row_b < M) ? Xq[(size_t)row_b * NB + kb].d : 0.f;
        f0 += 0.125f * ada * wd0 * (float)acc0; f1 += 0.125f * ada * wd1 * (float)acc1;
        f2 += 0.125f * adb * wd0 * (float)acc2; f3 += 0.125f * adb * wd1 * (float)acc3;
        __syncthreads();
    }
    if (row_a < M) { C[(size_t)row_a * N + col0] = f0; C[(size_t)row_a * N + col1] = f1; }
    if (row_b < M) { C[(size_t)row_b * N + col0] = f2; C[(size_t)row_b * N + col1] = f3; }
}

/* Q2_K down, GROUPED (see moe_iq2_mma_grouped). One m16n8k16 MMA per 16-group, int32
 * scaled by (sc&0xf); the -dmin*summs min term from (sc>>4).bsums folded per 256-block. */
__global__ static void moe_q2k_mma_grouped(const cuda_block_q8_K *__restrict__ Xq,
                                           const uint8_t *__restrict__ W_all, uint64_t W_stride,
                                           const int *__restrict__ tE, const int *__restrict__ tRow,
                                           const int *__restrict__ tCnt,
                                           float *__restrict__ C, int N, int K) {
    const int NB = K / 256;
    const int mtidx = blockIdx.y;
    const int M = tRow[mtidx] + tCnt[mtidx];
    const int mt = tRow[mtidx];
    const cuda_block_q2_K *W = (const cuda_block_q2_K *)(W_all + (uint64_t)tE[mtidx] * W_stride);
    const int nb = blockIdx.x * 64;
    const int warp = threadIdx.y, lane = threadIdx.x;
    const int gid = lane >> 2, tig = lane & 3;
    const int ncol0 = nb + warp * 8;
    __shared__ int8_t sA[16 * 256];
    __shared__ int8_t hs[64 * 16];
    float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    const int row_a = mt + gid, row_b = mt + gid + 8;
    const int col0 = ncol0 + tig * 2, col1 = ncol0 + tig * 2 + 1;
    const int wn = ncol0 + gid;
    const int cib0 = warp * 8 + tig * 2, cib1 = warp * 8 + tig * 2 + 1;
    for (int kb = 0; kb < NB; ++kb) {
        const int tid = threadIdx.y * 32 + threadIdx.x, nthreads = 32 * 8;
        for (int i = tid; i < 16 * 256; i += nthreads) {
            const int r = i >> 8, c = i & 255;
            sA[i] = (mt + r < M) ? Xq[(size_t)(mt + r) * NB + kb].qs[c] : (int8_t)0;
        }
        const cuda_block_q2_K wblk = W[(size_t)wn * NB + kb];
        const float wd_own = dev_f16_to_f32(wblk.d), wdmin_own = dev_f16_to_f32(wblk.dmin);
        if (tig == 0) {
            const int cib = warp * 8 + gid;
            #pragma unroll
            for (int g = 0; g < 16; ++g) hs[cib * 16 + g] = (int8_t)(wblk.scales[g] >> 4);
        }
        int bs_a[16], bs_b[16];
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            bs_a[g] = (row_a < M) ? (int)Xq[(size_t)row_a * NB + kb].bsums[g] : 0;
            bs_b[g] = (row_b < M) ? (int)Xq[(size_t)row_b * NB + kb].bsums[g] : 0;
        }
        __syncthreads();
        int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            const int kk = g >> 3, jj = (g >> 1) & 3, hh = g & 1;
            const int qbase = kk * 32 + hh * 16;
            const int abase = kk * 128 + jj * 32 + hh * 16 + tig * 4;
            uint32_t b0 = 0;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const uint32_t q = ((uint32_t)wblk.qs[qbase + tig * 4 + e] >> (2 * jj)) & 3u;
                b0 |= q << (8 * e);
            }
            const int32_t a0 = *(const int32_t *)&sA[gid * 256 + abase];
            const int32_t a1 = *(const int32_t *)&sA[(gid + 8) * 256 + abase];
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
            const int ls_own = (int)(wblk.scales[g] & 0xf);
            const int ls0 = __shfl_sync(0xffffffffu, ls_own, (tig * 2) * 4);
            const int ls1 = __shfl_sync(0xffffffffu, ls_own, (tig * 2 + 1) * 4);
            acc0 += d0 * ls0; acc1 += d1 * ls1; acc2 += d2 * ls0; acc3 += d3 * ls1;
        }
        const float wd0 = __shfl_sync(0xffffffffu, wd_own, (tig * 2) * 4);
        const float wd1 = __shfl_sync(0xffffffffu, wd_own, (tig * 2 + 1) * 4);
        const float wdm0 = __shfl_sync(0xffffffffu, wdmin_own, (tig * 2) * 4);
        const float wdm1 = __shfl_sync(0xffffffffu, wdmin_own, (tig * 2 + 1) * 4);
        const float ada = (row_a < M) ? Xq[(size_t)row_a * NB + kb].d : 0.f;
        const float adb = (row_b < M) ? Xq[(size_t)row_b * NB + kb].d : 0.f;
        int sma0 = 0, sma1 = 0, smb0 = 0, smb1 = 0;
        #pragma unroll
        for (int g = 0; g < 16; ++g) {
            const int h0 = hs[cib0 * 16 + g], h1 = hs[cib1 * 16 + g];
            sma0 += h0 * bs_a[g]; sma1 += h1 * bs_a[g];
            smb0 += h0 * bs_b[g]; smb1 += h1 * bs_b[g];
        }
        f0 += ada * (wd0 * (float)acc0 - wdm0 * (float)sma0);
        f1 += ada * (wd1 * (float)acc1 - wdm1 * (float)sma1);
        f2 += adb * (wd0 * (float)acc2 - wdm0 * (float)smb0);
        f3 += adb * (wd1 * (float)acc3 - wdm1 * (float)smb1);
        __syncthreads();
    }
    if (row_a < M) { C[(size_t)row_a * N + col0] = f0; C[(size_t)row_a * N + col1] = f1; }
    if (row_b < M) { C[(size_t)row_b * N + col0] = f2; C[(size_t)row_b * N + col1] = f3; }
}

/* SwiGLU + per-token routing weight (matches ds4_mxfp4_cutlass.cu swiglu_kernel). */
__global__ static void moe_mma_swiglu_kernel(float *mid, const float *gate, const float *up,
                                             const float *w, float clamp, int mid_dim, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    if (clamp > 1.0e-6f) { if (g > clamp) g = clamp; if (u > clamp) u = clamp; if (u < -clamp) u = -clamp; }
    const float s = g / (1.f + expf(-g));
    mid[i] = s * u * w[i / mid_dim];
}

/* ---- DS4_MOE_IMMA_DEBUG: reference dp4a recompute over the SAME sorted layout, to bisect
 * whether the IMMA gate/down MMA (vs the block reference dot) is the source of drift. ---- */
__global__ static void moe_ref_gate_dp4a_kernel(float *out, const uint8_t *W_all, uint64_t W_stride,
        const int *rowExpert, const cuda_block_q8_K *xq, int N, int in_nb, uint32_t TP) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)TP * N) return;
    int row = (int)(idx / N), col = (int)(idx % N);
    int e = rowExpert[row];
    const cuda_block_iq2_xxs *gw = (const cuda_block_iq2_xxs *)(W_all + (uint64_t)e * W_stride) + (size_t)col * in_nb;
    const cuda_block_q8_K *xr = xq + (size_t)row * in_nb;
    float acc = 0.f;
    for (int b = 0; b < in_nb; b++) acc += dev_dot_iq2_xxs_q8_K_block(gw + b, xr + b);
    out[idx] = acc;
}
__global__ static void moe_ref_down_dp4a_kernel(float *out, const uint8_t *W_all, uint64_t W_stride,
        const int *rowExpert, const cuda_block_q8_K *xq, int N, int in_nb, uint32_t TP) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)TP * N) return;
    int row = (int)(idx / N), col = (int)(idx % N);
    int e = rowExpert[row];
    const cuda_block_q2_K *gw = (const cuda_block_q2_K *)(W_all + (uint64_t)e * W_stride) + (size_t)col * in_nb;
    const cuda_block_q8_K *xr = xq + (size_t)row * in_nb;
    float acc = 0.f;
    for (int b = 0; b < in_nb; b++) acc += dev_dot_q2_K_q8_K_block(gw + b, xr + b);
    out[idx] = acc;
}

/* IQ2_XXS(gate/up)+Q2_K(down) grouped-expert FFN via fused-dequant IMMA. Mirrors
 * routed_moe_launch_cutlass's sort/gather/scatter, but per expert: q8_K-quantize gathered
 * acts, MMA gate+up, swiglu, q8_K-quantize mid, MMA down. n_tokens>=64 gate is in the caller. */
static int routed_moe_launch_2bit_mma(
        ds4_gpu_tensor *out, ds4_gpu_tensor *down,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_stride, uint64_t down_stride,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t n_tokens) {
    if (!out || !down || !model_map || !selected || !weights || !x || n_tokens == 0 ||
        n_total_expert == 0 || n_expert == 0 ||
        (in_dim % 256u) || (mid_dim % 256u) || (mid_dim % 64u) || (out_dim % 64u)) {
        return 0;
    }
    const uint64_t gate_total_bytes = (uint64_t)n_total_expert * gate_stride;
    const uint64_t down_total_bytes = (uint64_t)n_total_expert * down_stride;
    if (gate_total_bytes > model_size - gate_offset ||
        gate_total_bytes > model_size - up_offset ||
        down_total_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_total_bytes, "moe_imma_gate");
    const char *up_w   = cuda_model_range_ptr(model_map, up_offset, gate_total_bytes, "moe_imma_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_total_bytes, "moe_imma_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint32_t TP = n_tokens * n_expert;           /* total (token,expert) pairs */
    if (TP > 65535u) return 0;                          /* q8_K quant grid.y limit -> dp4a fallback */
    const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
    const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1) * sizeof(uint32_t);
    const uint64_t sorted_bytes = (uint64_t)TP * sizeof(uint32_t);
    const uint64_t in_nb = in_dim / 256, mid_nb = mid_dim / 256;
    const uint64_t max_tiles = (uint64_t)TP / 16 + n_total_expert + 1;
    const uint64_t tiles_bytes = max_tiles * sizeof(int);
    const uint64_t gx_bytes = (uint64_t)TP * in_dim * sizeof(float);
    const uint64_t gw_bytes = (uint64_t)TP * sizeof(float);
    const uint64_t xq_bytes = (uint64_t)TP * in_nb * sizeof(cuda_block_q8_K);
    const uint64_t gate_bytes = (uint64_t)TP * mid_dim * sizeof(float);
    const uint64_t midq_bytes = (uint64_t)TP * mid_nb * sizeof(cuda_block_q8_K);
    const uint64_t fo_bytes = (uint64_t)TP * out_dim * sizeof(float);

    const uint64_t align = 256;
    uint64_t off = 0;
    const uint64_t counts_off = off;  off = cutlass_moe_align_up(off + counts_bytes, align);
    const uint64_t offsets_off = off; off = cutlass_moe_align_up(off + offsets_bytes, align);
    const uint64_t cursors_off = off; off = cutlass_moe_align_up(off + counts_bytes, align);
    const uint64_t sorted_off = off;  off = cutlass_moe_align_up(off + sorted_bytes, align);
    const uint64_t tE_off = off;      off = cutlass_moe_align_up(off + tiles_bytes, align);
    const uint64_t tR_off = off;      off = cutlass_moe_align_up(off + tiles_bytes, align);
    const uint64_t tC_off = off;      off = cutlass_moe_align_up(off + tiles_bytes, align);
    const uint64_t gx_off = off;      off = cutlass_moe_align_up(off + gx_bytes, align);
    const uint64_t gw_off = off;      off = cutlass_moe_align_up(off + gw_bytes, align);
    const uint64_t xq_off = off;      off = cutlass_moe_align_up(off + xq_bytes, align);
    const uint64_t gate_off = off;    off = cutlass_moe_align_up(off + gate_bytes, align);
    const uint64_t up_off = off;      off = cutlass_moe_align_up(off + gate_bytes, align);
    const uint64_t mid_off = off;     off = cutlass_moe_align_up(off + gate_bytes, align);
    const uint64_t midq_off = off;    off = cutlass_moe_align_up(off + midq_bytes, align);
    const uint64_t fo_off = off;      off = cutlass_moe_align_up(off + fo_bytes, align);
    const uint64_t total = off;

    uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(total, "routed_moe 2bit imma");
    if (!scratch) return 0;
    uint32_t *counts = (uint32_t *)(scratch + counts_off);
    uint32_t *offsets = (uint32_t *)(scratch + offsets_off);
    uint32_t *cursors = (uint32_t *)(scratch + cursors_off);
    uint32_t *sorted_pairs = (uint32_t *)(scratch + sorted_off);
    int *d_tE = (int *)(scratch + tE_off);
    int *d_tR = (int *)(scratch + tR_off);
    int *d_tC = (int *)(scratch + tC_off);
    float *x_all = (float *)(scratch + gx_off);
    float *w_all = (float *)(scratch + gw_off);
    cuda_block_q8_K *xq = (cuda_block_q8_K *)(scratch + xq_off);
    float *gate_o = (float *)(scratch + gate_off);
    float *up_o = (float *)(scratch + up_off);
    float *mid_o = (float *)(scratch + mid_off);
    cuda_block_q8_K *midq = (cuda_block_q8_K *)(scratch + midq_off);
    float *ffn_out = (float *)(scratch + fo_off);

    const int32_t *selected_ptr = (const int32_t *)selected->ptr;
    int ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "moe_imma counts clear");
    if (ok) { moe_count_sorted_pairs_kernel<<<(TP + 255u) / 256u, 256>>>(counts, selected_ptr, TP);
              ok = cuda_ok(cudaGetLastError(), "moe_imma count"); }
    if (ok) { moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, n_total_expert);
              ok = cuda_ok(cudaGetLastError(), "moe_imma prefix"); }
    if (ok) { moe_scatter_sorted_pairs_kernel<<<(TP + 255u) / 256u, 256>>>(sorted_pairs, cursors, selected_ptr, TP);
              ok = cuda_ok(cudaGetLastError(), "moe_imma scatter"); }
    if (!ok) return 0;

    /* Host-side tile schedule from the expert offsets (one small readback per layer). */
    std::vector<uint32_t> h_offsets((size_t)n_total_expert + 1);
    if (!cuda_ok(cudaMemcpy(h_offsets.data(), offsets, offsets_bytes, cudaMemcpyDeviceToHost),
                 "moe_imma offsets readback")) return 0;
    std::vector<int> h_tE, h_tR, h_tC;
    h_tE.reserve(max_tiles); h_tR.reserve(max_tiles); h_tC.reserve(max_tiles);
    for (uint32_t e = 0; e < n_total_expert; e++) {
        const uint32_t base = h_offsets[e], cnt = h_offsets[e + 1] - base;
        for (uint32_t t = 0; t * 16u < cnt; t++) {
            h_tE.push_back((int)e);
            h_tR.push_back((int)(base + t * 16u));
            h_tC.push_back((int)((cnt - t * 16u < 16u) ? (cnt - t * 16u) : 16u));
        }
    }
    const int total_mt = (int)h_tE.size();
    if (total_mt == 0) {  /* no routed tokens: zero the output */
        moe_sum_kernel<<<(uint32_t)(((uint64_t)n_tokens * out_dim + 255u) / 256u), 256>>>(
                (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        return cuda_ok(cudaGetLastError(), "moe_imma empty sum");
    }
    if (!cuda_ok(cudaMemcpy(d_tE, h_tE.data(), total_mt * sizeof(int), cudaMemcpyHostToDevice), "moe_imma tE") ||
        !cuda_ok(cudaMemcpy(d_tR, h_tR.data(), total_mt * sizeof(int), cudaMemcpyHostToDevice), "moe_imma tR") ||
        !cuda_ok(cudaMemcpy(d_tC, h_tC.data(), total_mt * sizeof(int), cudaMemcpyHostToDevice), "moe_imma tC")) {
        return 0;
    }

    /* Gather + quantize ALL pairs once, in expert-contiguous (sorted) order. */
    moe_cutlass_gather_kernel<<<TP, 256>>>(x_all, w_all, (const float *)x->ptr, (const float *)weights->ptr,
                                           sorted_pairs, 0, TP, n_expert, in_dim);
    q8_K_quantize_kernel<<<dim3((unsigned)in_nb, TP), 256>>>(xq, x_all, in_dim, TP);
    if (!cuda_ok(cudaGetLastError(), "moe_imma gather/quant")) return 0;

    const dim3 tile_blk(32, 8);
    /* Grouped gate + up (one launch each over all experts' tiles). */
    moe_iq2_mma_grouped<<<dim3(mid_dim / 64, total_mt), tile_blk>>>(
            xq, (const uint8_t *)gate_w, gate_stride, d_tE, d_tR, d_tC, gate_o, (int)mid_dim, (int)in_dim);
    moe_iq2_mma_grouped<<<dim3(mid_dim / 64, total_mt), tile_blk>>>(
            xq, (const uint8_t *)up_w, gate_stride, d_tE, d_tR, d_tC, up_o, (int)mid_dim, (int)in_dim);

    const long sw_n = (long)TP * mid_dim;
    moe_mma_swiglu_kernel<<<(unsigned)((sw_n + 255) / 256), 256>>>(mid_o, gate_o, up_o, w_all, clamp, (int)mid_dim, sw_n);
    q8_K_quantize_kernel<<<dim3((unsigned)mid_nb, TP), 256>>>(midq, mid_o, mid_dim, TP);

    /* Grouped down. */
    moe_q2k_mma_grouped<<<dim3(out_dim / 64, total_mt), tile_blk>>>(
            midq, (const uint8_t *)down_w, down_stride, d_tE, d_tR, d_tC, ffn_out, (int)out_dim, (int)mid_dim);
    if (!cuda_ok(cudaGetLastError(), "moe_imma grouped gemms")) return 0;

    moe_cutlass_scatter_kernel<<<TP, 256>>>((float *)down->ptr, ffn_out, sorted_pairs, 0, TP, out_dim);
    const uint64_t sum_n = (uint64_t)n_tokens * out_dim;
    moe_sum_kernel<<<(uint32_t)((sum_n + 255u) / 256u), 256>>>(
            (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    int imma_ok = cuda_ok(cudaGetLastError(), "moe_imma sum");

    if (imma_ok && getenv("DS4_MOE_IMMA_DEBUG")) {
        /* Bisect: recompute gate + down with the dp4a block reference over the SAME sorted
         * xq/midq/weights and report rms vs the IMMA MMA outputs. Isolates MMA vs the rest. */
        std::vector<int> h_rowExpert((size_t)TP, -1);
        for (int t = 0; t < total_mt; t++)
            for (int r = 0; r < h_tC[t]; r++) h_rowExpert[(size_t)h_tR[t] + r] = h_tE[t];
        int *d_rowExpert = NULL;
        float *g_ref = NULL, *d_ref = NULL;
        cudaMalloc(&d_rowExpert, (size_t)TP * sizeof(int));
        cudaMalloc(&g_ref, (size_t)TP * mid_dim * sizeof(float));
        cudaMalloc(&d_ref, (size_t)TP * out_dim * sizeof(float));
        cudaMemcpy(d_rowExpert, h_rowExpert.data(), (size_t)TP * sizeof(int), cudaMemcpyHostToDevice);
        moe_ref_gate_dp4a_kernel<<<(unsigned)(((uint64_t)TP * mid_dim + 255) / 256), 256>>>(
                g_ref, (const uint8_t *)gate_w, gate_stride, d_rowExpert, xq, (int)mid_dim, (int)in_nb, TP);
        moe_ref_down_dp4a_kernel<<<(unsigned)(((uint64_t)TP * out_dim + 255) / 256), 256>>>(
                d_ref, (const uint8_t *)down_w, down_stride, d_rowExpert, midq, (int)out_dim, (int)mid_nb, TP);
        cudaDeviceSynchronize();
        auto rms = [](const float *a, const float *b, uint64_t n) {
            double ss = 0, rr = 0; for (uint64_t i = 0; i < n; i++) { double d = (double)a[i]-b[i]; ss += d*d; rr += (double)a[i]*a[i]; }
            double r = sqrt(ss/n), ref = sqrt(rr/n); return std::make_pair(r, ref > 0 ? r/ref : 0.0); };
        std::vector<float> hg((size_t)TP*mid_dim), hgr((size_t)TP*mid_dim), hd((size_t)TP*out_dim), hdr((size_t)TP*out_dim);
        cudaMemcpy(hg.data(), gate_o, hg.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hgr.data(), g_ref, hgr.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hd.data(), ffn_out, hd.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hdr.data(), d_ref, hdr.size()*sizeof(float), cudaMemcpyDeviceToHost);
        auto rg = rms(hg.data(), hgr.data(), hg.size());
        auto rd = rms(hd.data(), hdr.data(), hd.size());
        /* swiglu check: recompute mid from gate_o/up_o and compare to mid_o */
        std::vector<float> hup(hg.size()), hmid(hg.size()), hw(TP);
        cudaMemcpy(hup.data(), up_o, hup.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hmid.data(), mid_o, hmid.size()*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hw.data(), w_all, TP*sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<float> hmid_ref(hg.size());
        for (uint64_t i = 0; i < hg.size(); i++) {
            float g = hg[i], u = hup[i];
            if (clamp > 1e-6f) { if (g>clamp) g=clamp; if (u>clamp) u=clamp; if (u<-clamp) u=-clamp; }
            hmid_ref[i] = (g/(1.f+expf(-g)))*u*hw[i/mid_dim];
        }
        auto rs = rms(hmid.data(), hmid_ref.data(), hmid.size());
        /* scatter+sum check: scatter proven-exact ffn_out to a per-token accumulator, compare to out->ptr */
        std::vector<uint32_t> hsp(TP);
        cudaMemcpy(hsp.data(), sorted_pairs, TP*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::vector<float> hout((size_t)n_tokens*out_dim), hout_ref((size_t)n_tokens*out_dim, 0.f);
        cudaMemcpy(hout.data(), out->ptr, hout.size()*sizeof(float), cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < TP; i++) {
            uint32_t tok = hsp[i] / n_expert;
            for (uint32_t k = 0; k < out_dim; k++) hout_ref[(size_t)tok*out_dim+k] += hd[(size_t)i*out_dim+k];
        }
        auto ro = rms(hout.data(), hout_ref.data(), hout.size());
        fprintf(stderr, "IMMA_DEBUG TP=%u mid=%u out=%u  GATE rel=%.3f%%  DOWN rel=%.3f%%  SWIGLU rel=%.3f%%  SUM rel=%.3f%% (rms %.4g)\n",
                TP, mid_dim, out_dim, rg.second*100.0, rd.second*100.0, rs.second*100.0, ro.second*100.0, ro.first);
        cudaFree(d_rowExpert); cudaFree(g_ref); cudaFree(d_ref);
    }

    return imma_ok;
}

static int ds4_moe_imma_prefill_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DS4_MOE_IMMA_PREFILL"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v;
}

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
    const int mxfp4_path = (gate_type == 39u && down_type == 39u);
    if (!mxfp4_path && (gate_type != 16u || down_type != 10u)) return 0;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const uint64_t required_slot_count = (uint64_t)n_tokens * n_expert;
    const int use_stream_selected_cache =
        g_ssd_streaming_mode &&
        g_stream_selected_cache.valid &&
        g_stream_selected_cache.model_map == model_map &&
        g_stream_selected_cache.layer == layer_index &&
        g_stream_selected_cache.n_total_expert == n_total_expert &&
        g_stream_selected_cache.slot_count >= required_slot_count &&
        g_stream_selected_cache.gate_offset == gate_offset &&
        g_stream_selected_cache.up_offset == up_offset &&
        g_stream_selected_cache.down_offset == down_offset &&
        g_stream_selected_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_selected_cache.down_expert_bytes == down_expert_bytes &&
        g_stream_selected_cache.gate_ptr &&
        g_stream_selected_cache.up_ptr &&
        g_stream_selected_cache.down_ptr &&
        g_stream_selected_cache.slot_selected_tensor.ptr &&
        g_stream_selected_cache.slot_selected_tensor.bytes >=
            required_slot_count * sizeof(int32_t);
    const ds4_gpu_tensor *selected_tensor =
        use_stream_selected_cache ? &g_stream_selected_cache.slot_selected_tensor : selected;
    const int32_t *selected_ptr = (const int32_t *)selected_tensor->ptr;
    const char *gate_w = use_stream_selected_cache
        ? g_stream_selected_cache.gate_ptr
        : cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = use_stream_selected_cache
        ? g_stream_selected_cache.up_ptr
        : cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = use_stream_selected_cache
        ? g_stream_selected_cache.down_ptr
        : cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
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
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_sorted_pairs = n_tokens > 1u && !mxfp4_path;
        const uint32_t use_expert_tiles = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL;
        const uint32_t expert_tile_m = getenv("DS4_CUDA_MOE_TILE4") ? 4u : 8u;
        const uint32_t write_gate_up = getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;
        const uint32_t use_p2_sorted = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_P2") == NULL;
        const uint32_t use_atomic_down = use_expert_tiles &&
            getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == NULL &&
            (getenv("DS4_CUDA_MOE_ATOMIC_DOWN") != NULL ||
             n_tokens >= 128u);
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW128") != NULL ||
             (n_tokens >= 128u &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW128") == NULL));
        const uint32_t use_down_tile16 = expert_tile_m == 8u &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_DOWN_TILE16") == NULL &&
            use_atomic_down;
        const uint32_t use_decode_lut_gate =
            !mxfp4_path && n_tokens == 1u && xq_blocks <= 16u &&
            getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL;
        const uint32_t gate_row_span =
            getenv("DS4_CUDA_MOE_GATE_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ? 2048u : 1024u;
        const uint32_t down_row_span =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ? 1024u : 2048u;
        const uint32_t use_down_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW128") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW64") != NULL ||
             (use_down_tile16 &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW128") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW64") == NULL));
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert == 6u &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL;
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
            const uint32_t sort_expert_count =
                use_stream_selected_cache ? g_stream_selected_cache.compact_count :
                n_total_expert;
            if (sort_expert_count == 0) ok = 0;
            const uint64_t counts_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + sort_expert_count;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + sort_expert_count) : 0u;
            const uint64_t tile_offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
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
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
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
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, sort_expert_count, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile_experts, tile_starts, tile_offsets, counts, sort_expert_count, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (mxfp4_path) {
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
                        write_gate_up,
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
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (mxfp4_path) {
                    moe_down_mxfp4_sum6_qwarp32_kernel<<<sgrid, 256>>>(
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
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                if (mxfp4_path) {
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
        if (ok && !use_atomic_down && !use_direct_down_sum6) {
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



extern "C" int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected) {
    (void)selected;
    (void)n_selected;
    return 1;
}



extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index) {
    if (gate_type == 40u && down_type == 40u) {
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


extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
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
                if (cuda_ok(cudaGetLastError(), "moe fp4 gemv sum")) return 1;
            }
            /* any failure: fall through to the grouped CUTLASS path */
        }
        return routed_moe_launch_cutlass(out, down, model_map, model_size,
                                         gate_offset, up_offset, down_offset,
                                         gate_expert_bytes, gate_row_bytes,
                                         down_expert_bytes, down_row_bytes,
                                         expert_in_dim, expert_mid_dim, out_dim,
                                         selected, weights, n_total_expert, n_expert, clamp, x,
                                         n_tokens);
    }
    /* 2-bit ordinary layers (IQ2_XXS gate/up, Q2_K down), large-batch prefill: fused-dequant
     * int8 tensor-core (IMMA) path -- reads 2-bit directly, no repack. Preferred over dp4a and
     * the fp4-repack path. Batch-gated (decode stays dp4a); falls through on any early-out. */
    if (gate_type == 16u && down_type == 10u && n_tokens >= 64u && ds4_moe_imma_prefill_enabled()) {
        int rc = routed_moe_launch_2bit_mma(out, down, model_map, model_size,
                                            gate_offset, up_offset, down_offset,
                                            gate_expert_bytes, down_expert_bytes,
                                            expert_in_dim, expert_mid_dim, out_dim,
                                            selected, weights, n_total_expert, n_expert,
                                            clamp, x, n_tokens);
        if (rc != 0) return rc;
    }
    /* 2-bit ordinary layers (IQ2_XXS gate/up, Q2_K down): for large-batch prefill, route through
     * the CUTLASS fp4 tensor-core path (dequant->fp4-pack per expert). Opt-in + batch-gated so
     * decode/small batches keep the cheaper dp4a path. Falls through to dp4a on any early-out. */
    if (gate_type == 16u && down_type == 10u && n_tokens >= 64u && ds4_moe_fp4_prefill_enabled()) {
        int rc = routed_moe_launch_2bit_cutlass(out, down, model_map, model_size,
                                                gate_offset, up_offset, down_offset,
                                                gate_expert_bytes, down_expert_bytes,
                                                expert_in_dim, expert_mid_dim, out_dim,
                                                selected, weights, n_total_expert, n_expert,
                                                clamp, x, n_tokens);
        if (rc != 0) return rc;
        /* rc==0 => unsupported shape/alloc failure: fall through to the dp4a path below. */
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

