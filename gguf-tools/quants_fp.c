#include "quants_internal.h"

/* ---- MXFP8: E4M3 values + per-32 E8M0 block scale; block = 33 B / 32 elems
 * (8.25 bpw). byte[0] = E8M0 scale (value = 2^(b-127)); byte[1..32] = E4M3.
 * Hardware-native on Blackwell MX tensor cores (cuBLASLt VEC32_UE8M0). For the
 * FP8-source attn/dense/shared weights this is near-lossless and >= Q8_0. ---- */
static uint8_t ds4q_f32_to_e4m3(float x) {
    if (x != x) return 0x7f;                          /* NaN */
    uint8_t s = (x < 0.0f) ? 0x80 : 0x00;
    float a = fabsf(x);
    if (a >= 448.0f) return (uint8_t)(s | 0x7e);      /* saturate (incl inf) */
    if (a == 0.0f)   return s;                        /* signed zero */
    union { float f; uint32_t u; } v; v.f = a;
    int32_t exp = (int32_t)((v.u >> 23) & 0xff) - 127;
    uint32_t sig = (1u << 23) | (v.u & 0x7fffffu);    /* 24-bit significand */
    int32_t e8 = exp + 7;                             /* E4M3 biased exponent */
    if (e8 <= 0) {                                    /* subnormal: code * 2^-9 */
        int code = (int)lrintf(a * 512.0f);
        if (code > 7) return (uint8_t)(s | (1u << 3));   /* round up to min normal */
        return (uint8_t)(s | (code & 0x7));
    }
    uint32_t mant3 = (sig >> 20) & 0x7;
    uint32_t rem = sig & ((1u << 20) - 1), half = 1u << 19;
    if (rem > half || (rem == half && (mant3 & 1))) {     /* round to nearest even */
        if (++mant3 == 8) { mant3 = 0; e8++; }
    }
    if (e8 > 15 || (e8 == 15 && mant3 >= 7)) return (uint8_t)(s | 0x7e); /* saturate */
    return (uint8_t)(s | ((e8 & 0xf) << 3) | (mant3 & 0x7));
}

static float ds4q_e4m3_to_f32(uint8_t b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    uint32_t e = (b >> 3) & 0xf, m = b & 0x7;
    if (e == 0) return sign * (float)m * (1.0f / 512.0f);          /* subnormal */
    if (e == 15 && m == 7) return sign * (float)NAN;
    return sign * (1.0f + (float)m * 0.125f) * ldexpf(1.0f, (int)e - 7);
}

size_t ds4q_quantize_fp8_e4m3(const float *src, void *dst, int64_t start,
                                     int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_FP8_E4M3, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t nblocks = nrows * (ncols / qk);
    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + start + (size_t)b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) { float av = fabsf(x[j]); if (av > amax) amax = av; }
        int scale_exp = -127;
        if (amax > 0.0f) { int e; frexpf(amax, &e); scale_exp = (e - 1) - 7; } /* max elem -> [128,256) */
        if (scale_exp < -127) scale_exp = -127;
        if (scale_exp >  127) scale_exp =  127;
        out[0] = (uint8_t)(scale_exp + 127);            /* E8M0 */
        const float inv = ldexpf(1.0f, -scale_exp);
        for (int j = 0; j < qk; j++) out[1 + j] = ds4q_f32_to_e4m3(x[j] * inv);
        out += 33;
    }
    return (size_t)nrows * row_size;
}

void ds4q_dequantize_fp8_e4m3(const void *blocks, float *out, int64_t n) {
    const uint8_t *b = (const uint8_t *)blocks;
    const int64_t nblocks = n / 32;
    for (int64_t i = 0; i < nblocks; i++) {
        const float scale = ldexpf(1.0f, (int)b[0] - 127);
        for (int j = 0; j < 32; j++) out[i * 32 + j] = ds4q_e4m3_to_f32(b[1 + j]) * scale;
        b += 33;
    }
}

/* ---- MXFP4: E2M1 values + per-32 E8M0 block scale; block = 17 B / 32 elems
 * (4.25 bpw). byte[0] = E8M0 scale (value = 2^(b-127)); byte[1..16] = packed
 * nibbles: byte[1+j] = v[2j] | (v[2j+1] << 4).  E2M1 representable magnitudes
 * are {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign via bit 3.  This format matches
 * DS4_TENSOR_FP4_E2M1 in the runtime. ---- */
static const float ds4q_e2m1_mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

static uint8_t ds4q_f32_to_e2m1(float x) {
    uint8_t s = (x < 0.0f) ? 0x80 : 0x00;
    float a = fabsf(x);
    if (a == 0.0f) return s;
    if (a >= 6.0f) return (uint8_t)(s | 0x07);
    int best = 0;
    float best_diff = fabsf(a - ds4q_e2m1_mag[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(a - ds4q_e2m1_mag[i]);
        if (d < best_diff) { best = i; best_diff = d; }
    }
    return (uint8_t)(s | (best & 0x7));
}

size_t ds4q_quantize_mxfp4(const float *src, void *dst, int64_t start,
                                   int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_MXFP4, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t nblocks = nrows * (ncols / qk);
    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + start + (size_t)b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) { float av = fabsf(x[j]); if (av > amax) amax = av; }
        int scale_exp = -127;
        if (amax > 0.0f) scale_exp = (int)ceilf(log2f(amax / 6.0f));
        if (scale_exp < -127) scale_exp = -127;
        if (scale_exp >  127) scale_exp =  127;
        out[0] = (uint8_t)(scale_exp + 127);
        const float inv = ldexpf(1.0f, -scale_exp);
        for (int j = 0; j < 16; j++) {
            uint8_t lo = ds4q_f32_to_e2m1(x[2*j]     * inv);
            uint8_t hi = ds4q_f32_to_e2m1(x[2*j + 1] * inv);
            out[1 + j] = (lo & 0x0f) | (hi << 4);
        }
        out += 17;
    }
    return (size_t)nrows * row_size;
}

void ds4q_dequantize_mxfp4(const void *blocks, float *out, int64_t n) {
    const uint8_t *b = (const uint8_t *)blocks;
    const int64_t nblocks = n / 32;
    for (int64_t i = 0; i < nblocks; i++) {
        const float scale = ldexpf(1.0f, (int)b[0] - 127);
        for (int j = 0; j < 16; j++) {
            out[i * 32 + 2*j]     = ds4q_e2m1_mag[b[1 + j] & 0x0f] * scale;
            out[i * 32 + 2*j + 1] = ds4q_e2m1_mag[b[1 + j] >> 4]   * scale;
        }
        b += 17;
    }
}
