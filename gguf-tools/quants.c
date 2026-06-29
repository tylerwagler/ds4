/*
 * DS4 quantization facade.
 *
 * These are the small GGUF quantization pieces needed by our DeepSeek V4
 * Flash recipes: float conversion, q8_0, q2_K, q4_K, and iq2_xxs.  The code is
 * local C and deliberately narrow; other GGUF type IDs are named for metadata
 * compatibility, but cannot be emitted by this tool.
 *
 * The quantized block layouts and search procedures are derived from the
 * MIT-licensed GGML/llama.cpp quantizers.  Keep changes conservative: byte
 * layout compatibility is more important here than generality.
 */

#include "quants.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static pthread_mutex_t ds4q_init_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const char *name;
    int64_t block_size;
    size_t type_size;
    bool can_quantize;
    bool requires_imatrix;
} ds4q_traits;

#define QK_K 256
#define DS4Q_GROUP_MAX_EPS 1e-15f
#define DS4Q_MIN(a, b) ((a) < (b) ? (a) : (b))
#define DS4Q_MAX(a, b) ((a) > (b) ? (a) : (b))

static const ds4q_traits ds4q_type_traits[DS4Q_TYPE_COUNT] = {
    [DS4Q_TYPE_F32]     = { "f32",      1,   4, false, false },
    [DS4Q_TYPE_F16]     = { "f16",      1,   2, false, false },
    [DS4Q_TYPE_Q4_0]    = { "q4_0",    32,  18, false, false },
    [DS4Q_TYPE_Q4_1]    = { "q4_1",    32,  20, false, false },
    [DS4Q_TYPE_Q5_0]    = { "q5_0",    32,  22, false, false },
    [DS4Q_TYPE_Q5_1]    = { "q5_1",    32,  24, false, false },
    [DS4Q_TYPE_Q8_0]    = { "q8_0",    32,  34, true,  false },
    [DS4Q_TYPE_Q8_1]    = { "q8_1",    32,  36, false, false },
    [DS4Q_TYPE_Q2_K]    = { "q2_K",  QK_K,  84, true,  false },
    [DS4Q_TYPE_Q3_K]    = { "q3_K",  QK_K, 110, true,  false },
    [DS4Q_TYPE_Q4_K]    = { "q4_K",  QK_K, 144, true,  false },
    [DS4Q_TYPE_Q5_K]    = { "q5_K",  QK_K, 176, true,  false },
    [DS4Q_TYPE_Q6_K]    = { "q6_K",  QK_K, 210, true,  false },
    [DS4Q_TYPE_Q8_K]    = { "q8_K",  QK_K, 292, false, false },
    [DS4Q_TYPE_FP8_E4M3]= { "fp8_e4m3", 32,  33, true,  false },
    [DS4Q_TYPE_IQ2_XXS] = { "iq2_xxs", QK_K,  66, true,  true  },
    [DS4Q_TYPE_IQ2_XS]  = { "iq2_xs",  QK_K,  74, false, true  },
    [DS4Q_TYPE_IQ3_XXS] = { "iq3_xxs", QK_K,  98, false, false },
    [DS4Q_TYPE_IQ1_S]   = { "iq1_s",   QK_K,  50, false, true  },
    [DS4Q_TYPE_IQ4_NL]  = { "iq4_nl",     32,  18, false, false },
    [DS4Q_TYPE_IQ3_S]   = { "iq3_s",   QK_K, 110, false, false },
    [DS4Q_TYPE_IQ2_S]   = { "iq2_s",   QK_K,  82, false, false },
    [DS4Q_TYPE_IQ4_XS]  = { "iq4_xs",  QK_K, 136, false, false },
    [DS4Q_TYPE_I8]      = { "i8",          1,   1, false, false },
    [DS4Q_TYPE_I16]     = { "i16",         1,   2, false, false },
    [DS4Q_TYPE_I32]     = { "i32",         1,   4, false, false },
    [DS4Q_TYPE_I64]     = { "i64",         1,   8, false, false },
    [DS4Q_TYPE_F64]     = { "f64",         1,   8, false, false },
    [DS4Q_TYPE_IQ1_M]   = { "iq1_m",   QK_K,  56, false, false },
    [DS4Q_TYPE_BF16]    = { "bf16",        1,   2, false, false },
    [DS4Q_TYPE_TQ1_0]   = { "tq1_0",   QK_K,  54, false, false },
    [DS4Q_TYPE_TQ2_0]   = { "tq2_0",   QK_K,  66, false, false },
    [DS4Q_TYPE_MXFP4]   = { "mxfp4",      32,  17, false, false },
    [DS4Q_TYPE_NVFP4]   = { "nvfp4",      64,  36, false, false },
    [DS4Q_TYPE_Q1_0]    = { "q1_0",      128,  18, false, false },
};

static float ds4q_f32_from_bits(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v = { .u = bits };
    return v.f;
}

static uint32_t ds4q_f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };
    return v.u;
}

static uint16_t ds4q_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = ds4q_f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);

    base = ds4q_f32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t out = ds4q_f32_to_bits(base);
    const uint32_t exp_bits = (out >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = out & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

static int ds4q_nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static float ds4q_make_qkx2_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                memcpy(L, Laux, (size_t)n);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float ds4q_make_qkx3_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0] * x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i] * x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max <= min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i] * x[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights ? weights[i] : x[i] * x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                memcpy(L, Laux, (size_t)n);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float ds4q_make_qp_quants(int n, int nmax, const float *x, uint8_t *L, const float *quant_weights) {
    float max = 0;
    for (int i = 0; i < n; i++) max = DS4Q_MAX(max, x[i]);
    if (max < DS4Q_GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; i++) L[i] = ds4q_nearest_int(iscale * x[i]);
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - scale * L[i];
        best_mse += quant_weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float iscale_is = (0.1f * is + nmax) / max;
        float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale_is * x[i]);
            l = DS4Q_MIN(nmax, l);
            float diff = x[i] - scale_is * l;
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * x[i]);
        l = DS4Q_MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int n_changed = 0;
        for (int i = 0; i < n; i++) {
            float w = quant_weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = ds4q_nearest_int(x[i] * sl2 / slx);
                new_l = DS4Q_MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed++;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static size_t ds4q_quantize_q8_0(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q8_0, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t nblocks = nrows * (ncols / qk);

    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + start + (size_t)b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float av = fabsf(x[j]);
            if (av > amax) amax = av;
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = ds4q_f32_to_f16(d);
        memcpy(out, &hd, sizeof(hd));

        int8_t *qs = (int8_t *)(out + sizeof(hd));
        for (int j = 0; j < qk; j++) qs[j] = (int8_t)roundf(x[j] * id);
        out += sizeof(hd) + qk;
    }
    return (size_t)nrows * row_size;
}

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

static size_t ds4q_quantize_fp8_e4m3(const float *src, void *dst, int64_t start,
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

static void ds4q_write_q4_k_block_ref(const float *x, uint8_t *y) {
    enum { scales_off = 4, qs_off = 16 };
    uint8_t L[QK_K];
    uint8_t Laux[32];
    float weights[32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; j++) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; l++) sum_x2 += x[32 * j + l] * x[32 * j + l];
        float av_x = sqrtf(sum_x2 / 32);
        for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(x[32 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                           &mins[j], Laux, -1.0f, 0.1f, 20, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t ls = ds4q_nearest_int(inv_scale * scales[j]);
        uint8_t lm = ds4q_nearest_int(inv_min * mins[j]);
        ls = DS4Q_MIN(63, ls);
        lm = DS4Q_MIN(63, lm);
        if (j < 4) {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= ((ls >> 4) << 6);
            scales_out[j - 0] |= ((lm >> 4) << 6);
        }
    }

    uint16_t d = ds4q_f32_to_f16(max_scale / 63.0f);
    uint16_t dmin = ds4q_f32_to_f16(max_min / 63.0f);
    memcpy(y + 0, &d, sizeof(d));
    memcpy(y + 2, &dmin, sizeof(dmin));

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; j++) {
        ds4q_get_scale_min_k4(j, scales_out, &sc, &m);
        const float dd = ds4q_f16_to_f32(d) * sc;
        if (!dd) continue;
        const float dm = ds4q_f16_to_f32(dmin) * m;
        for (int ii = 0; ii < 32; ii++) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / dd);
            l = DS4Q_MAX(0, DS4Q_MIN(15, l));
            L[32 * j + ii] = l;
        }
    }

    uint8_t *q = qs_out;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
        q += 32;
    }
}

static void ds4q_write_q4_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights) {
    enum { scales_off = 4, qs_off = 16 };
    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float weights[32];
    float sw[QK_K / 32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; l++) sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; j++) {
        if (quant_weights) {
            const float *qw = quant_weights + 32 * j;
            for (int l = 0; l < 32; l++) weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
        } else {
            for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(x[32 * j + l]);
        }
        float sumw = 0;
        for (int l = 0; l < 32; l++) sumw += weights[l];
        sw[j] = sumw;
        scales[j] = ds4q_make_qkx3_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                           &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = ds4q_make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = ds4q_make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t ls = Ls[j];
        uint8_t lm = Lm[j];
        if (j < 4) {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= ((ls >> 4) << 6);
            scales_out[j - 0] |= ((lm >> 4) << 6);
        }
    }

    uint16_t d = ds4q_f32_to_f16(d_block);
    uint16_t dmin = ds4q_f32_to_f16(m_block);
    memcpy(y + 0, &d, sizeof(d));
    memcpy(y + 2, &dmin, sizeof(dmin));

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; j++) {
        ds4q_get_scale_min_k4(j, scales_out, &sc, &m);
        const float dd = ds4q_f16_to_f32(d) * sc;
        if (!dd) continue;
        const float dm = ds4q_f16_to_f32(dmin) * m;
        for (int ii = 0; ii < 32; ii++) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / dd);
            l = DS4Q_MAX(0, DS4Q_MIN(15, l));
            L[32 * j + ii] = l;
        }
    }

    uint8_t *q = qs_out;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
        q += 32;
    }
}

static size_t ds4q_quantize_q4_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q4_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q4_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q4_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q4_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_q2_k_block_ref(const float *x, uint8_t *y) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    const float q4scale = 15.0f;
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float weights[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; j++) {
        for (int l = 0; l < 16; l++) weights[l] = fabsf(x[16 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j,
                                           &mins[j], Laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    uint16_t hd, hmin;
    if (max_scale > 0) {
        float iscale = q4scale / max_scale;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] = ds4q_nearest_int(iscale * scales[j]);
        hd = ds4q_f32_to_f16(max_scale / q4scale);
    } else {
        memset(scales_out, 0, QK_K / 16);
        hd = ds4q_f32_to_f16(0.0f);
    }
    if (max_min > 0) {
        float iscale = q4scale / max_min;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] |= ds4q_nearest_int(iscale * mins[j]) << 4;
        hmin = ds4q_f32_to_f16(max_min / q4scale);
    } else {
        hmin = ds4q_f32_to_f16(0.0f);
    }
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = ds4q_f16_to_f32(hd) * (scales_out[j] & 0xF);
        if (!d) continue;
        const float dm = ds4q_f16_to_f32(hmin) * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + dm) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static void ds4q_write_q2_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    float sw[QK_K / 16];
    float weight[16];
    uint8_t Ls[QK_K / 16], Lm[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    memset(sw, 0, sizeof(sw));
    float sumx2 = 0;
    for (int j = 0; j < QK_K; j++) sumx2 += x[j] * x[j];
    float sigma2 = sumx2 / QK_K;
    for (int j = 0; j < QK_K / 16; j++) {
        const float *qw = quant_weights + 16 * j;
        for (int l = 0; l < 16; l++) weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
        for (int l = 0; l < QK_K / 16; l++) sw[j] += weight[l];
        scales[j] = ds4q_make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j,
                                           &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float dm = ds4q_make_qp_quants(QK_K / 16, 15, scales, Ls, sw);
    float mm = ds4q_make_qp_quants(QK_K / 16, 15, mins, Lm, sw);
    uint16_t hd = ds4q_f32_to_f16(dm);
    uint16_t hmin = ds4q_f32_to_f16(mm);
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));
    dm = ds4q_f16_to_f32(hd);
    mm = ds4q_f16_to_f32(hmin);

    for (int j = 0; j < QK_K / 16; j++) scales_out[j] = Ls[j] | (Lm[j] << 4);

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = dm * (scales_out[j] & 0xF);
        if (!d) continue;
        const float m = mm * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + m) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static size_t ds4q_quantize_q2_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q2_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q2_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q2_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q2_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

typedef struct {
    uint64_t *grid;
    int *map;
    uint16_t *neighbours;
} ds4q_iq2_data;

static ds4q_iq2_data ds4q_iq2_xxs_data;

static int ds4q_iq2_compare_func(const void *left, const void *right) {
    const int *l = (const int *)left;
    const int *r = (const int *)right;
    return l[0] < r[0] ? -1 :
           l[0] > r[0] ?  1 :
           l[1] < r[1] ? -1 :
           l[1] > r[1] ?  1 : 0;
}

/*
 * IQ2_XXS quantizes a 256-value row block as eight 32-value groups.  Each
 * group stores four 8-value grid indices plus four 7-bit sign masks; the
 * single f16 block scale is refined by 4-bit per-group scale nibbles.
 *
 * The grid is tiny, but not every possible 2-bit 8-tuple is allowed.  During
 * initialization we build the direct map for allowed tuples and a nearest-grid
 * list for the missing ones, matching the GGML search exactly.
 */
static void ds4q_iq2_xxs_init(void) {
    if (ds4q_iq2_xxs_data.grid) return;
    pthread_mutex_lock(&ds4q_init_mutex);
    if (ds4q_iq2_xxs_data.grid) {
        pthread_mutex_unlock(&ds4q_init_mutex);
        return;
    }

    enum { grid_size = 256, map_size = 43692, neighbour_shells = 2 };
    static const uint16_t kgrid[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };

    uint64_t *grid = malloc((size_t)grid_size * sizeof(grid[0]));
    int *map = malloc((size_t)map_size * sizeof(map[0]));
    int *dist2 = malloc((size_t)2 * grid_size * sizeof(dist2[0]));
    assert(grid && map && dist2);

    for (int k = 0; k < grid_size; k++) {
        int8_t *pos = (int8_t *)(grid + k);
        for (int i = 0; i < 8; i++) {
            int l = (kgrid[k] >> (2 * i)) & 3;
            pos[i] = 2 * l + 1;
        }
    }

    for (int i = 0; i < map_size; i++) map[i] = -1;
    for (int i = 0; i < grid_size; i++) map[kgrid[i]] = i;

    int8_t pos[8];
    int num_neighbors = 0;
    int num_not_in_map = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        num_not_in_map++;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        int d2 = dist2[0], have = 1;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            num_neighbors++;
        }
    }

    uint16_t *neighbours = malloc((size_t)(num_neighbors + num_not_in_map) * sizeof(neighbours[0]));
    assert(neighbours);
    int counter = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        map[i] = -(counter + 1);
        int d2 = dist2[0], have = 1;
        uint16_t *start = &neighbours[counter++];
        int n = 0;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            neighbours[counter++] = (uint16_t)dist2[2 * j + 1];
            n++;
        }
        *start = (uint16_t)n;
    }

    free(dist2);
    ds4q_iq2_xxs_data.map = map;
    ds4q_iq2_xxs_data.neighbours = neighbours;
    ds4q_iq2_xxs_data.grid = grid;
    pthread_mutex_unlock(&ds4q_init_mutex);
}

static int ds4q_iq2_find_best_neighbour(const uint16_t *neighbours, const uint64_t *grid,
                                        const float *xval, const float *weight,
                                        float scale, uint8_t *L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; j++) {
        const int8_t *pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; i++) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t *pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)((pg[i] - 1) / 2);
    return grid_index;
}

static void ds4q_write_iq2_xxs_block(const float *x, uint8_t *y, const float *quant_weights) {
    enum { d_off = 0, qs_off = 2, block_size = 32, k_max_q = 3 };
    assert(quant_weights);

    uint32_t q2[2 * (QK_K / block_size)];
    float scales[QK_K / block_size];
    float weight[block_size];
    float xval[block_size];
    uint8_t L[block_size];
    uint8_t Laux[block_size];
    float waux[block_size];
    uint8_t block_signs[4];

    uint16_t hd = ds4q_f32_to_f16(0.0f);
    memcpy(y + d_off, &hd, sizeof(hd));
    memset(q2, 0, sizeof(q2));

    const uint64_t *grid = ds4q_iq2_xxs_data.grid;
    const int *map = ds4q_iq2_xxs_data.map;
    const uint16_t *neighbours = ds4q_iq2_xxs_data.neighbours;
    assert(grid && map && neighbours);

    float sumx2 = 0;
    for (int i = 0; i < QK_K; i++) sumx2 += x[i] * x[i];
    float sigma2 = sumx2 / QK_K;
    float max_scale = 0;

    for (int ib = 0; ib < QK_K / block_size; ib++) {
        const float *xb = x + block_size * ib;
        const float *qw = quant_weights + block_size * ib;
        for (int i = 0; i < block_size; i++) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; k++) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; i++) {
                float v = xb[8 * k + i];
                if (v >= 0) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    nflip++;
                    s |= (uint8_t)(1u << i);
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for (int i = 1; i < 8; i++) {
                    float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < min) {
                        min = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (uint8_t)(1u << imin);
            }
            block_signs[k] = s & 127;
        }

        float max = xval[0];
        for (int i = 1; i < block_size; i++) max = DS4Q_MAX(max, xval[i]);
        if (max < DS4Q_GROUP_MAX_EPS) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float scale = ds4q_make_qp_quants(block_size, k_max_q + 1, xval, L, weight);
        float eff_max = scale * k_max_q;
        if (eff_max <= 0) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float best = 0;
        for (int is = -6; is <= 6; is++) {
            float id = (2 * k_max_q - 1 + is * 0.1f) / eff_max;
            float this_scale = 1 / id;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    Laux[8 * k + i] = (uint8_t)l;
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k, waux + 8 * k,
                                                 this_scale, Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                memcpy(L, Laux, sizeof(L));
            }
        }

        if (scale > 0) {
            float id = 1 / scale;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k,
                                                              waux + 8 * k, scale, L + 8 * k);
                }
                const int8_t *pg = (const int8_t *)(grid + grid_index);
                for (int i = 0; i < 8; i++) L[8 * k + i] = (uint8_t)((pg[i] - 1) / 2);
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; k++) block_signs[k] = (~block_signs[k]) & 127;
        }

        for (int k = 0; k < 4; k++) {
            uint16_t u = 0;
            for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            int grid_index = map[u];
            assert(grid_index >= 0);
            q2[2 * ib + 0] |= (uint32_t)grid_index << (8 * k);
            q2[2 * ib + 1] |= (uint32_t)block_signs[k] << (7 * k);
        }
        assert(scale >= 0);
        scales[ib] = scale;
        max_scale = DS4Q_MAX(max_scale, scale);
    }

    if (!max_scale) {
        memset(y + qs_off, 0, QK_K / 4);
        return;
    }

    float d = max_scale / 31;
    hd = ds4q_f32_to_f16(d);
    memcpy(y + d_off, &hd, sizeof(hd));
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ib++) {
        int l = ds4q_nearest_int(0.5f * (id * scales[ib] - 1));
        l = DS4Q_MAX(0, DS4Q_MIN(15, l));
        q2[2 * ib + 1] |= (uint32_t)l << 28;
    }
    memcpy(y + qs_off, q2, QK_K / 4);
}

static size_t ds4q_quantize_iq2_xxs(const float *src, void *dst, int64_t start,
                                    int64_t nrows, int64_t ncols, const float *quant_weights) {
    assert(quant_weights);
    ds4q_iq2_xxs_init();
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_IQ2_XXS, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_IQ2_XXS].type_size;
            ds4q_write_iq2_xxs_block(xrow + (size_t)b * QK_K, block,
                                     quant_weights + (size_t)b * QK_K);
        }
    }
    return (size_t)nrows * row_size;
}

const char *ds4q_type_name(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return NULL;
    return ds4q_type_traits[type].name;
}

bool ds4q_can_quantize(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].can_quantize;
}

int64_t ds4q_block_size(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    return ds4q_type_traits[type].block_size;
}

size_t ds4q_row_size(ds4q_type type, int64_t ne) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    const ds4q_traits *tr = &ds4q_type_traits[type];
    if (tr->block_size <= 0 || tr->type_size == 0 || ne % tr->block_size != 0) return 0;
    return tr->type_size * (size_t)(ne / tr->block_size);
}

bool ds4q_requires_imatrix(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].requires_imatrix;
}

/* ---- Q3_K (ported from ggml; layout MUST match ds4's vec_dot_q3_K) ----
 * block_q3_K (110 B / 256 elems): hmask[32] high bit, qs[64] low 2 bits,
 * scales[12] (16 signed 6-bit), d (f16). */
enum { Q3K_hmask_off = 0, Q3K_qs_off = 32, Q3K_scales_off = 96, Q3K_d_off = 108 };

/* RMSE-optimal signed quant of n values to [-nmax, nmax-1], weighted (ggml
 * make_qx_quants). Returns block scale; L[i] in [0, 2*nmax-1] (offset +nmax). */
static float ds4q_make_qx_quants(int n, int nmax, const float *x, int8_t *L,
                                 const float *weights) {
    float max = 0, amax = 0;
    for (int i = 0; i < n; i++) { float ax = fabsf(x[i]); if (ax > amax) { amax = ax; max = x[i]; } }
    if (amax < 1e-30f) { for (int i = 0; i < n; i++) L[i] = (int8_t)nmax; return 0.f; }
    float iscale = -nmax / max, bscale = iscale, best = 0;
    for (int is = -9; is <= 9; is++) {
        float isc = -(nmax + 0.1f * is) / max, sumlx = 0, suml2 = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(isc * x[i]); l = DS4Q_MAX(-nmax, DS4Q_MIN(nmax - 1, l));
            float w = weights ? weights[i] : x[i] * x[i];
            sumlx += w * x[i] * l; suml2 += w * (float)l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) { bscale = sumlx / suml2; best = bscale * sumlx; iscale = isc; }
    }
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * x[i]); L[i] = (int8_t)(DS4Q_MAX(-nmax, DS4Q_MIN(nmax - 1, l)) + nmax);
    }
    return bscale;
}

static size_t ds4q_quantize_q3_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q3_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t bpr = ncols / QK_K;
    int8_t L[QK_K]; float scales[QK_K / 16], weight[16];
    for (int64_t r = 0; r < nrows; r++) {
        const float *xrow = src + start + (size_t)r * (size_t)ncols;
        for (int64_t b = 0; b < bpr; b++) {
            const float *x = xrow + (size_t)b * QK_K;
            uint8_t *y = out + (size_t)r * row_size + (size_t)b * 110;
            uint8_t *hmask = y + Q3K_hmask_off, *qs = y + Q3K_qs_off, *sc = y + Q3K_scales_off;
            float sumx2 = 0; for (int j = 0; j < QK_K; j++) sumx2 += x[j] * x[j];
            float sigma2 = 2.f * sumx2 / QK_K;
            for (int j = 0; j < QK_K / 16; j++) {
                if (quant_weights) {
                    const float *qw = quant_weights + (size_t)b * QK_K + 16 * j;
                    for (int l = 0; l < 16; l++) weight[l] = qw[l] * sqrtf(sigma2 + x[16*j+l]*x[16*j+l]);
                } else for (int l = 0; l < 16; l++) weight[l] = x[16*j+l] * x[16*j+l];
                scales[j] = ds4q_make_qx_quants(16, 4, x + 16*j, L + 16*j, weight);
            }
            memset(sc, 0, 12);
            float amax = 0, mscale = 0;
            for (int j = 0; j < 16; j++) if (fabsf(scales[j]) > amax) { amax = fabsf(scales[j]); mscale = scales[j]; }
            uint16_t hd;
            if (amax < 1e-30f) { hd = ds4q_f32_to_f16(0.f); memset(qs, 0, 64); memset(hmask, 0, 32); memcpy(y + Q3K_d_off, &hd, 2); continue; }
            float iscale = -32.f / mscale; int8_t sc6[16];
            for (int j = 0; j < 16; j++) {
                int l = DS4Q_MAX(-32, DS4Q_MIN(31, ds4q_nearest_int(iscale * scales[j])));
                sc6[j] = (int8_t)l; uint8_t u = (uint8_t)(l + 32);
                if (j < 8) sc[j] = u & 0xF; else sc[j - 8] |= (u & 0xF) << 4;
                sc[8 + (j % 4)] |= (uint8_t)(((u >> 4) & 3) << (2 * (j / 4)));
            }
            hd = ds4q_f32_to_f16(1.f / iscale); memcpy(y + Q3K_d_off, &hd, 2);
            float dall = ds4q_f16_to_f32(hd);
            for (int j = 0; j < 16; j++) {
                float d = dall * sc6[j];
                if (d == 0.f) { for (int ii = 0; ii < 16; ii++) L[16*j+ii] = 4; continue; }
                for (int ii = 0; ii < 16; ii++) {
                    int l = DS4Q_MAX(-4, DS4Q_MIN(3, ds4q_nearest_int(x[16*j+ii] / d)));
                    L[16*j+ii] = (int8_t)(l + 4);
                }
            }
            memset(hmask, 0, 32);
            { uint8_t m = 1; int mi = 0;
              for (int j = 0; j < QK_K; j++) { if (L[j] > 3) hmask[mi] |= m; if (++mi == 32) { mi = 0; m <<= 1; } } }
            memset(qs, 0, 64);
            for (int j = 0; j < QK_K; j += 128)
                for (int l = 0; l < 32; l++)
                    qs[j/4 + l] = (uint8_t)((L[j+l]&3) | ((L[j+l+32]&3)<<2) | ((L[j+l+64]&3)<<4) | ((L[j+l+96]&3)<<6));
        }
    }
    return (size_t)nrows * row_size;
}

void ds4q_dequantize_q3_k(const void *blocks, float *out, int64_t n) {
    const uint8_t *p = (const uint8_t *)blocks;
    const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *y = p + (size_t)i * 110, *hm = y + Q3K_hmask_off, *q = y + Q3K_qs_off;
        uint16_t hd; memcpy(&hd, y + Q3K_d_off, 2); float dall = ds4q_f16_to_f32(hd);
        uint32_t aux[4]; memcpy(aux, y + Q3K_scales_off, 12); uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        aux[0] = (aux[0] & km2) | (((tmp >> 0) & km1) << 4);
        aux[1] = (aux[1] & km2) | (((tmp >> 2) & km1) << 4);
        const int8_t *scales = (const int8_t *)aux;
        float *o = out + (size_t)i * QK_K; int is = 0; uint8_t m = 1;
        for (int n0 = 0; n0 < QK_K; n0 += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = dall * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *o++ = dl * (((q[l]>>shift)&3) - ((hm[l]&m)?0:4));
                dl = dall * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) *o++ = dl * (((q[l+16]>>shift)&3) - ((hm[l+16]&m)?0:4));
                shift += 2; m <<= 1;          /* hmask bit advances per shift-group (ggml) */
            }
            q += 32;
        }
    }
}

/* ---- Q6_K (ggml-faithful). 210 B: ql[128] low4, qh[64] high2, scales[16] i8, d f16 ---- */
enum { Q6K_ql_off = 0, Q6K_qh_off = 128, Q6K_scales_off = 192, Q6K_d_off = 208 };

static size_t ds4q_quantize_q6_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q6_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t bpr = ncols / QK_K;
    int8_t L[QK_K]; float scales[QK_K / 16], weight[16];
    for (int64_t r = 0; r < nrows; r++) {
        const float *xrow = src + start + (size_t)r * (size_t)ncols;
        for (int64_t b = 0; b < bpr; b++) {
            const float *x = xrow + (size_t)b * QK_K;
            uint8_t *y = out + (size_t)r * row_size + (size_t)b * 210;
            uint8_t *ql = y + Q6K_ql_off, *qh = y + Q6K_qh_off; int8_t *sc = (int8_t *)(y + Q6K_scales_off);
            float sumx2 = 0; for (int j = 0; j < QK_K; j++) sumx2 += x[j] * x[j];
            float sigma2 = sumx2 / QK_K, mscale = 0, mabs = 0;
            for (int ib = 0; ib < 16; ib++) {
                if (quant_weights) { const float *qw = quant_weights + (size_t)b*QK_K + 16*ib;
                    for (int l = 0; l < 16; l++) weight[l] = qw[l] * sqrtf(sigma2 + x[16*ib+l]*x[16*ib+l]); }
                else for (int l = 0; l < 16; l++) weight[l] = x[16*ib+l]*x[16*ib+l];
                scales[ib] = ds4q_make_qx_quants(16, 32, x + 16*ib, L + 16*ib, weight);
                if (fabsf(scales[ib]) > mabs) { mabs = fabsf(scales[ib]); mscale = scales[ib]; }
            }
            uint16_t hd;
            if (mabs < 1e-30f) { memset(y, 0, 210); hd = ds4q_f32_to_f16(0.f); memcpy(y + Q6K_d_off, &hd, 2); continue; }
            float iscale = -128.f / mscale;
            hd = ds4q_f32_to_f16(1.f / iscale); memcpy(y + Q6K_d_off, &hd, 2);
            for (int ib = 0; ib < 16; ib++) sc[ib] = (int8_t)DS4Q_MIN(127, ds4q_nearest_int(iscale * scales[ib]));
            float dall = ds4q_f16_to_f32(hd);
            for (int j = 0; j < 16; j++) {
                float d = dall * sc[j]; if (d == 0.f) { for (int ii = 0; ii < 16; ii++) L[16*j+ii] = 32; continue; }
                for (int ii = 0; ii < 16; ii++) { int l = DS4Q_MAX(-32, DS4Q_MIN(31, ds4q_nearest_int(x[16*j+ii]/d))); L[16*j+ii] = (int8_t)(l + 32); }
            }
            for (int j = 0; j < QK_K; j += 128) {
                for (int l = 0; l < 32; l++) {
                    uint8_t q1 = L[j+l]&0xF, q2 = L[j+l+32]&0xF, q3 = L[j+l+64]&0xF, q4 = L[j+l+96]&0xF;
                    ql[l] = q1 | (q3 << 4); ql[l+32] = q2 | (q4 << 4);
                    qh[l] = (uint8_t)((L[j+l]>>4) | ((L[j+l+32]>>4)<<2) | ((L[j+l+64]>>4)<<4) | ((L[j+l+96]>>4)<<6));
                }
                ql += 64; qh += 32;
            }
        }
    }
    return (size_t)nrows * row_size;
}

void ds4q_dequantize_q6_k(const void *blocks, float *out, int64_t n) {
    const uint8_t *p = (const uint8_t *)blocks;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *y = p + (size_t)i * 210, *ql = y + Q6K_ql_off, *qh = y + Q6K_qh_off;
        const int8_t *sc = (const int8_t *)(y + Q6K_scales_off);
        uint16_t hd; memcpy(&hd, y + Q6K_d_off, 2); float d = ds4q_f16_to_f32(hd);
        float *o = out + (size_t)i * QK_K;
        for (int nn = 0; nn < QK_K; nn += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l]&0xF) | (((qh[l]>>0)&3)<<4)) - 32;
                int8_t q2 = (int8_t)((ql[l+32]&0xF) | (((qh[l]>>2)&3)<<4)) - 32;
                int8_t q3 = (int8_t)((ql[l]>>4) | (((qh[l]>>4)&3)<<4)) - 32;
                int8_t q4 = (int8_t)((ql[l+32]>>4) | (((qh[l]>>6)&3)<<4)) - 32;
                o[l] = d*sc[is]*q1; o[l+32] = d*sc[is+2]*q2; o[l+64] = d*sc[is+4]*q3; o[l+96] = d*sc[is+6]*q4;
            }
            ql += 64; qh += 32; sc += 8; o += 128;
        }
    }
}

/* ---- Q5_K (ggml-faithful). 176 B: d, dmin, scales[12], qh[32], qs[128]. q4_K + 5th bit ---- */
enum { Q5K_d_off = 0, Q5K_dmin_off = 2, Q5K_scales_off = 4, Q5K_qh_off = 16, Q5K_qs_off = 48 };

static size_t ds4q_quantize_q5_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q5_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t bpr = ncols / QK_K;
    uint8_t L[QK_K], Laux[32], Ls[QK_K/32], Lm[QK_K/32];
    float weights[32], sw[QK_K/32], mins[QK_K/32], scales[QK_K/32];
    for (int64_t r = 0; r < nrows; r++) {
        const float *xrow = src + start + (size_t)r * (size_t)ncols;
        for (int64_t b = 0; b < bpr; b++) {
            const float *x = xrow + (size_t)b * QK_K;
            uint8_t *y = out + (size_t)r * row_size + (size_t)b * 176;
            uint8_t *scales_out = y + Q5K_scales_off, *qh = y + Q5K_qh_off, *qs = y + Q5K_qs_off;
            float sumx2 = 0; for (int l = 0; l < QK_K; l++) sumx2 += x[l]*x[l];
            float sigma2 = 2*sumx2/QK_K, avx = sqrtf(sigma2);
            for (int j = 0; j < QK_K/32; j++) {
                if (quant_weights) { const float *qw = quant_weights + (size_t)b*QK_K + 32*j;
                    for (int l = 0; l < 32; l++) weights[l] = qw[l]*sqrtf(sigma2 + x[32*j+l]*x[32*j+l]); }
                else for (int l = 0; l < 32; l++) weights[l] = avx + fabsf(x[32*j+l]);
                float s = 0; for (int l = 0; l < 32; l++) s += weights[l]; sw[j] = s;
                scales[j] = ds4q_make_qkx3_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
            }
            float d_block = ds4q_make_qp_quants(QK_K/32, 63, scales, Ls, sw);
            float m_block = ds4q_make_qp_quants(QK_K/32, 63, mins, Lm, sw);
            for (int j = 0; j < QK_K/32; j++) {
                uint8_t ls = Ls[j], lm = Lm[j];
                if (j < 4) { scales_out[j] = ls; scales_out[j+4] = lm; }
                else { scales_out[j+4] = (ls&0xF)|((lm&0xF)<<4); scales_out[j-4] |= (uint8_t)((ls>>4)<<6); scales_out[j] |= (uint8_t)((lm>>4)<<6); }
            }
            uint16_t d = ds4q_f32_to_f16(d_block), dmin = ds4q_f32_to_f16(m_block);
            memcpy(y + Q5K_d_off, &d, 2); memcpy(y + Q5K_dmin_off, &dmin, 2);
            uint8_t sc, m;
            for (int j = 0; j < QK_K/32; j++) {
                ds4q_get_scale_min_k4(j, scales_out, &sc, &m);
                float dd = ds4q_f16_to_f32(d) * sc; if (!dd) continue;
                float dm = ds4q_f16_to_f32(dmin) * m;
                for (int ii = 0; ii < 32; ii++) { int l = ds4q_nearest_int((x[32*j+ii]+dm)/dd); l = DS4Q_MAX(0, DS4Q_MIN(31, l)); L[32*j+ii] = (uint8_t)l; }
            }
            memset(qh, 0, 32);
            uint8_t m1 = 1, m2 = 2; uint8_t *qsp = qs;
            for (int nn = 0; nn < QK_K; nn += 64) {
                for (int l = 0; l < 32; l++) {
                    qsp[l] = (L[nn+l]&0xF) | ((L[nn+l+32]&0xF)<<4);
                    if (L[nn+l] > 15) qh[l] |= m1;
                    if (L[nn+l+32] > 15) qh[l] |= m2;
                }
                m1 <<= 2; m2 <<= 2; qsp += 32;
            }
        }
    }
    return (size_t)nrows * row_size;
}

void ds4q_dequantize_q5_k(const void *blocks, float *out, int64_t n) {
    const uint8_t *p = (const uint8_t *)blocks;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *y = p + (size_t)i * 176, *qs = y + Q5K_qs_off, *qh = y + Q5K_qh_off, *scales = y + Q5K_scales_off;
        uint16_t hd, hm; memcpy(&hd, y + Q5K_d_off, 2); memcpy(&hm, y + Q5K_dmin_off, 2);
        float d = ds4q_f16_to_f32(hd), dmin = ds4q_f16_to_f32(hm);
        float *o = out + (size_t)i * QK_K; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K/64; j++) {
            uint8_t sc, mm;
            ds4q_get_scale_min_k4(2*j, scales, &sc, &mm); float d1 = d*sc, m1 = dmin*mm;
            ds4q_get_scale_min_k4(2*j+1, scales, &sc, &mm); float d2 = d*sc, m2 = dmin*mm;
            for (int l = 0; l < 32; l++) o[l]    = d1 * ((qs[l]&0xF) + ((qh[l]&u1)?16:0)) - m1;
            for (int l = 0; l < 32; l++) o[l+32] = d2 * ((qs[l]>>4)  + ((qh[l]&u2)?16:0)) - m2;
            o += 64; qs += 32; u1 <<= 2; u2 <<= 2;
        }
    }
}

/* ds4-spark: dequants for the remaining ladder tiers, needed by the MSE oracle
 * (round-trip vs FP4 source). q4_K = q5_K without the qh 5th bit; canonical
 * ggml layouts (d@0 dmin@2 scales@4 qs@16, 144B). */
void ds4q_dequantize_q4_k(const void *blocks, float *out, int64_t n) {
    const uint8_t *p = (const uint8_t *)blocks;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *y = p + (size_t)i * 144, *scales = y + 4, *qs = y + 16;
        uint16_t hd, hm; memcpy(&hd, y, 2); memcpy(&hm, y + 2, 2);
        float d = ds4q_f16_to_f32(hd), dmin = ds4q_f16_to_f32(hm);
        float *o = out + (size_t)i * QK_K;
        for (int j = 0; j < QK_K/64; j++) {
            uint8_t sc, mm;
            ds4q_get_scale_min_k4(2*j,   scales, &sc, &mm); float d1 = d*sc, m1 = dmin*mm;
            ds4q_get_scale_min_k4(2*j+1, scales, &sc, &mm); float d2 = d*sc, m2 = dmin*mm;
            for (int l = 0; l < 32; l++) o[l]    = d1 * (qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) o[l+32] = d2 * (qs[l] >> 4)  - m2;
            o += 64; qs += 32;
        }
    }
}

/* q2_K: scales[16]@0, qs[64]@16, d@80, dmin@82 (84B); 2-bit, 16 sub-blocks. */
void ds4q_dequantize_q2_k(const void *blocks, float *out, int64_t n) {
    const uint8_t *p = (const uint8_t *)blocks;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *y = p + (size_t)i * 84, *scales = y, *q = y + 16;
        uint16_t hd, hm; memcpy(&hd, y + 80, 2); memcpy(&hm, y + 82, 2);
        float d = ds4q_f16_to_f32(hd), dmin = ds4q_f16_to_f32(hm);
        float *o = out + (size_t)i * QK_K; int is = 0;
        for (int g = 0; g < QK_K; g += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = scales[is++]; float dl = d*(sc&0xF), ml = dmin*(sc>>4);
                for (int l = 0; l < 16; l++) *o++ = dl * ((q[l]    >> shift) & 3) - ml;
                sc = scales[is++]; dl = d*(sc&0xF); ml = dmin*(sc>>4);
                for (int l = 0; l < 16; l++) *o++ = dl * ((q[l+16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

/* iq2_xxs dequant (oracle floor tier). Inverts ds4q_quantize_iq2_xxs: block is
 * d(f16)@0 then 8 groups of [u32 grid-indices, u32 signs|scale] (66B total). The
 * codebook is the global grid built by ds4q_iq2_xxs_init (8 int8 {1,3,5,7} per
 * entry); signs use the canonical even-parity 7-bit ksigns table. */
static const uint8_t ds4q_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
void ds4q_dequantize_iq2_xxs(const void *blocks, float *out, int64_t n) {
    ds4q_iq2_xxs_init();
    const int8_t *grid = (const int8_t *)ds4q_iq2_xxs_data.grid;
    static const uint8_t kmask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    const uint8_t *p = (const uint8_t *)blocks;
    for (int64_t i = 0; i < n / QK_K; i++) {
        const uint8_t *blk = p + (size_t)i * 66;
        uint16_t hd; memcpy(&hd, blk, 2);
        const float d = ds4q_f16_to_f32(hd);
        const uint8_t *q2 = blk + 2;
        float *y = out + (size_t)i * QK_K;
        for (int ib = 0; ib < QK_K/32; ib++) {
            uint32_t aux32[2]; memcpy(aux32, q2 + (size_t)ib * 8, 8);
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const float db = d * (float)(2u * (aux32[1] >> 28) + 1u);
            for (int l = 0; l < 4; l++) {
                const int8_t *g = grid + (size_t)aux8[l] * 8;
                const uint8_t signs = ds4q_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++)
                    *y++ = db * (float)g[j] * ((signs & kmask[j]) ? -1.f : 1.f);
            }
        }
    }
}

void ds4q_quantize_init(ds4q_type type) {
    if (type == DS4Q_TYPE_IQ2_XXS) {
        ds4q_iq2_xxs_init();
    }
}

size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix) {
    if (type == DS4Q_TYPE_Q8_0) {
        (void)imatrix;
        return ds4q_quantize_q8_0(src, dst, start, nrows, ncols);
    }
    if (type == DS4Q_TYPE_Q2_K) {
        return ds4q_quantize_q2_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q3_K) {
        return ds4q_quantize_q3_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q4_K) {
        return ds4q_quantize_q4_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q5_K) {
        return ds4q_quantize_q5_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q6_K) {
        return ds4q_quantize_q6_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_IQ2_XXS) {
        return ds4q_quantize_iq2_xxs(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_FP8_E4M3) {
        (void)imatrix;
        return ds4q_quantize_fp8_e4m3(src, dst, start, nrows, ncols);
    }
    (void)src;
    (void)dst;
    (void)start;
    (void)nrows;
    (void)ncols;
    (void)imatrix;
    assert(!"unsupported DS4 quantization target");
    return 0;
}

float ds4q_f16_to_f32(uint16_t bits) {
    const uint32_t w = (uint32_t)bits << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = ds4q_f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = ds4q_f32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? ds4q_f32_to_bits(denormalized_value) : ds4q_f32_to_bits(normalized_value));
    return ds4q_f32_from_bits(result);
}

float ds4q_bf16_to_f32(uint16_t bits) {
    return ds4q_f32_from_bits((uint32_t)bits << 16);
}

void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = ds4q_f32_to_f16(src[i]);
}

void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = ds4q_f32_to_bits(src[i]);
        if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
            dst[i] = (uint16_t)((bits >> 16) | 64);
        } else {
            dst[i] = (uint16_t)((bits + (UINT32_C(0x7fff) + ((bits >> 16) & 1))) >> 16);
        }
    }
}
