#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                               comp,
                                               0,
                                               0,
                                               n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_dspark_non_causal_attention(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_tokens = 5;
    const uint32_t n_raw = 5;
    const uint32_t raw_cap = 5;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t heads_count = (uint64_t)n_tokens * n_head * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *kvrow = (float *)calloc((size_t)head_dim, sizeof(float));
    float *q_row = (float *)calloc((size_t)n_head * head_dim, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count * n_tokens, sizeof(float));
    float *heads_causal = (float *)calloc((size_t)heads_count, sizeof(float));
    float *heads_non_causal = (float *)calloc((size_t)heads_count, sizeof(float));
    if (!sinks || !kvrow || !q_row || !raw_host || !q_host || !heads_causal || !heads_non_causal) return 1;

    for (uint32_t d = 0; d < head_dim; d++) {
        kvrow[d] = (float)(d % 16) * 0.1f;
    }
    for (uint32_t d = 0; d < n_head * head_dim; d++) {
        q_row[d] = (float)(d % 8) * 0.2f;
    }
    for (uint32_t t = 0; t < n_raw; t++) {
        memcpy(raw_host + (uint64_t)t * head_dim, kvrow, head_dim * sizeof(float));
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        memcpy(q_host + (uint64_t)(t * n_head) * head_dim, q_row, (uint64_t)n_head * head_dim * sizeof(float));
    }

    ds4_gpu_tensor *heads_c = ds4_gpu_tensor_alloc(heads_count * sizeof(float));
    ds4_gpu_tensor *heads_nc = ds4_gpu_tensor_alloc(heads_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)n_tokens * q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    int rc = 1;
    if (heads_c && heads_nc && q && raw &&
        ds4_gpu_tensor_write(q, 0, q_host, (uint64_t)n_tokens * q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float))) {

        int ok_c = ds4_gpu_attention_decode_raw_batch_heads_tensor(heads_c,
                                                                    sinks,
                                                                    n_head * sizeof(float),
                                                                    0,
                                                                    q,
                                                                    raw,
                                                                    n_tokens, 0,
                                                                    n_raw, raw_cap, 0,
                                                                    0, n_head, head_dim, 0);
        int ok_nc = ds4_gpu_attention_decode_raw_batch_heads_tensor(heads_nc,
                                                                     sinks,
                                                                     n_head * sizeof(float),
                                                                     0,
                                                                     q,
                                                                     raw,
                                                                     n_tokens, 0,
                                                                     n_raw, raw_cap, 0,
                                                                     0, n_head, head_dim, 1);
        if (ok_c && ok_nc && ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(heads_c, 0, heads_causal, heads_count * sizeof(float)) &&
            ds4_gpu_tensor_read(heads_nc, 0, heads_non_causal, heads_count * sizeof(float))) {

            int causal_all_equal = 1;
            for (uint32_t t = 1; t < n_tokens; t++) {
                if (memcmp(heads_causal, heads_causal + (uint64_t)t * n_head * head_dim,
                           n_head * head_dim * sizeof(float)) != 0) {
                    causal_all_equal = 0;
                    break;
                }
            }

            int non_causal_all_equal = 1;
            for (uint32_t t = 1; t < n_tokens; t++) {
                if (memcmp(heads_non_causal, heads_non_causal + (uint64_t)t * n_head * head_dim,
                           n_head * head_dim * sizeof(float)) != 0) {
                    non_causal_all_equal = 0;
                    break;
                }
            }

            if (causal_all_equal) {
                fprintf(stderr, "dspark_attn: causal outputs unexpectedly equal\n");
                rc = 1;
            } else if (!non_causal_all_equal) {
                fprintf(stderr, "dspark_attn: non-causal outputs not all equal\n");
                rc = 1;
            } else {
                rc = 0;
            }
        }
    }

    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads_nc);
    ds4_gpu_tensor_free(heads_c);
    free(heads_non_causal);
    free(heads_causal);
    free(q_host);
    free(raw_host);
    free(q_row);
    free(kvrow);
    free(sinks);
    return rc;
}

static int check_dspark_markov_head(void) {
    const uint32_t vocab_size = 4096;
    const uint32_t embed_dim = 256;
    const uint32_t n_draft = 5;
    int32_t prev_tokens[5] = {42, 100, 500, 1200, 3000};

    float *w1_host = (float *)calloc((size_t)vocab_size * embed_dim, sizeof(float));
    float *w2_host = (float *)calloc((size_t)vocab_size * embed_dim, sizeof(float));
    float *base_host = (float *)calloc((size_t)vocab_size, sizeof(float));
    if (!w1_host || !w2_host || !base_host) return 1;

    for (uint32_t v = 0; v < vocab_size; v++) {
        for (uint32_t i = 0; i < embed_dim; i++) {
            w1_host[(uint64_t)v * embed_dim + i] = (float)((v * 7 + i * 13) % 100) * 0.01f;
            w2_host[(uint64_t)v * embed_dim + i] = (float)((v * 3 + i * 11) % 50) * 0.02f;
        }
        base_host[v] = (float)(v % 200) * 0.01f;
    }

    ds4_gpu_tensor *w1 = ds4_gpu_tensor_alloc((uint64_t)vocab_size * embed_dim * sizeof(float));
    ds4_gpu_tensor *w2 = ds4_gpu_tensor_alloc((uint64_t)vocab_size * embed_dim * sizeof(float));
    ds4_gpu_tensor *base = ds4_gpu_tensor_alloc((uint64_t)vocab_size * sizeof(float));
    ds4_gpu_tensor *ref_logits = ds4_gpu_tensor_alloc((uint64_t)vocab_size * sizeof(float));
    int rc = 1;
    if (w1 && w2 && base && ref_logits &&
        ds4_gpu_tensor_write(w1, 0, w1_host, (uint64_t)vocab_size * embed_dim * sizeof(float)) &&
        ds4_gpu_tensor_write(w2, 0, w2_host, (uint64_t)vocab_size * embed_dim * sizeof(float)) &&
        ds4_gpu_tensor_write(base, 0, base_host, (uint64_t)vocab_size * sizeof(float))) {

        int32_t id = prev_tokens[0];
        for (uint32_t step = 0; step < n_draft; step++) {
            int32_t gpu_id = 0;
            if (!ds4_gpu_dspark_markov_step(ref_logits, &gpu_id, base, w1, w2,
                                            id, vocab_size, embed_dim)) {
                rc = 1;
                goto cleanup;
            }

            const float *embed = w1_host + (uint64_t)id * embed_dim;
            float cpu_best = -1e30f;
            int32_t cpu_id = 0;
            for (uint32_t v = 0; v < vocab_size; v++) {
                float dot = 0.0f;
                const float *w2r = w2_host + (uint64_t)v * embed_dim;
                for (uint32_t i = 0; i < embed_dim; i++)
                    dot += w2r[i] * embed[i];
                float val = base_host[v] + dot;
                if (val > cpu_best) { cpu_best = val; cpu_id = (int32_t)v; }
            }

            if (gpu_id != cpu_id) {
                fprintf(stderr, "markov step %u: GPU=%d CPU=%d\n", step, gpu_id, cpu_id);
                rc = 1;
                goto cleanup;
            }
            id = gpu_id;
        }
        rc = 0;
    }

cleanup:
    ds4_gpu_tensor_free(ref_logits);
    ds4_gpu_tensor_free(base);
    ds4_gpu_tensor_free(w2);
    ds4_gpu_tensor_free(w1);
    free(base_host);
    free(w2_host);
    free(w1_host);
    return rc;
}

static int check_dspark_confidence_head(void) {
    const uint32_t n_positions = 3;
    const uint32_t hidden_dim = 32;
    const uint32_t embed_dim = 8;
    const uint32_t total_dim = hidden_dim + embed_dim;

    float *proj_host = (float *)calloc(total_dim, sizeof(float));
    float *hidden_host = (float *)calloc((size_t)n_positions * hidden_dim, sizeof(float));
    float *embed_host = (float *)calloc((size_t)n_positions * embed_dim, sizeof(float));
    float scores_host[3];
    if (!proj_host || !hidden_host || !embed_host) return 1;

    for (uint32_t i = 0; i < total_dim; i++)
        proj_host[i] = (float)((i % 7) + 1) * 0.1f;
    for (uint32_t p = 0; p < n_positions; p++) {
        for (uint32_t i = 0; i < hidden_dim; i++)
            hidden_host[(uint64_t)p * hidden_dim + i] = (float)((p + i) % 5) * 0.5f;
        for (uint32_t i = 0; i < embed_dim; i++)
            embed_host[(uint64_t)p * embed_dim + i] = (float)((p + i) % 3) * 0.3f;
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)n_positions * sizeof(float));
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_alloc((uint64_t)n_positions * hidden_dim * sizeof(float));
    ds4_gpu_tensor *embed = ds4_gpu_tensor_alloc((uint64_t)n_positions * embed_dim * sizeof(float));
    ds4_gpu_tensor *proj = ds4_gpu_tensor_alloc((uint64_t)total_dim * sizeof(float));
    int rc = 1;
    if (scores && hidden && embed && proj &&
        ds4_gpu_tensor_write(hidden, 0, hidden_host, (uint64_t)n_positions * hidden_dim * sizeof(float)) &&
        ds4_gpu_tensor_write(embed, 0, embed_host, (uint64_t)n_positions * embed_dim * sizeof(float)) &&
        ds4_gpu_tensor_write(proj, 0, proj_host, (uint64_t)total_dim * sizeof(float)) &&
        ds4_gpu_dspark_confidence_score(scores, hidden, embed, proj,
                                        n_positions, hidden_dim, embed_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(scores, 0, scores_host, (uint64_t)n_positions * sizeof(float))) {
        rc = 0;
        for (uint32_t p = 0; p < n_positions; p++) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < hidden_dim; i++)
                dot += hidden_host[(uint64_t)p * hidden_dim + i] * proj_host[i];
            for (uint32_t i = 0; i < embed_dim; i++)
                dot += embed_host[(uint64_t)p * embed_dim + i] * proj_host[hidden_dim + i];
            float expected = 1.0f / (1.0f + expf(-dot));
            if (fabsf(scores_host[p] - expected) > 1e-5f) {
                fprintf(stderr, "confidence pos %u: GPU=%.6f CPU=%.6f\n",
                        p, (double)scores_host[p], (double)expected);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(proj);
    ds4_gpu_tensor_free(embed);
    ds4_gpu_tensor_free(hidden);
    ds4_gpu_tensor_free(scores);
    free(embed_host);
    free(hidden_host);
    free(proj_host);
    return rc;
}

static int check_dspark_fp8_kv_pack(void) {
    const uint32_t head_dim = 512;
    const uint32_t nblk = (head_dim + 63) / 64;
    float *src = (float *)calloc(head_dim, sizeof(float));
    if (!src) return 1;
    for (uint32_t i = 0; i < head_dim; i++) src[i] = ((float)i - 256.0f) * 0.5f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(head_dim * sizeof(float));
    ds4_gpu_tensor *packed = ds4_gpu_tensor_alloc(head_dim);
    ds4_gpu_tensor *scales = ds4_gpu_tensor_alloc(nblk * sizeof(float));
    if (!x || !packed || !scales) { free(src); return 1; }
    if (!ds4_gpu_tensor_write(x, 0, src, head_dim * sizeof(float))) { free(src); return 1; }

    int rc = 0;
    if (!ds4_gpu_dsv4_fp8_kv_pack_tensor(x, packed, scales, 1, head_dim)) {
        fprintf(stderr, "ds4: fp8 kv pack failed\n");
        rc = 1;
    }
    if (rc == 0) {
        uint8_t *packed_host = (uint8_t *)malloc(head_dim);
        float *scales_host = (float *)malloc(nblk * sizeof(float));
        if (!packed_host || !scales_host) rc = 1;
        else {
            if (!ds4_gpu_tensor_read(packed, 0, packed_host, head_dim))
                rc = 1;
            if (!ds4_gpu_tensor_read(scales, 0, scales_host, nblk * sizeof(float)))
                rc = 1;
            if (rc == 0) {
                for (uint32_t i = 0; i < head_dim; i++) {
                    float scale = scales_host[i >> 6];
                    int idx = packed_host[i] & 0x7f;
                    float val = 0.0f;
                    int exp = (idx >> 3) & 15;
                    int mant = idx & 7;
                    if (exp == 0) val = (float)mant * 0.001953125f;
                    else val = (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
                    if (packed_host[i] & 0x80) val = -val;
                    float recon = val * scale;
                    float diff = fabsf(recon - src[i]);
                    float tol = fmaxf(fabsf(src[i]) * 0.3f + 1e-5f, 1e-4f);
                    if (diff > tol) {
                        fprintf(stderr, "ds4: fp8 packing error at %u: src=%.6f recon=%.6f\n",
                                i, (double)src[i], (double)recon);
                        rc = 1;
                        break;
                    }
                }
            }
            free(packed_host);
            free(scales_host);
        }
    }

    ds4_gpu_tensor_free(scales);
    ds4_gpu_tensor_free(packed);
    ds4_gpu_tensor_free(x);
    free(src);
    return rc;
}

/* ---- Stage 1: MX (microscaling) compressed-KV pack/dequant round-trip ---- */
#define MXKV_FP8 1u
#define MXKV_FP4 2u
#define MXKV_BLK 32u

static float mxkv_e4m3_value(int i) {
    int exp = (i >> 3) & 15, mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}
static float mxkv_e4m3_snap(float x) { /* returns dequantized magnitude, mirrors device */
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (mxkv_e4m3_value(mid) <= ax) lo = mid; else hi = mid - 1; }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - mxkv_e4m3_value(best)), nd = fabsf(ax - mxkv_e4m3_value(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return (x < 0.0f ? -1.0f : 1.0f) * mxkv_e4m3_value(best);
}
static float mxkv_e2m1_value(int i) {
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return t[i & 7];
}
static float mxkv_e2m1_snap(float x) {
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0; float bd = fabsf(ax - mxkv_e2m1_value(0));
    for (int i = 1; i < 8; i++) { float d = fabsf(ax - mxkv_e2m1_value(i));
        if (d < bd || (d == bd && ((i & 1) == 0) && ((best & 1) != 0))) { best = i; bd = d; } }
    return (x < 0.0f ? -1.0f : 1.0f) * mxkv_e2m1_value(best);
}
static float mxkv_ref_scale(float amax, float max_repr) {
    float ratio = fmaxf(amax, 1.0e-20f) / max_repr;
    int k = (int)ceilf(log2f(ratio)), e = k + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return exp2f((float)(e - 127));
}

static int mxkv_run_case(uint32_t fmt, uint32_t n_tok, uint32_t head_dim, const float *src) {
    const uint32_t nblk = head_dim / MXKV_BLK;
    const uint32_t rowbytes = (fmt == MXKV_FP4 ? head_dim / 2 : head_dim) + nblk;
    const float max_repr = (fmt == MXKV_FP4) ? 6.0f : 448.0f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n_tok * head_dim * sizeof(float));
    ds4_gpu_tensor *packed = ds4_gpu_tensor_alloc((uint64_t)n_tok * rowbytes);
    ds4_gpu_tensor *deq = ds4_gpu_tensor_alloc((uint64_t)n_tok * head_dim * sizeof(float));
    float *gpu_deq = (float *)malloc((size_t)n_tok * head_dim * sizeof(float));
    int rc = 1;
    if (x && packed && deq && gpu_deq &&
        ds4_gpu_tensor_write(x, 0, src, (uint64_t)n_tok * head_dim * sizeof(float)) &&
        ds4_gpu_mxkv_pack_tensor(x, packed, fmt, n_tok, head_dim) &&
        ds4_gpu_mxkv_dequant_tensor(packed, deq, fmt, n_tok, head_dim) &&
        ds4_gpu_tensor_read(deq, 0, gpu_deq, (uint64_t)n_tok * head_dim * sizeof(float))) {
        rc = 0;
        double sum_abs = 0.0, sum_ref = 0.0; float worst = 0.0f;
        for (uint32_t r = 0; r < n_tok && rc == 0; r++) {
            for (uint32_t b = 0; b < nblk; b++) {
                float amax = 0.0f;
                for (uint32_t j = 0; j < MXKV_BLK; j++) amax = fmaxf(amax, fabsf(src[r * head_dim + b * MXKV_BLK + j]));
                float scale = mxkv_ref_scale(amax, max_repr);
                for (uint32_t j = 0; j < MXKV_BLK; j++) {
                    uint32_t d = b * MXKV_BLK + j;
                    float in = src[r * head_dim + d];
                    float ref = (fmt == MXKV_FP4 ? mxkv_e2m1_snap(in / scale) : mxkv_e4m3_snap(in / scale)) * scale;
                    float got = gpu_deq[r * head_dim + d];
                    if (fabsf(got - ref) > 1e-3f) {
                        fprintf(stderr, "mxkv fmt=%u mismatch @row%u d%u: gpu=%.6f ref=%.6f (in=%.6f scale=%.6g)\n",
                                fmt, r, d, (double)got, (double)ref, (double)in, (double)scale);
                        rc = 1; break;
                    }
                    sum_abs += fabs((double)got - (double)in);
                    sum_ref += fabs((double)in);
                    worst = fmaxf(worst, fabsf(got - in));
                }
            }
        }
        if (rc == 0) {
            double rel = sum_ref > 0 ? sum_abs / sum_ref : 0.0;
            /* FP8 ~<3% mean rel error, FP4 much coarser but bounded. */
            double bound = (fmt == MXKV_FP4) ? 0.20 : 0.05;
            printf("  mxkv fmt=%u rowbytes=%u mean_rel_err=%.4f worst_abs=%.4f -> %s\n",
                   fmt, rowbytes, rel, (double)worst, rel <= bound ? "OK" : "HIGH");
            if (rel > bound) rc = 1;
        }
    }
    ds4_gpu_tensor_free(deq); ds4_gpu_tensor_free(packed); ds4_gpu_tensor_free(x);
    free(gpu_deq);
    return rc;
}

static int check_mxkv_roundtrip(void) {
    const uint32_t n_tok = 48, head_dim = 512;
    float *src = (float *)malloc((size_t)n_tok * head_dim * sizeof(float));
    if (!src) return 1;
    /* Vary magnitude across blocks and rows so E8M0 per-block scaling is exercised. */
    for (uint32_t r = 0; r < n_tok; r++)
        for (uint32_t d = 0; d < head_dim; d++) {
            float blkscale = exp2f((float)((int)(d / MXKV_BLK % 12) - 6));   /* 2^-6 .. 2^5 */
            float v = sinf((float)(d * 3 + r * 7) * 0.031f) * blkscale * (1.0f + 0.1f * (float)r);
            src[r * head_dim + d] = v;
        }
    int rc = 0;
    if (mxkv_run_case(MXKV_FP8, n_tok, head_dim, src) != 0) rc = 1;
    if (mxkv_run_case(MXKV_FP4, n_tok, head_dim, src) != 0) rc = 1;
    free(src);
    return rc;
}

/* MXKV gather primitive: dequant selected rows (contiguous and transposed). */
static int check_mxkv_gather(void) {
    const uint32_t n_cap = 100, n_sel = 32, head_dim = 512, nblk = head_dim / MXKV_BLK;
    const uint32_t rowbytes = head_dim + nblk;   /* MXFP8 */
    float *src = (float *)malloc((size_t)n_cap * head_dim * sizeof(float));
    int32_t *rows = (int32_t *)malloc(n_sel * sizeof(int32_t));
    if (!src || !rows) return 1;
    for (uint32_t r = 0; r < n_cap; r++)
        for (uint32_t d = 0; d < head_dim; d++)
            src[r * head_dim + d] = sinf((float)(r * 5 + d) * 0.02f) * (1.0f + 0.05f * (float)r);
    for (uint32_t i = 0; i < n_sel; i++) rows[i] = (int32_t)((i * 37 + 11) % n_cap);   /* scattered picks */

    ds4_gpu_tensor *xf = ds4_gpu_tensor_alloc((uint64_t)n_cap * head_dim * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc((uint64_t)n_cap * rowbytes);
    ds4_gpu_tensor *idx = ds4_gpu_tensor_alloc((uint64_t)n_sel * sizeof(int32_t));
    ds4_gpu_tensor *g0 = ds4_gpu_tensor_alloc((uint64_t)n_sel * head_dim * sizeof(float));
    ds4_gpu_tensor *g1 = ds4_gpu_tensor_alloc((uint64_t)n_sel * head_dim * sizeof(float));
    float *out0 = (float *)malloc((size_t)n_sel * head_dim * sizeof(float));
    float *out1 = (float *)malloc((size_t)n_sel * head_dim * sizeof(float));
    int rc = 1;
    if (xf && cache && idx && g0 && g1 && out0 && out1 &&
        ds4_gpu_tensor_write(xf, 0, src, (uint64_t)n_cap * head_dim * sizeof(float)) &&
        ds4_gpu_tensor_write(idx, 0, rows, (uint64_t)n_sel * sizeof(int32_t)) &&
        ds4_gpu_mxkv_pack_tensor(xf, cache, MXKV_FP8, n_cap, head_dim) &&
        ds4_gpu_mxkv_gather_dequant_tensor(cache, g0, idx, n_sel, n_cap, head_dim, MXKV_FP8, 0) &&
        ds4_gpu_mxkv_gather_dequant_tensor(cache, g1, idx, n_sel, n_cap, head_dim, MXKV_FP8, 1) &&
        ds4_gpu_tensor_read(g0, 0, out0, (uint64_t)n_sel * head_dim * sizeof(float)) &&
        ds4_gpu_tensor_read(g1, 0, out1, (uint64_t)n_sel * head_dim * sizeof(float))) {
        rc = 0;
        for (uint32_t i = 0; i < n_sel && rc == 0; i++) {
            const float *sr = src + (size_t)rows[i] * head_dim;
            for (uint32_t b = 0; b < nblk; b++) {
                float amax = 0.0f;
                for (uint32_t j = 0; j < MXKV_BLK; j++) amax = fmaxf(amax, fabsf(sr[b * MXKV_BLK + j]));
                float scale = mxkv_ref_scale(amax, 448.0f);
                for (uint32_t j = 0; j < MXKV_BLK; j++) {
                    uint32_t d = b * MXKV_BLK + j;
                    float ref = mxkv_e4m3_snap(sr[d] / scale) * scale;
                    if (fabsf(out0[(size_t)i * head_dim + d] - ref) > 1e-3f ||    /* contiguous */
                        fabsf(out1[(size_t)d * n_sel + i] - ref) > 1e-3f) {       /* transposed */
                        fprintf(stderr, "mxkv gather mismatch row%u d%u\n", i, d);
                        rc = 1; break;
                    }
                }
            }
        }
        if (rc == 0) printf("  mxkv gather n_sel=%u (contiguous + transposed) -> OK\n", n_sel);
    }
    ds4_gpu_tensor_free(g1); ds4_gpu_tensor_free(g0); ds4_gpu_tensor_free(idx);
    ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(xf);
    free(src); free(rows); free(out0); free(out1);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_mxkv_roundtrip() != 0) rc = 1;
    if (check_mxkv_gather() != 0) rc = 1;
    if (check_dspark_fp8_kv_pack() != 0) rc = 1;
    if (check_dspark_markov_head() != 0) rc = 1;
    if (check_dspark_confidence_head() != 0) rc = 1;
    if (check_dspark_non_causal_attention() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
