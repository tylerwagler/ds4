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

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_dspark_markov_head() != 0) rc = 1;
    if (check_dspark_confidence_head() != 0) rc = 1;
    if (check_dspark_non_causal_attention() != 0) rc = 1;
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
