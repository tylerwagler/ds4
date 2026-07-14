/* Round-trip test for the four live quant formats (iq2_xxs, q2_K, fp8_e4m3,
 * mxfp4). Guards three properties the offline pipeline depends on:
 *   1. relative RMSE on Gaussian data is sane for each format's bit budget
 *      (a sign or layout bug shows up as rel-RMSE near sqrt(2));
 *   2. signs survive the round-trip (the historical mxfp4 encoder bug wrote
 *      the sign at bit 7 of a 4-bit nibble, silently flipping every negative
 *      weight positive — this test exists because of it);
 *   3. fp8/mxfp4 are value-level idempotent: re-quantizing a dequantized
 *      block reproduces exactly the same reals (byte layouts may differ when
 *      the block scale re-buckets, values may not).
 */
#include "quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned long st = 88172645463325252ULL;
static double rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (double)(st >> 11) / 9007199254740992.0; }
static float gauss(void) { double u = rnd(), v = rnd(); return (float)(sqrt(-2 * log(u + 1e-12)) * cos(6.2831853 * v)); }

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static void quant_roundtrip(ds4q_type t, void (*deq)(const void *, float *, int64_t),
                            const float *x, int n, int ncols, float *rc) {
    int nrows = n / ncols;
    size_t rs = ds4q_row_size(t, ncols);
    unsigned char *q = malloc(rs * nrows);
    /* iq2_xxs requires an imatrix; a flat one exercises the same code path. */
    float *iw = malloc((size_t)ncols * sizeof(float));
    for (int i = 0; i < ncols; i++) iw[i] = 1.0f;
    ds4q_quantize_init(t);
    ds4q_quantize_chunk(t, x, q, 0, nrows, ncols, iw);
    for (int r = 0; r < nrows; r++) deq(q + (size_t)r * rs, rc + (size_t)r * ncols, ncols);
    free(iw);
    free(q);
}

static void check_format(const char *name, ds4q_type t,
                         void (*deq)(const void *, float *, int64_t),
                         const float *x, int n, int ncols,
                         double max_rel_rmse, float sign_floor) {
    float *rc = malloc((size_t)n * sizeof(float));
    quant_roundtrip(t, deq, x, n, ncols, rc);

    double se = 0, sx = 0;
    for (int i = 0; i < n; i++) { double e = x[i] - rc[i]; se += e * e; sx += (double)x[i] * x[i]; }
    double rel = sqrt(se / sx);
    printf("  %-9s rel_rmse %.4f (max %.4f)\n", name, rel, max_rel_rmse);
    CHECK(rel <= max_rel_rmse, "%s rel_rmse %.4f exceeds %.4f", name, rel, max_rel_rmse);

    /* Signs must survive for every value large enough that the format cannot
     * round it to zero. sign_floor is per-format: |x| above this fraction of
     * the global amax cannot quantize to 0 in any block. */
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    int sign_flips = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(x[i]) < sign_floor * amax) continue;
        if (x[i] * rc[i] < 0) sign_flips++;
    }
    CHECK(sign_flips == 0, "%s flipped the sign of %d large-magnitude values", name, sign_flips);
    free(rc);
}

/* fp8/mxfp4 value-level idempotency: quantize(dequant(q)) must dequantize to
 * exactly the same reals. Catches encoder/decoder disagreement on layout,
 * sign position, or rounding of exactly-representable values. */
static void check_idempotent(const char *name, ds4q_type t,
                             void (*deq)(const void *, float *, int64_t),
                             const float *x, int n, int ncols) {
    float *r1 = malloc((size_t)n * sizeof(float));
    float *r2 = malloc((size_t)n * sizeof(float));
    quant_roundtrip(t, deq, x, n, ncols, r1);
    quant_roundtrip(t, deq, r1, n, ncols, r2);
    int diffs = 0;
    for (int i = 0; i < n; i++) if (r1[i] != r2[i]) diffs++;
    CHECK(diffs == 0, "%s not value-idempotent: %d/%d values changed on requant", name, diffs, n);
    free(r1); free(r2);
}

int main(void) {
    const int NC = 4096, NR = 8, N = NC * NR;
    float *x = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) x[i] = gauss() * 0.05f;

    printf("quant round-trip (Gaussian, zero mean — half the mass is negative):\n");
    /* Thresholds are ~2x the measured healthy values; the sign bug pushed
     * mxfp4 rel_rmse to ~1.4, far past any of these. */
    check_format("iq2_xxs", DS4Q_TYPE_IQ2_XXS,  ds4q_dequantize_iq2_xxs, x, N, NC, 0.50, 0.30f);
    check_format("q2_K",    DS4Q_TYPE_Q2_K,     ds4q_dequantize_q2_k,    x, N, NC, 0.40, 0.30f);
    check_format("fp8_e4m3",DS4Q_TYPE_FP8_E4M3, ds4q_dequantize_fp8_e4m3,x, N, NC, 0.05, 0.01f);
    check_format("mxfp4",   DS4Q_TYPE_MXFP4,    ds4q_dequantize_mxfp4,   x, N, NC, 0.30, 0.10f);

    check_idempotent("fp8_e4m3", DS4Q_TYPE_FP8_E4M3, ds4q_dequantize_fp8_e4m3, x, N, NC);
    check_idempotent("mxfp4",    DS4Q_TYPE_MXFP4,    ds4q_dequantize_mxfp4,    x, N, NC);

    /* Exactly-representable mxfp4 values must round-trip bit-perfectly,
     * including their signs. */
    {
        const float mags[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
        float xv[32], rc[32];
        for (int i = 0; i < 32; i++) {
            float m = mags[(i / 2) % 8];
            xv[i] = (i & 1) ? -m : m;
        }
        quant_roundtrip(DS4Q_TYPE_MXFP4, ds4q_dequantize_mxfp4, xv, 32, 32, rc);
        for (int i = 0; i < 32; i++)
            CHECK(rc[i] == xv[i] || (xv[i] == 0.0f && rc[i] == 0.0f),
                  "mxfp4 exact value %g came back as %g (i=%d)", xv[i], rc[i], i);
    }

    /* CUTLASS_MXFP4 (type 40) pack invariants:
     *   1. sizing matches the real dspark expert blobs (N=2048, K=4096 ->
     *      4456448 B/expert incl. 262144 B of SF; ground truth measured from
     *      gguf/dspark.gguf, produced by the validated CUTLASS packer);
     *   2. the data region is the source E2M1 array byte-verbatim;
     *   3. the SF swizzle is a bijection into the padded tile (every source
     *      scale lands on a unique in-range offset; padding bytes stay 0),
     *      and unswizzling recovers the source scales exactly. */
    {
        CHECK(ds4q_cutlass_mxfp4_sf_bytes(2048, 4096) == 262144,
              "cutlass_mxfp4 sf_bytes(2048,4096) = %zu, want 262144",
              ds4q_cutlass_mxfp4_sf_bytes(2048, 4096));
        CHECK(ds4q_cutlass_mxfp4_bytes(2048, 4096) == 4456448,
              "cutlass_mxfp4 bytes(2048,4096) = %zu, want 4456448",
              ds4q_cutlass_mxfp4_bytes(2048, 4096));

        const int64_t NN = 320, KK = 224;   /* exercises padding: 320 rows -> 384, 7 kblocks -> 8 */
        const size_t db = (size_t)(NN * KK / 2);
        const size_t sb = ds4q_cutlass_mxfp4_sf_bytes(NN, KK);
        const int64_t kbn = KK / 32;
        uint8_t *e2m1 = malloc(db);
        uint8_t *e8m0 = malloc((size_t)(NN * kbn));
        uint8_t *blob = malloc(db + sb);
        for (size_t i = 0; i < db; i++) e2m1[i] = (uint8_t)(rnd() * 256);
        for (int64_t i = 0; i < NN * kbn; i++) e8m0[i] = (uint8_t)(1 + i % 253); /* nonzero */
        ds4q_pack_cutlass_mxfp4(e2m1, e8m0, blob, NN, KK);

        CHECK(memcmp(blob, e2m1, db) == 0, "cutlass_mxfp4 data region is not byte-verbatim");

        /* Unswizzle by scanning the SF region: each nonzero byte must map
         * back from exactly one (row, kb); count and compare values. */
        const uint8_t *sf = blob + db;
        size_t nonzero = 0;
        for (size_t i = 0; i < sb; i++) if (sf[i]) nonzero++;
        CHECK(nonzero == (size_t)(NN * kbn),
              "cutlass_mxfp4 SF nonzero count %zu, want %lld (swizzle not injective or out of range)",
              nonzero, (long long)(NN * kbn));
        int sf_bad = 0;
        const int64_t kbp = (kbn + 3) / 4 * 4;
        for (int64_t r = 0; r < NN && sf_bad < 8; r++) {
            for (int64_t kb = 0; kb < kbn; kb++) {
                size_t off = (size_t)((r / 128) * (kbp / 4) + kb / 4) * 512
                           + (size_t)(r % 32) * 16 + (size_t)((r % 128) / 32) * 4 + (size_t)(kb % 4);
                if (off >= sb || sf[off] != e8m0[r * kbn + kb]) { sf_bad++; break; }
            }
        }
        CHECK(sf_bad == 0, "cutlass_mxfp4 SF unswizzle mismatch on %d row(s)", sf_bad);
        free(e2m1); free(e8m0); free(blob);
    }

    free(x);
    if (failures) { printf("%d FAILURE(S)\n", failures); return 1; }
    printf("OK\n");
    return 0;
}
