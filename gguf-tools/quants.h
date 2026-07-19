#ifndef DS4_QUANTS_H
#define DS4_QUANTS_H

/*
 * Narrow quantization API used by the DS4 GGUF writer.
 *
 * The enum values reuse the GGUF/GGML type-ID numbering so template metadata
 * can be copied without translation, but this is a PRIVATE fork namespace:
 * most IDs (MXFP8, MXFP4, CUTLASS_MXFP4, ...) do not exist upstream, and even
 * the shared ones are written with DS4-specific layouts.  Files produced by
 * these tools load ONLY in the DwarfStar engine, not in llama.cpp/GGML.
 * Only the formats used by the DS4 Flash quantization recipes are implemented
 * as output targets.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4Q_MAX_DIMS 4

typedef enum {
    DS4Q_TYPE_F32     = 0,
    DS4Q_TYPE_F16     = 1,
    DS4Q_TYPE_Q4_0    = 2,
    DS4Q_TYPE_Q4_1    = 3,
    DS4Q_TYPE_Q5_0    = 6,
    DS4Q_TYPE_Q5_1    = 7,
    DS4Q_TYPE_Q8_0    = 8,
    DS4Q_TYPE_Q8_1    = 9,
    DS4Q_TYPE_Q2_K    = 10,
    DS4Q_TYPE_Q3_K    = 11,
    DS4Q_TYPE_Q4_K    = 12,
    DS4Q_TYPE_Q5_K    = 13,
    DS4Q_TYPE_Q6_K    = 14,
    DS4Q_TYPE_Q8_K    = 15,
    DS4Q_TYPE_IQ2_XXS = 16,
    DS4Q_TYPE_IQ2_XS  = 17,
    DS4Q_TYPE_IQ3_XXS = 18,
    DS4Q_TYPE_IQ1_S   = 19,
    DS4Q_TYPE_IQ4_NL  = 20,
    DS4Q_TYPE_IQ3_S   = 21,
    DS4Q_TYPE_IQ2_S   = 22,
    DS4Q_TYPE_IQ4_XS  = 23,
    DS4Q_TYPE_I8      = 24,
    DS4Q_TYPE_I16     = 25,
    DS4Q_TYPE_I32     = 26,
    DS4Q_TYPE_I64     = 27,
    DS4Q_TYPE_F64     = 28,
    DS4Q_TYPE_IQ1_M   = 29,
    DS4Q_TYPE_BF16    = 30,
    DS4Q_TYPE_TQ1_0   = 34,
    DS4Q_TYPE_TQ2_0   = 35,
    DS4Q_TYPE_FP8_E4M3 = 38,   /* MXFP8: E4M3 + per-32 E8M0 block scale (33 B/32) */
    DS4Q_TYPE_MXFP4   = 39,
    DS4Q_TYPE_CUTLASS_MXFP4 = 40, /* per-expert CUTLASS B layout: E2M1 data
                                   * (K-major, byte-verbatim from source) +
                                   * Blackwell 128x4 swizzled E8M0 SF tile */
    DS4Q_TYPE_MXFP8_LT = 41,   /* pre-swizzled FP8 workhorse layout: the type-38
                                * E4M3 weights + E8M0 scales stored in the exact
                                * device layout the engine builds at first use
                                * (de-interleaved [in,out] col-major E4M3 data +
                                * mx_sfoff-swizzled E8M0 scale). Wire-matches the
                                * engine's DS4_TENSOR_MXFP8_LT. */
    DS4Q_TYPE_COUNT   = 42,
} ds4q_type;

static inline size_t ds4q_pad(size_t x, size_t n) {
    return ((x + n - 1) / n) * n;
}

const char *ds4q_type_name(ds4q_type type);
bool ds4q_can_quantize(ds4q_type type);
int64_t ds4q_block_size(ds4q_type type);
size_t ds4q_row_size(ds4q_type type, int64_t ne);
bool ds4q_requires_imatrix(ds4q_type type);
void ds4q_dequantize_iq2_xxs(const void *blocks, float *out, int64_t n);
void ds4q_dequantize_q2_k(const void *blocks, float *out, int64_t n);
void ds4q_dequantize_fp8_e4m3(const void *blocks, float *out, int64_t n);
void ds4q_dequantize_mxfp4(const void *blocks, float *out, int64_t n);

/* CUTLASS_MXFP4 (type 40) helpers.  One expert of shape [nrows=N(out),
 * ncols=K(in)] packs as data (N*K/2 bytes, the source E2M1 [N,K/2] array
 * verbatim) followed by the swizzled scale-factor tile (one E8M0 byte per
 * 32-elem K-block; rows padded to 128, K-blocks padded to 4).  Matches the
 * engine's DS4_TENSOR_CUTLASS_MXFP4 and the CUTLASS Sm1xx SFB atom layout;
 * validated byte-identical to the mxfp4_pack_source_cli splice output. */
size_t ds4q_cutlass_mxfp4_sf_bytes(int64_t nrows, int64_t ncols);
size_t ds4q_cutlass_mxfp4_bytes(int64_t nrows, int64_t ncols);
void ds4q_pack_cutlass_mxfp4(const uint8_t *e2m1, const uint8_t *e8m0,
                             void *dst, int64_t nrows, int64_t ncols);

/* MXFP8_LT (type 41) helpers.  One workhorse tensor of shape [nrows=out(N),
 * ncols=in(K)] packs as E4M3 data (nrows*ncols bytes, de-interleaved from the
 * type-38 [E8M0][32 x E4M3] blocks to [out,in] row-major) followed by the
 * mx_sfoff-swizzled E8M0 scale (rows padded to 128, K-blocks padded to 4).
 * Byte-identical to repack_tensor() in tools/mxfp8_prestore/repack_mxfp8_lt.py
 * and to the engine's DS4_TENSOR_MXFP8_LT device layout. */
size_t ds4q_mxfp8_lt_sf_bytes(int64_t nrows, int64_t ncols);
size_t ds4q_mxfp8_lt_bytes(int64_t nrows, int64_t ncols);
void ds4q_pack_mxfp8_lt(const uint8_t *fp8_blocks, void *dst,
                        int64_t nrows, int64_t ncols);

/* The canonical Blackwell 128x4 scale-factor swizzle, shared by the CUTLASS
 * MXFP4 and MXFP8_LT packers.  Byte-identical to the __device__ mx_sfoff in
 * src/cuda/ds4_cuda_matmul.cu and build_scale_dest_index() in
 * tools/mxfp8_prestore/repack_mxfp8_lt.py. */
size_t ds4q_mx_sfoff(int64_t row, int64_t kb, int64_t kbp);
void ds4q_quantize_init(ds4q_type type);
size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix);

float ds4q_f16_to_f32(uint16_t bits);
float ds4q_bf16_to_f32(uint16_t bits);
void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n);
void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n);

#endif
