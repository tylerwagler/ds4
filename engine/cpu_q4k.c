#include "ds4_engine_internal.h"



static void matvec_q4_k_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q4_k_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_q4_K *gate_row = (const block_q4_K *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &gate, gate_row, ctx->xq);

        const block_q4_K *up_row = (const block_q4_K *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &up, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}



static void matvec_q4_k_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q4_K || up_w->type != DS4_TENSOR_Q4_K)
        ds4_die("expected Q4_K expert tensors");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q4_k_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q4_K expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q4_K expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q4_K expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q4_k_mid_worker, &ctx);
}



static void matvec_q4_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q4_k_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_q4_K *br = (const block_q4_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
            ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}



static void matvec_q4_k_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != DS4_TENSOR_Q4_K) ds4_die("expected a Q4_K expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q4_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q4_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_q4_k_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_q4_k_accum_worker, &ctx);
}



void matvec_q4_k_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q4_k_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_q4_K *gate_row = (const block_q4_K *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_q4_K *up_row = (const block_q4_K *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &gate, gate_row, xq);
            ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &up, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}



void matvec_q4_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q4_k_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_q4_K *br = (const block_q4_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}



/* Dispatch: call the right gate/up mid builder based on tensor type. */
void matvec_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type == DS4_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_experts_mid_prequant(mid, m, gate_w, up_w, xq,
                                            selected, expert_weight, n_expert, clamp);
    } else if (gate_w->type == DS4_TENSOR_Q4_K) {
        matvec_q4_k_experts_mid_prequant(mid, m, gate_w, up_w, xq,
                                         selected, expert_weight, n_expert, clamp);
    } else {
        ds4_die("unsupported gate/up expert tensor type");
    }
}



/* Dispatch: call the right down-projection accumulator based on tensor type. */
void matvec_experts_down_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type == DS4_TENSOR_Q2_K) {
        matvec_q2_k_experts_accum_prequant(out, m, w, xq, selected, n_expert);
    } else if (w->type == DS4_TENSOR_Q4_K) {
        matvec_q4_k_experts_accum_prequant(out, m, w, xq, selected, n_expert);
    } else {
        ds4_die("unsupported down expert tensor type");
    }
}



/* Dispatch: single-expert gate/up pair for tracing. */
void matvec_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const ds4_model  *m,
        const ds4_tensor *w0,
        const ds4_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert) {
    if (w0->type == DS4_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_expert_pair_prequant(out0, out1, m, w0, w1, xq, expert);
    } else if (w0->type == DS4_TENSOR_Q4_K) {
        uint64_t in_dim0, out_dim0, rb0;
        uint64_t in_dim1, out_dim1, rb1;
        const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &rb0);
        const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &rb1);
        if (in_dim0 != in_dim1 || out_dim0 != out_dim1) ds4_die("paired Q4_K expert tensors do not match");

        for (uint64_t row = 0; row < out_dim0; row++) {
            const block_q4_K *gr = (const block_q4_K *)(base0 + row * rb0);
            ds4_vec_dot_q4_K_q8_K((int)in_dim0, &out0[row], gr, xq);
            const block_q4_K *ur = (const block_q4_K *)(base1 + row * rb1);
            ds4_vec_dot_q4_K_q8_K((int)in_dim0, &out1[row], ur, xq);
        }
    } else {
        ds4_die("unsupported gate/up expert tensor type");
    }
}



/* Dispatch: single-expert down projection for tracing. */
void matvec_expert_down(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type == DS4_TENSOR_Q2_K) {
        matvec_q2_k_expert(out, m, w, x, expert);
    } else if (w->type == DS4_TENSOR_Q4_K) {
        uint64_t in_dim, out_dim, row_bytes;
        const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
        if (in_dim % QK_K != 0) ds4_die("Q4_K expert row is not QK_K aligned");

        block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
        ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);

        for (uint64_t row = 0; row < out_dim; row++) {
            const block_q4_K *br = (const block_q4_K *)(base + row * row_bytes);
            ds4_vec_dot_q4_K_q8_K((int)in_dim, &out[row], br, xq);
        }
        free(xq);
    } else {
        ds4_die("unsupported down expert tensor type");
    }
}



static DS4_MAYBE_UNUSED void sum_down_pairs_worker(void *vctx, uint64_t row0, uint64_t row1) {
    sum_down_pairs_ctx *ctx = vctx;
    for (uint64_t idx = row0; idx < row1; idx++) {
        const uint32_t token = (uint32_t)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)token * ctx->out_dim;
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint64_t pair_id = (uint64_t)token * DS4_N_EXPERT_USED + slot;
            acc += ctx->down_pair[pair_id * ctx->out_dim + row];
        }
        ctx->moe[idx] = acc;
    }
}



/* =========================================================================
 * Hyper-Connection Transforms.
 * =========================================================================
 *
 * DeepSeek V4 Flash keeps four hyper-connection streams per token.  Before
 * attention or FFN, a learned small projection chooses how to reduce the HC
 * state into the 4096-wide sublayer input.  After the sublayer, the post and
 * combine weights expand the result back into the four-stream HC state.
 */

/* Decode the HC control projection.  The output contains pre weights, post
 * gates, and a small doubly-normalized combine matrix. */
void hc_split_sinkhorn_one(
        float       * out,
        const float * mix,
        const float * scale,
        const float * base,
        int           n_hc,
        int           iters,
        float         eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; i++) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];

    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = DS4_NEG_INF;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }

        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }

        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}

