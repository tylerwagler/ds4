#include "ds4_engine_internal.h"



/* Encode a full single-token decode step on Metal.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool gpu_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);

    bool ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    const uint32_t split_after_layers = gpu_graph_token_split_after_layers();

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = gpu_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = g->cur_hc;
        g->cur_hc = g->after_ffn_hc;
        g->after_ffn_hc = tmp;
        if (ok && allow_split_flush && split_after_layers != 0 && il + 1u == split_after_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}



ds4_gpu_tensor *gpu_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}



/* Upload prompt token ids for kernels that need token-aware hash routing. */
bool gpu_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}



/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool gpu_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens < 4) {
        return true;
    }
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            g->batch_attn_norm,
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (ok) {
        ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              kv_weight,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                             model,
                                             score_weight,
                                         DS4_N_EMBD,
                                         width,
                                         tail_hc,
                                         4) != 0;
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              g->batch_comp_kv,
                                                              g->batch_comp_sc,
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}



/* CPU fallback for seeding batched HC state from token embeddings.  It is still
 * useful for tiny speculative verifier batches where a separate GPU embedding
 * command buffer costs more than the small host write. */
static bool gpu_graph_upload_prompt_embeddings_hc_cpu(
        ds4_gpu_tensor   *out_hc,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t total = (uint64_t)n_tokens * hc_dim;
    float *hc = xmalloc((size_t)total * sizeof(hc[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));

    for (uint32_t t = 0; t < n_tokens; t++) {
        embed_token_f16(model, weights, prompt->v[pos0 + t], plain);
        float *dst = hc + (uint64_t)t * hc_dim;
        for (uint32_t h = 0; h < DS4_N_HC; h++) {
            memcpy(dst + (uint64_t)h * DS4_N_EMBD,
                   plain,
                   (size_t)DS4_N_EMBD * sizeof(plain[0]));
        }
    }

    const bool ok = ds4_gpu_tensor_write(out_hc, 0, hc, total * sizeof(hc[0])) != 0;
    free(plain);
    free(hc);
    return ok;
}



/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  Long prefill chunks use the Metal get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
bool gpu_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;

    uint32_t gpu_min = 512;
    const char *gpu_min_env = getenv("DS4_CUDA_GPU_BATCH_EMBED_MIN");
    if (gpu_min_env && gpu_min_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(gpu_min_env, &end, 10);
        if (end != gpu_min_env && v <= UINT32_MAX) gpu_min = (uint32_t)v;
    }

    if (tokens && n_tokens >= gpu_min) {
        return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                                tokens,
                                                model->map,
                                                model->size,
                                                weights->token_embd->abs_offset,
                                                (uint32_t)weights->token_embd->dim[1],
                                                n_tokens,
                                                DS4_N_EMBD,
                                                DS4_N_HC) != 0;
    }

    return gpu_graph_upload_prompt_embeddings_hc_cpu(out_hc,
                                                       model,
                                                       weights,
                                                       prompt,
                                                       pos0,
                                                       n_tokens);
}



bool gpu_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (g && g->ssd_streaming) return true;
    if (warmed) return true;
    if (getenv("DS4_CUDA_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;

    /*
     * The first batched F16 matmul can pay Metal's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = gpu_graph_matmul_plain_tensor(g->batch_hc_mix,
                                             model,
                                             weights->layer[0].hc_attn_fn,
                                         hc_dim,
                                         mix_hc,
                                         g->batch_flat_hc,
                                         n_tokens) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "ds4: Metal prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}



/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: metal indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}



static bool gpu_graph_profile_layer_env_match(const char *env_name, uint32_t il) {
    const char *layer_env = getenv(env_name);
    if (!layer_env || !layer_env[0]) return true;

    char *end = NULL;
    const unsigned long layer = strtoul(layer_env, &end, 10);
    return end != layer_env &&
           *end == '\0' &&
           layer <= UINT32_MAX &&
           (uint32_t)layer == il;
}



static bool gpu_graph_layer_stage_profile_enabled(uint32_t il) {
    return getenv("DS4_CUDA_LAYER_STAGE_PROFILE") != NULL &&
           gpu_graph_profile_layer_env_match("DS4_CUDA_LAYER_STAGE_PROFILE_LAYER", il);
}



bool gpu_graph_decode_stage_profile_enabled(uint32_t il) {
    return getenv("DS4_CUDA_DECODE_STAGE_PROFILE") != NULL &&
           gpu_graph_profile_layer_env_match("DS4_CUDA_DECODE_STAGE_PROFILE_LAYER", il);
}



/* Optional prefill stage profiler. It intentionally ends the current Metal
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            part,
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}



static bool gpu_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: metal Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}



bool gpu_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    const bool zero_prefix = pos0 == 0;
    const bool index_stage_profile = getenv("DS4_CUDA_INDEXER_STAGE_PROFILE") != NULL;
    const bool layer_stage_profile = gpu_graph_layer_stage_profile_enabled(il);
    const bool q_stage_profile = getenv("DS4_CUDA_Q_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
    double q_stage_t0 = q_stage_profile ? now_sec() : 0.0;
#define DS4_CUDA_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define DS4_CUDA_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = gpu_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    uint32_t *comp_counts = compressed ? xcalloc(n_tokens, sizeof(comp_counts[0])) : NULL;
    uint32_t *index_counts = ratio == 4 ? xcalloc(n_tokens, sizeof(index_counts[0])) : NULL;
    const bool qkv_rms_fused = !gpu_graph_use_reference_qkv_norm();
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_attn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            g->batch_after_attn_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    const bool fuse_hc_norm = DS4_N_HC == 4 &&
                              !gpu_graph_use_reference_hc_decode() &&
                              gpu_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(hc_mix_view,
                                              model,
                                              layer->hc_attn_fn,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (gpu_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            g->batch_cur_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 g->batch_attn_norm,
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->batch_cur_hc,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_attn_scale->abs_offset,
                                                                 layer->hc_attn_base->abs_offset,
                                                                 layer->attn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_cur_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_attn_scale->abs_offset,
                                                            layer->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->batch_attn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                  g->batch_attn_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->attn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->batch_attn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("norm");
    DS4_CUDA_PROFILE_Q_STAGE("pre_q");
    if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_q_a",
                                                      il,
                                                      pos0,
                                                      g->batch_qr,
                                                      model,
                                                      layer->attn_q_a,
                                                      DS4_N_EMBD,
                                                      q_rank,
                                                      g->batch_attn_norm,
                                                      n_tokens);
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->batch_qr,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    DS4_CUDA_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          g->batch_kv_raw,
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          g->batch_attn_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->batch_qr_norm,
                                                             g->batch_qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->batch_kv,
                                                             g->batch_kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             n_tokens,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_qr_norm,
                                                           g->batch_qr,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->batch_qr_norm,
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (qkv_rms_fused && ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_CUDA_PROFILE_Q_STAGE("q_a_norm");
    {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_q_b",
                                                          il,
                                                          pos0,
                                                          g->batch_q,
                                                          model,
                                                          layer->attn_q_b,
                                                          q_rank,
                                                          q_dim,
                                                          g->batch_qr_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_tensor("Qraw", g->batch_q,
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_CUDA_PROFILE_Q_STAGE("q_b");
        const bool prefill_q_norm_debug = gpu_graph_debug_wants("Qnorm", il, pos0);
        bool prefill_q_norm_rope_fused = false;
        if (ok && !prefill_q_norm_debug) {
            prefill_q_norm_rope_fused =
                ds4_gpu_head_rms_norm_rope_tail_tensor(g->batch_q,
                                                       n_tokens,
                                                       DS4_N_HEAD,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT,
                                                       pos0,
                                                       compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                       false,
                                                       freq_base,
                                                       freq_scale,
                                                       ext_factor,
                                                       attn_factor,
                                                       DS4_ROPE_YARN_BETA_FAST,
                                                       DS4_ROPE_YARN_BETA_SLOW,
                                                       DS4_RMS_EPS) != 0;
        }
        if (!prefill_q_norm_rope_fused) {
            if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->batch_q,
                                                        n_tokens,
                                                        DS4_N_HEAD,
                                                        DS4_N_HEAD_DIM,
                                                        DS4_RMS_EPS) != 0;
            if (ok) {
                gpu_graph_debug_dump_tensor("Qnorm", g->batch_q,
                                              (uint64_t)n_tokens * q_dim, il, pos0);
            }
            DS4_CUDA_PROFILE_Q_STAGE("head_norm");
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                                    n_tokens,
                                                    DS4_N_HEAD,
                                                    DS4_N_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
        } else {
            DS4_CUDA_PROFILE_Q_STAGE("head_norm");
        }
        if (ok) {
            gpu_graph_debug_dump_tensor("Qcur", g->batch_q,
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_CUDA_PROFILE_Q_STAGE("rope");
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          g->batch_kv_raw,
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          g->batch_attn_norm,
                                                          n_tokens);
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->batch_kv_raw,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_kv,
                                                           g->batch_kv_raw,
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVnorm", g->batch_kv,
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_kv,
                                            n_tokens,
                                            DS4_N_HEAD_KV,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(g->batch_kv,
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->batch_kv,
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("kv_path");
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    g->batch_kv,
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM,
                                                                    (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                          model->map,
                                                          model->size,
                                                          layer->attn_sinks->abs_offset,
                                                          g->batch_q,
                                                          g->batch_kv,
                                                          n_tokens,
                                                          g->raw_window,
                                                          DS4_N_HEAD,
                                                          DS4_N_HEAD_DIM,
                                                          0 /* batch_kv is f32 */) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 g->batch_kv,
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM,
                                                 (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
        if (ok && !gpu_graph_raw_f16_enabled()) {
            /* diag-only dump; the dumper reads f32 and would misinterpret a
             * __half ring — skip it under DS4_RAW_F16. */
            gpu_graph_debug_dump_tensor("raw_cache",
                                          g->layer_raw_cache[il],
                                          (uint64_t)n_raw * DS4_N_HEAD_DIM,
                                          il,
                                          pos0);
        }
        if (ok) {
            ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(g->batch_heads,
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   g->batch_q,
                                                                   g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   g->raw_cap,
                                                                   raw_start,
                                                                    g->raw_window,
                                                                    DS4_N_HEAD,
                                                                    DS4_N_HEAD_DIM,
                                                                    0,
                                                                    (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: Metal layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) {
            ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              layer->attn_compressor_kv,
                                             DS4_N_EMBD,
                                             comp_width,
                                             g->batch_attn_norm,
                                             n_tokens) != 0;
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                              model,
                                              layer->attn_compressor_gate,
                                                     DS4_N_EMBD,
                                                     comp_width,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
        }
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              g->batch_comp_kv,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) gpu_graph_debug_dump_tensor("attn_comp_score_raw",
                                              g->batch_comp_sc,
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        /* Stage-B save: keep this batch's per-position compressor projections so
         * a partial spec accept can roll the pool state forward without a
         * transformer replay. Must run here -- the indexer section below reuses
         * batch_comp_kv/sc. */
        if (ok && g->spec_comp_save_n && g->spec_comp_kv_save[il]) {
            uint32_t sn = g->spec_comp_save_n;
            if (sn > n_tokens) sn = n_tokens;
            if (sn > 17u) sn = 17u;
            const uint64_t sb = (uint64_t)sn * comp_width * sizeof(float);
            ok = ds4_gpu_tensor_copy(g->spec_comp_kv_save[il], 0, g->batch_comp_kv, 0, sb) != 0 &&
                 ds4_gpu_tensor_copy(g->spec_comp_sc_save[il], 0, g->batch_comp_sc, 0, sb) != 0;
        }
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && (DS4_GPU_ATTN_COMP_CACHE_F16 || gpu_graph_attn_mx_enabled() ||
                       gpu_graph_attn_pack_enabled()) &&
                n_comp > g->attn_comp_stage_cap) {
                fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                ok = false;
            }
            ds4_gpu_tensor *attn_comp_target = NULL;
            if (ok) {
                attn_comp_target = gpu_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                ok = attn_comp_target != NULL &&
                     ds4_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         g->batch_comp_kv,
                                                         g->batch_comp_sc,
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         /* Under DS4_ATTN_PACK the commit's pack-store kernel
                                                          * is the single fp8 quantizer (same recipe); a second
                                                          * quantize is NOT bit-idempotent for borderline block
                                                          * amax (scale can shift, re-rounding small values). */
                                                         !gpu_graph_attn_pack_enabled(),
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS) != 0;
                if (ok && n_comp != 0) {
                    ok = gpu_graph_commit_attn_comp_stage(g, il, 0, n_comp);
                }
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                    gpu_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)n_comp * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                gpu_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                gpu_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
            gpu_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
            const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && (DS4_GPU_ATTN_COMP_CACHE_F16 || gpu_graph_attn_mx_enabled() ||
                           gpu_graph_attn_pack_enabled()) &&
                    comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *attn_comp_target =
                    ok ? gpu_graph_attn_comp_prefill_target(g, il, comp_before, comp_chunk) : NULL;
                if (ok && !attn_comp_target) ok = false;
                if (ok && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            /* pack: quantize once, in the commit (see above) */
                            !gpu_graph_attn_pack_enabled(),
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                } else if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            g->batch_comp_kv,
                            g->batch_comp_sc,
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            /* pack: quantize once, in the commit (see above) */
                            !gpu_graph_attn_pack_enabled(),
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                }
                if (ok && comp_chunk != 0) {
                    ok = gpu_graph_commit_attn_comp_stage(g, il, comp_before, comp_chunk);
                }
                if (ok && ratio == 4) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                    gpu_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)comp_chunk * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("attn_state_kv",
                                                  g->layer_attn_state_kv[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("attn_state_score",
                                                  g->layer_attn_state_score[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
                gpu_graph_attn_comp_prefill_target_free(attn_comp_target);
            } else {
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = pos0 + t;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    ds4_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, comp_width);
                    ds4_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, comp_width);
                    const uint32_t comp_row = g->layer_n_comp[il];
                    ok = kv_view && sc_view &&
                         ds4_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            g->layer_attn_state_kv[il],
                                                            g->layer_attn_state_score[il],
                                                            gpu_graph_attn_comp_update_target(g, il),
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            DS4_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            gpu_graph_attn_comp_update_row(comp_row),
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS) != 0;
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = gpu_graph_attn_comp_row_view(g, il, comp_row);
                        if (gpu_graph_attn_pack_enabled()) {
                            /* comp_row_view aliases the f32 stage; commit below
                             * quantizes+packs and roundtrips the stage in place
                             * — dump after commit to match f32-mode values. */
                            ok = comp_row_view != NULL;
                        } else if (gpu_graph_attn_mx_enabled()) {
                            /* comp_row_view aliases the f32 stage; commit packs
                             * it to MXFP8 below. */
                            ok = comp_row_view != NULL;
                        } else if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
                            ds4_gpu_tensor *packed_dst = ds4_gpu_tensor_view(
                                g->layer_attn_comp_cache[il],
                                (uint64_t)comp_row * DS4_FP8_KV_ROWBYTES(DS4_N_HEAD_DIM),
                                (uint64_t)DS4_N_HEAD_DIM);
                            ds4_gpu_tensor *scales_dst = ds4_gpu_tensor_view(
                                g->layer_attn_comp_cache[il],
                                (uint64_t)comp_row * DS4_FP8_KV_ROWBYTES(DS4_N_HEAD_DIM) + DS4_N_HEAD_DIM,
                                (uint64_t)DS4_FP8_KV_NBLK(DS4_N_HEAD_DIM) * sizeof(float));
                            ok = comp_row_view && packed_dst && scales_dst &&
                                 ds4_gpu_dsv4_fp8_kv_pack_tensor(comp_row_view, packed_dst, scales_dst,
                                                                 1, DS4_N_HEAD_DIM) != 0;
                            if (packed_dst) ds4_gpu_tensor_free(packed_dst);
                            if (scales_dst) ds4_gpu_tensor_free(scales_dst);
                        } else {
                            ok = comp_row_view &&
                                 ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view,
                                                                      1,
                                                                      DS4_N_HEAD_DIM,
                                                                      DS4_N_ROT) != 0;
                        }
                        if (ok && !gpu_graph_attn_pack_enabled()) {
                            gpu_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          DS4_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        if (ok) ok = gpu_graph_commit_attn_comp_stage(g, il, comp_row, 1);
                        if (ok && gpu_graph_attn_pack_enabled()) {
                            gpu_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          DS4_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        ds4_gpu_tensor_free(comp_row_view);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    if (ok && t == 0) ok = gpu_graph_capture_prefix1_attn_state(g, il);
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }
        DS4_CUDA_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: Metal layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) {
                ok = gpu_graph_matmul_plain_tensor(g->batch_comp_kv,
                                              model,
                                              layer->indexer_compressor_kv,
                                                 DS4_N_EMBD,
                                                 index_width,
                                                 g->batch_attn_norm,
                                                 n_tokens) != 0;
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_comp_sc,
                                              model,
                                              layer->indexer_compressor_gate,
                                                         DS4_N_EMBD,
                                                         index_width,
                                                         g->batch_attn_norm,
                                                         n_tokens) != 0;
            }
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  g->batch_comp_kv,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) gpu_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  g->batch_comp_sc,
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            /* Stage-B save (indexer variant; see the attn compressor hook). */
            if (ok && g->spec_comp_save_n && g->spec_icomp_kv_save[il]) {
                uint32_t sn = g->spec_comp_save_n;
                if (sn > n_tokens) sn = n_tokens;
                if (sn > 17u) sn = 17u;
                const uint64_t sb = (uint64_t)sn * index_width * sizeof(float);
                ok = ds4_gpu_tensor_copy(g->spec_icomp_kv_save[il], 0, g->batch_comp_kv, 0, sb) != 0 &&
                     ds4_gpu_tensor_copy(g->spec_icomp_sc_save[il], 0, g->batch_comp_sc, 0, sb) != 0;
            }
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_indexer_q,
                                                          model,
                                                          layer->indexer_attn_q_b,
                                                          q_rank,
                                                          (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                          g->batch_qr_norm,
                                                          n_tokens);
            if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_indexer_q,
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                          n_tokens * DS4_N_INDEXER_HEAD,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_indexer_weights,
                                              model,
                                              layer->indexer_proj,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     g->batch_attn_norm,
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                const int idx_fp4 = gpu_graph_idx_fp4_enabled();
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(idx_fp4 ? g->idx_comp_stage
                                                                   : g->layer_index_comp_cache[il],
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             g->batch_comp_kv,
                                                             g->batch_comp_sc,
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = idx_fp4
                        ? ds4_gpu_dsv4_indexer_qat_pack_tensor(g->idx_comp_stage,
                                                                g->layer_index_comp_cache[il],
                                                                0,
                                                                n_comp,
                                                                DS4_N_INDEXER_HEAD_DIM) != 0
                        : ds4_gpu_dsv4_indexer_qat_tensor(g->layer_index_comp_cache[il],
                                                          n_comp,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                        gpu_graph_debug_dump_tensor("indexer_KVcompress",
                                                      idx_fp4 ? g->idx_comp_stage
                                                              : g->layer_index_comp_cache[il],
                                                      (uint64_t)n_comp * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    gpu_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    gpu_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                const bool aligned_chunk = (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    const int idx_fp4 = gpu_graph_idx_fp4_enabled();
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                idx_fp4 ? g->idx_comp_stage : g->layer_index_comp_cache[il],
                                (uint64_t)index_before * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                g->batch_comp_kv,
                                g->batch_comp_sc,
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = idx_fp4
                            ? ds4_gpu_dsv4_indexer_qat_pack_tensor(index_view,
                                                                    g->layer_index_comp_cache[il],
                                                                    index_before,
                                                                    index_chunk,
                                                                    DS4_N_INDEXER_HEAD_DIM) != 0
                            : ds4_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    if (ok) {
                        ok = gpu_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                        gpu_graph_debug_dump_tensor("indexer_KVcompress",
                                                      index_view,
                                                      (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_kv",
                                                      g->layer_index_state_kv[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                        gpu_graph_debug_dump_tensor("indexer_state_score",
                                                      g->layer_index_state_score[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                    }
                    ds4_gpu_tensor_free(index_view);
                } else {
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = pos0 + t;
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        ds4_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->batch_comp_kv, t, index_width);
                        ds4_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->batch_comp_sc, t, index_width);
                        const uint32_t index_row = g->layer_n_index_comp[il];
                        const int idx_fp4 = gpu_graph_idx_fp4_enabled();
                        ok = kv_view && sc_view &&
                             ds4_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                g->layer_index_state_kv[il],
                                                                g->layer_index_state_score[il],
                                                                idx_fp4 ? g->idx_comp_stage
                                                                        : g->layer_index_comp_cache[il],
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                index_row,
                                                                DS4_N_ROT,
                                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                DS4_ROPE_YARN_BETA_FAST,
                                                                DS4_ROPE_YARN_BETA_SLOW,
                                                                DS4_RMS_EPS) != 0;
                        if (ok && emit) {
                            ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                                    idx_fp4 ? g->idx_comp_stage : g->layer_index_comp_cache[il],
                                    (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                            if (!index_row_view) {
                                ok = false;
                            } else if (idx_fp4) {
                                ok = ds4_gpu_dsv4_indexer_qat_pack_tensor(index_row_view,
                                                                           g->layer_index_comp_cache[il],
                                                                           index_row,
                                                                           1,
                                                                           DS4_N_INDEXER_HEAD_DIM) != 0;
                                ds4_gpu_tensor_free(index_row_view);
                            } else {
                                ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD_DIM) != 0;
                                ds4_gpu_tensor_free(index_row_view);
                            }
                        }
                        if (ok && emit) g->layer_n_index_comp[il]++;
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        if (ok && t == 0) ok = gpu_graph_capture_prefix1_index_state(g, il);
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) DS4_CUDA_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = gpu_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;
            double index_stage_t0 = 0.0;

            ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     g->batch_kv,
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM,
                                                     (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                ok = ds4_gpu_indexer_scores_decode_batch_tensor(g->indexer_scores,
                                                                  g->batch_indexer_q,
                                                                  g->batch_indexer_weights,
                                                                  g->layer_index_comp_cache[il],
                                                                  n_comp,
                                                                  n_tokens,
                                                                  pos0,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  ratio,
                                                                  index_scale) != 0;
                if (ok && index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("score",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    gpu_graph_debug_dump_tensor("indexer_scores",
                                                  g->indexer_scores,
                                                  (uint64_t)n_comp * n_tokens,
                                                  il,
                                                  pos0);
                }
                if (ok) {
                    ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       n_comp,
                                                       n_tokens,
                                                       DS4_N_INDEXER_TOP_K) != 0;
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary("topk",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        gpu_graph_debug_dump_i32_tensor("indexer_topk",
                                                          g->comp_selected,
                                                          (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                          il,
                                                          pos0);
                    }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = 1;
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              g->batch_q,
                                                                              g->layer_raw_cache[il],
                                                                              gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                              gpu_graph_attn_comp_read_is_f16(), gpu_graph_attn_comp_read_is_fp8(),
                                                                              0 /* shadow is f32 */,
                                                                              g->comp_selected,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              DS4_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM,
                                                                              (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                    if (ok && index_stage_profile) {
                        ok = gpu_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(g->batch_heads,
                                                                             model->map,
                                                                             model->size,
                                                                             layer->attn_sinks->abs_offset,
                                                                             g->batch_q,
                                                                             g->layer_raw_cache[il],
                                                                             gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                             gpu_graph_attn_comp_read_is_f16(), gpu_graph_attn_comp_read_is_fp8(),
                                                                             0 /* shadow is f32 */,
                                                                             use_comp_mask ? g->comp_mask : NULL,
                                                                             use_comp_mask,
                                                                             n_tokens,
                                                                             pos0,
                                                                             n_raw,
                                                                             g->raw_cap,
                                                                             raw_start,
                                                                             n_comp,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM,
                                                                              0,
                                                                              (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                }
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && zero_prefix && topk_prefill_needed && n_comp != 0) {
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            ok = ds4_gpu_indexer_scores_prefill_tensor(g->indexer_scores,
                                                         g->batch_indexer_q,
                                                         g->batch_indexer_weights,
                                                         g->layer_index_comp_cache[il],
                                                         n_comp,
                                                         n_tokens,
                                                         DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM,
                                                         ratio,
                                                         index_scale) != 0;
            if (ok && index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary("score",
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            if (ok) {
                gpu_graph_debug_dump_tensor("indexer_scores",
                                              g->indexer_scores,
                                              (uint64_t)n_comp * n_tokens,
                                              il,
                                              pos0);
            }
            if (ok) {
                ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                   g->indexer_scores,
                                                   n_comp,
                                                   n_tokens,
                                                   DS4_N_INDEXER_TOP_K) != 0;
                if (ok && index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("topk",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    gpu_graph_debug_dump_i32_tensor("indexer_topk",
                                                      g->comp_selected,
                                                      (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                      il,
                                                      pos0);
                }
            }
            if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(g->batch_heads,
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          g->batch_q,
                                                                          g->layer_raw_cache[il],
                                                                          gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                          gpu_graph_attn_comp_read_is_f16(), gpu_graph_attn_comp_read_is_fp8(),
                                                                          0 /* shadow is f32 */,
                                                                          g->comp_selected,
                                                                          n_tokens,
                                                                          pos0,
                                                                          n_tokens,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM,
                                                                          (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                if (ok && index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && zero_prefix && !topk_prefill_needed && n_comp != 0) {
            ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(g->batch_heads,
                                                                       model->map,
                                                                       model->size,
                                                                       layer->attn_sinks->abs_offset,
                                                                       g->batch_q,
                                                                       g->batch_kv,
                                                                       gpu_graph_attn_comp_read_cache(g, il, n_comp),
                                                                       gpu_graph_attn_comp_read_is_f16(), gpu_graph_attn_comp_read_is_fp8(),
                                                                       n_tokens,
                                                                       n_comp,
                                                                       g->raw_window,
                                                                       ratio,
                                                                       DS4_N_HEAD,
                                                                       DS4_N_HEAD_DIM,
                                                                       0 /* batch_kv is f32 */) != 0;
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(g->batch_heads,
                                                              model->map,
                                                              model->size,
                                                              layer->attn_sinks->abs_offset,
                                                              g->batch_q,
                                                              g->batch_kv,
                                                              raw_prefix_tokens,
                                                              g->raw_window,
                                                              DS4_N_HEAD,
                                                              DS4_N_HEAD_DIM,
                                                              0 /* batch_kv is f32 */) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                ds4_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_q, t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = gpu_graph_tensor_row_view(
                            g->batch_indexer_weights, t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                            indexer_q_view,
                                                            indexer_w_view,
                                                            g->layer_index_comp_cache[il],
                                                            cur_index,
                                                            DS4_N_INDEXER_HEAD,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            index_scale) != 0 &&
                         ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                       g->indexer_scores,
                                                       cur_index,
                                                       1,
                                                       DS4_N_INDEXER_TOP_K) != 0 &&
                         ds4_gpu_dsv4_topk_mask_tensor(g->comp_mask,
                                                         g->comp_selected,
                                                         cur_index,
                                                         1,
                                                         DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = g->comp_mask;
                        n_selected = DS4_N_INDEXER_TOP_K < cur_index
                            ? DS4_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = gpu_graph_tensor_row_view(g->batch_q, t, q_dim);
                ds4_gpu_tensor *kv_cache_view = gpu_graph_tensor_row_view(g->batch_kv, t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = gpu_graph_tensor_row_view(g->batch_heads, t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM,
                                                       (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              gpu_graph_attn_comp_read_cache(g, il, cur_comp),
                                                                              gpu_graph_attn_comp_read_is_f16(), gpu_graph_attn_comp_read_is_fp8(),
                                                                              0 /* shadow is f32 */,
                                                                              g->comp_selected,
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM,
                                                                              (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                } else if (ok) {
                    ok = ds4_gpu_attention_decode_heads_tensor(heads_view,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->attn_sinks->abs_offset,
                                                                 q_view,
                                                                 g->layer_raw_cache[il],
                                                                 n_raw,
                                                                 g->raw_cap,
                                                                 raw_start,
                                                                 cur_comp ? gpu_graph_attn_comp_read_cache(g, il, cur_comp) : NULL,
                                                                 gpu_graph_attn_comp_read_is_f16(), gpu_graph_attn_comp_read_is_fp8(),
                                                                 0 /* shadow is f32 */,
                                                                 cur_comp,
                                                                 comp_mask,
                                                                 n_selected,
                                                                 DS4_N_HEAD,
                                                                 DS4_N_HEAD_DIM,
                                                                 (uint32_t)gpu_graph_raw_f16_enabled()) != 0;
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("attention");

    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_heads,
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->batch_heads,
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("inv_rope");
    if (ok) {
        ok = ds4_gpu_attention_output_batch_tensor(g->batch_attn_out,
                                                   g->batch_attn_low,
                                                   model->map,
                                                   model->size,
                                                   layer->attn_output_a->abs_offset,
                                                   layer->attn_output_b->abs_offset,
                                                   group_dim,
                                                   rank,
                                                   n_groups,
                                                   DS4_N_EMBD,
                                                   g->batch_heads,
                                                   n_tokens) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->batch_attn_low,
                                      (uint64_t)n_tokens * n_groups * rank,
                                      il,
                                      pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("output_proj");
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->batch_attn_out, il, n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                            g->batch_attn_out,
                                            g->batch_cur_hc,
                                            hc_split_view,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_post", g->batch_after_attn_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_CUDA_PROFILE_ATTN_STAGE("hc_post");
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    free(index_counts);
    free(comp_counts);
#undef DS4_CUDA_PROFILE_ATTN_STAGE
#undef DS4_CUDA_PROFILE_Q_STAGE
    return ok;
}



/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post. */
bool gpu_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    uint64_t gate_expert_bytes = 0, gate_row_bytes = 0;
    uint64_t down_expert_bytes = 0, down_row_bytes = 0;
    if (!routed_expert_gate_down_layout(layer->ffn_gate_exps, layer->ffn_down_exps,
                                        &gate_expert_bytes, &gate_row_bytes,
                                        &down_expert_bytes, &down_row_bytes)) {
        return false;
    }
    const bool layer_stage_profile = gpu_graph_layer_stage_profile_enabled(il);
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_CUDA_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            g->batch_next_hc, 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    const bool fuse_hc_norm = DS4_N_HC == 4 &&
                              !gpu_graph_use_reference_hc_decode() &&
                              gpu_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_after_attn_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(hc_mix_view,
                                              model,
                                              layer->hc_ffn_fn,
                                             hc_dim,
                                             mix_hc,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (gpu_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            g->batch_after_attn_hc,
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(ffn_cur_view,
                                                                 g->batch_ffn_norm,
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->batch_after_attn_hc,
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_ffn_scale->abs_offset,
                                                                 layer->hc_ffn_base->abs_offset,
                                                                 layer->ffn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->batch_after_attn_hc,
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->batch_ffn_cur,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_CUDA_PROFILE_FFN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                  g->batch_ffn_cur,
                                                  model->map,
                                                  model->size,
                                                  layer->ffn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->batch_ffn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_CUDA_PROFILE_FFN_STAGE("norm");
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->batch_router_logits,
                                              model,
                                              layer->ffn_gate_inp,
                                             DS4_N_EMBD,
                                             DS4_N_EXPERT,
                                             g->batch_ffn_norm,
                                             n_tokens) != 0;

    if (ok) ok = ds4_gpu_router_select_batch_tensor(g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      g->batch_router_logits,
                                                      g->prefill_tokens,
                                                      DS4_N_EXPERT,
                                                      DS4_N_EXPERT_USED,
                                                      DS4_EXPERT_WEIGHT_SCALE,
                                                      n_tokens) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->batch_router_logits,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->batch_router_probs,
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->batch_router_selected,
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->batch_router_weights,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_CUDA_PROFILE_FFN_STAGE("router");

    if (ok) {
        ok = gpu_graph_cuda_stream_prefill_batch_selected_load(g,
                                                                 model,
                                                                 layer,
                                                                 il,
                                                                 n_tokens,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes);
    }


    const bool selected_readahead_shared =
        gpu_graph_stream_prefill_selected_readahead_shared_enabled(g)
        ;
    if (ok &&
        gpu_graph_stream_prefill_selected_readahead_enabled(g) &&
        !selected_readahead_shared) {
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = gpu_graph_stream_readahead_selected_experts_from_gpu(g,
                                                                        model,
                                                                        layer,
                                                                        il,
                                                                        n_tokens,
                                                                        gate_expert_bytes,
                                                                        down_expert_bytes) &&
                 ds4_gpu_begin_commands() != 0;
        }
    }

    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos0);

#define DS4_CUDA_ENCODE_PREFILL_SHARED_EXPERT() do { \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_gate", \
                                                          il, \
                                                          pos0, \
                                                          g->batch_shared_gate, \
                                                          model, \
                                                          layer->ffn_gate_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_up", \
                                                          il, \
                                                          pos0, \
                                                          g->batch_shared_up, \
                                                          model, \
                                                          layer->ffn_up_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
        DS4_CUDA_PROFILE_FFN_STAGE("shared_gate_up"); \
        if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid, \
                                             g->batch_shared_gate, \
                                             g->batch_shared_up, \
                                             (uint32_t)((uint64_t)n_tokens * shared_dim), \
                                             DS4_SWIGLU_CLAMP_EXP, \
                                             1.0f) != 0; \
        if (ok) ok = gpu_graph_matmul_mxfp8_named_tensor("shared_down", \
                                                                              il, \
                                                                              pos0, \
                                                                              g->batch_shared_out, \
                                                                              model, \
                                                                              layer->ffn_down_shexp, \
                                                                              shared_dim, \
                                                                              DS4_N_EMBD, \
                                                                              g->batch_shared_mid, \
                                                                              n_tokens); \
        DS4_CUDA_PROFILE_FFN_STAGE("shared_down"); \
        if (ok) { \
            gpu_graph_debug_dump_tensor("ffn_shexp", g->batch_shared_out, \
                                          (uint64_t)n_tokens * DS4_N_EMBD, il, pos0); \
        } \
    } while (0)

    bool shared_done = false;
    if (ok && selected_readahead_shared) {
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = gpu_graph_stream_readahead_selected_experts_from_gpu(g,
                                                                        model,
                                                                        layer,
                                                                        il,
                                                                        n_tokens,
                                                                        gate_expert_bytes,
                                                                        down_expert_bytes) &&
                 ds4_gpu_begin_commands() != 0;
        }
        if (ok) {
            DS4_CUDA_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        if (ok) {
            if (ds4_gpu_end_commands() == 0) {
                ok = false;
            } else {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
    }

    if (ok &&
        !shared_done &&
        (gpu_graph_stream_prefill_selected_pagein_enabled(g) ||
         gpu_graph_stream_prefill_selected_madvise_enabled(g))) {
        gpu_graph_stream_pagein_job pagein_job;
        memset(&pagein_job, 0, sizeof(pagein_job));
        bool pagein_commands_open = false;
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = gpu_graph_stream_prefill_selected_pagein_start(g,
                                                                  model,
                                                                  layer,
                                                                  il,
                                                                  n_tokens,
                                                                  gate_expert_bytes,
                                                                  down_expert_bytes,
                                                                  &pagein_job);
        }
        if (ok) {
            if (ds4_gpu_begin_commands() == 0) {
                ok = false;
            } else {
                pagein_commands_open = true;
            }
        }
        if (ok) {
            DS4_CUDA_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        if (pagein_commands_open) {
            if (ds4_gpu_end_commands() == 0) ok = false;
        }
        if (!gpu_graph_stream_prefill_selected_pagein_join(&pagein_job)) {
            ok = false;
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
    }


    if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(g->batch_routed_out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               g->batch_routed_mid,
                                               g->batch_routed_down,
                                               tensor_map_base(model, layer->ffn_gate_exps),
                                               tensor_map_size(model, layer->ffn_gate_exps),
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               g->batch_router_selected,
                                               g->batch_router_weights,
                                               ds4_layer_n_expert(il),
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               g->batch_ffn_norm,
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->batch_routed_gate,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->batch_routed_up,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
            gpu_graph_debug_dump_f16_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                              routed_mid_elems, il, pos0);
        } else {
            gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->batch_routed_mid,
                                          routed_mid_elems, il, pos0);
        }
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->batch_routed_down,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->batch_routed_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_CUDA_PROFILE_FFN_STAGE("routed_moe");
    if (!shared_done) {
        DS4_CUDA_ENCODE_PREFILL_SHARED_EXPERT();
    }
#undef DS4_CUDA_ENCODE_PREFILL_SHARED_EXPERT

    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(g->batch_ffn_out,
                                  g->batch_shared_out,
                                  g->batch_routed_out,
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->batch_ffn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->batch_ffn_out, il, n_tokens);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              g->batch_ffn_out,
                                              g->batch_after_attn_hc,
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  g->batch_routed_out,
                                                  g->batch_shared_out,
                                                  g->batch_after_attn_hc,
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_post", g->batch_next_hc,
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_CUDA_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_CUDA_PROFILE_FFN_STAGE
    return ok;
}



/* Encode one complete layer for prefill by chaining attention and FFN batches. */
bool gpu_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = gpu_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    if (!ok) {
        fprintf(stderr, "ds4: gpu layer %u attention batch encode failed\n", il);
    }
    if (ok) {
        ok = gpu_graph_encode_layer_ffn_batch(g, model, layer, il, pos0, n_tokens);
        if (!ok) {
            fprintf(stderr, "ds4: gpu layer %u ffn batch encode failed\n", il);
        }
    }
    if (ok) {
        ds4_gpu_tensor *tmp = g->batch_cur_hc;
        g->batch_cur_hc = g->batch_next_hc;
        g->batch_next_hc = tmp;
    }
    /* Fused spec loop (P2): when armed, capture the drafter's anchor hidden for
     * every batch position at the anchor layers, so the last-accepted position's
     * hidden is available without a replay decode. Off (0) during prefill and
     * plain decode. */
    if (ok && g->dspark_capture_batch_n) {
        for (int slot = 0; slot < 3; slot++) {
            if (il != g->dspark_target_layer_ids[slot]) continue;
            if (!g->dspark_target_h_batch[slot]) break;
            uint32_t cap_n = g->dspark_capture_batch_n;
            if (cap_n > n_tokens) cap_n = n_tokens;
            if (!ds4_gpu_dspark_hc_mean_reduce_batch(g->dspark_target_h_batch[slot],
                                                     g->batch_cur_hc,
                                                     DS4_N_EMBD, DS4_N_HC, cap_n)) {
                ok = false;
            }
            break;
        }
    }
    /* Bulk prefill capture for drafter retraining (DS4_DSPARK_PREFILL_DUMP):
     * same reduction as the verify capture above, but over EVERY chunk position
     * into the per-layer bulk buffers. Armed only by the prefill path. */
    if (ok && g->dspark_bulk_n) {
        for (int slot = 0; slot < 3; slot++) {
            if (il != g->dspark_target_layer_ids[slot]) continue;
            if (!g->dspark_bulk_h[slot]) break;
            uint32_t cap_n = g->dspark_bulk_n;
            if (cap_n > n_tokens) cap_n = n_tokens;
            if (!ds4_gpu_dspark_hc_mean_reduce_batch(g->dspark_bulk_h[slot],
                                                     g->batch_cur_hc,
                                                     DS4_N_EMBD, DS4_N_HC, cap_n)) {
                ok = false;
            }
            break;
        }
    }
    return ok;
}



static bool gpu_graph_eval_token_raw_swa_streaming(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }

    const bool profile = getenv("DS4_CUDA_GRAPH_TOKEN_PROFILE") != NULL;
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos, 1);

    const bool static_decode_map = gpu_graph_stream_decode_static_map_enabled();
    const bool static_map_state_cache =
        static_decode_map && gpu_graph_stream_decode_static_map_state_cache_enabled();
    const bool batch_static_decode =
        static_decode_map && gpu_graph_stream_decode_layer_batch_enabled(g);
    bool ok = true;
    if (static_decode_map) {
        if (!static_map_state_cache || !g->streaming_static_decode_map_current) {
            ok = gpu_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = static_map_state_cache;
        }
    } else {
        g->streaming_static_decode_map_current = false;
        ok = gpu_graph_stream_map_token(model, weights);
    }
    if (ok && !static_decode_map && DS4_N_LAYER > 0) {
        gpu_graph_stream_readahead_layer_decode(model, weights, 0);
    }
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                           model->map,
                                           model->size,
                                           weights->token_embd->abs_offset,
                                           (uint32_t)weights->token_embd->dim[1],
                                           (uint32_t)token,
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
    }
    if (batch_static_decode) {
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = gpu_graph_encode_decode_layer(g,
                                                 model,
                                                 &weights->layer[il],
                                                 il,
                                                 pos,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 token);
            if (ok) {
                ds4_gpu_tensor *tmp = g->cur_hc;
                g->cur_hc = g->after_ffn_hc;
                g->after_ffn_hc = tmp;
            }
        }
        if (ok && logits) {
            ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        }
        const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = (profile || throttle) ? now_sec() : 0.0;
        if (ok && logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        const double t_read = (profile || throttle) ? now_sec() : 0.0;
        if (profile) {
            fprintf(stderr,
                    "ds4: metal SSD streaming batched token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                    pos,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_done) * 1000.0,
                    (t_read - t0) * 1000.0,
                    logits != NULL);
        }
        if (ok && throttle) {
            graph_power_note_decode_token(g, t_read - t0);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after batched SSD streaming graph eval failure also failed\n");
            }
        }
        return ok;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;

    double encode_s = 0.0;
    double execute_s = 0.0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const double tl0 = profile ? now_sec() : 0.0;
        if (!static_decode_map && !gpu_graph_stream_map_layer_decode(model, weights, il)) {
            ok = false;
            break;
        }
        if (!static_decode_map && il + 1 < DS4_N_LAYER) {
            gpu_graph_stream_readahead_layer_decode(model, weights, il + 1);
        } else if (!static_decode_map && logits) {
            gpu_graph_stream_readahead_output(model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        bool encoded_layer = false;
        if (ok) {
            ok = gpu_graph_encode_decode_layer(g,
                                                 model,
                                                 &weights->layer[il],
                                                 il,
                                                 pos,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 token);
            encoded_layer = true;
        }
        if (encoded_layer) {
            ds4_gpu_tensor *tmp = g->cur_hc;
            g->cur_hc = g->after_ffn_hc;
            g->after_ffn_hc = tmp;
        }
        const double tl_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double tl_done = profile ? now_sec() : 0.0;
        if (profile) {
            encode_s += tl_encoded - tl0;
            execute_s += tl_done - tl_encoded;
        }
    }

    if (ok && logits && !static_decode_map) ok = gpu_graph_stream_map_output(model, weights);
    const double t_head0 = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_begin_commands() != 0;
    if (ok && logits) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;

    if (profile) {
        if (logits) {
            encode_s += t_head_encoded - t_head0;
            execute_s += t_done - t_head_encoded;
        }
        fprintf(stderr,
                "ds4: metal SSD streaming token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after SSD streaming graph eval failure also failed\n");
        }
    }
    return ok;
}



/* Execute one Metal decode token and read back logits. */
bool gpu_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    if (g && g->ssd_streaming) {
        return gpu_graph_eval_token_raw_swa_streaming(g, model, weights, token, pos, logits);
    }

    const bool profile = getenv("DS4_CUDA_GRAPH_TOKEN_PROFILE") != NULL;
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}



static bool gpu_graph_streaming_decode_prefill_wide_default(
        const ds4_weights *weights) {
    /* Only the (removed) Q4_K Flash layout defaulted to wide decode-prefill. */
    (void)weights;
    return false;
}



static uint32_t gpu_graph_streaming_decode_prefill_max_tokens(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights) {
    (void)g;
    if (getenv("DS4_CUDA_DISABLE_STREAMING_DECODE_PREFILL") != NULL) return 0;

    const char *env = getenv("DS4_CUDA_STREAMING_DECODE_PREFILL_MAX");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }

    if (DS4_MODEL_VARIANT != DS4_VARIANT_PRO &&
        DS4_MODEL_VARIANT != DS4_VARIANT_FLASH) {
        return 0u;
    }
    return gpu_graph_streaming_decode_prefill_wide_default(weights) ? 64u : 18u;
}



static bool gpu_graph_use_streaming_decode_prefill(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             n_tokens) {
    const uint32_t max_tokens =
        gpu_graph_streaming_decode_prefill_max_tokens(g, weights);
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           n_tokens != 0 &&
           max_tokens != 0 &&
           n_tokens <= max_tokens;
}



bool gpu_graph_use_streaming_decode_prefill_range(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             start,
        uint32_t             n_tokens) {
    /*
     * Short streamed prefill is latency-sensitive.  Use the decode-style path
     * by default for SSD streaming, while keeping a cold-only escape hatch for
     * strict-vector tests that need canonical layer-major prefill semantics.
     */
    if (start == 0) {
        if (getenv("DS4_CUDA_DISABLE_STREAMING_COLD_DECODE_PREFILL") != NULL)
            return false;
    }
    return gpu_graph_use_streaming_decode_prefill(g, weights, n_tokens);
}



bool gpu_graph_prefill_decode_streaming_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (!gpu_graph_use_streaming_decode_prefill(g, weights, n_tokens)) return false;
    if (!prompt || start > (uint32_t)prompt->len ||
        n_tokens > (uint32_t)prompt->len - start) return false;
    if (start == 0) {
        ds4_gpu_stream_expert_cache_reset_route_hotness();
    }

    const bool profile = getenv("DS4_CUDA_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    /*
     * `prefill_chunk` is not just UI progress: ds4_session_sync() wraps it to
     * advance the live checkpoint, and ds4-server may save that checkpoint.
     * Decode-style prefill only reads logits for the final token, so report one
     * cacheable chunk at the end. `prefill_display` remains per-token UI only.
     */
    if (progress) progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        const uint32_t pos = start + i;
        const bool last = i + 1u == n_tokens;
        float *token_logits = (last && logits) ? logits : NULL;
        if (!gpu_graph_eval_token_raw_swa(g,
                                            model,
                                            weights,
                                            prompt->v[pos],
                                            pos,
                                            token_logits)) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after decode-style streaming prefill failure also failed\n");
            }
            return false;
        }

        if (last && progress && logits) {
            progress(progress_ud, "prefill_chunk", (int)(pos + 1u), prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)(pos + 1u), prompt->len);
        }
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        if (show_progress) {
            fprintf(stderr, "ds4: gpu streaming prefill token %u/%u\r",
                    i + 1u,
                    n_tokens);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    if (profile) {
        const double t1 = now_sec();
        fprintf(stderr,
                "ds4: gpu decode-style streaming prefill start=%u tokens=%u total=%.3f ms\n",
                start,
                n_tokens,
                (t1 - t0) * 1000.0);
    }
    return true;
}



bool gpu_graph_capture_prefill_seed_router_selected(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_tokens) {
    uint32_t k = gpu_graph_streaming_prefill_cache_seed_k(g);
    if (k == 0) return true;
    if (k > n_tokens) k = n_tokens;
    g->prefill_seed_tokens = k;
    if (!g->prefill_seed_router_selected || !g->batch_router_selected ||
        il >= DS4_N_LAYER || n_tokens == 0 || sizeof(int) != sizeof(int32_t)) {
        return false;
    }

    const uint64_t bytes = (uint64_t)k * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t src_off = (uint64_t)(n_tokens - k) *
                             DS4_N_EXPERT_USED * sizeof(int);
    const uint64_t dst_off = (uint64_t)il *
                             DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                             DS4_N_EXPERT_USED * sizeof(int32_t);
    return ds4_gpu_tensor_copy(g->prefill_seed_router_selected,
                               dst_off,
                               g->batch_router_selected,
                               src_off,
                               bytes) != 0;
}



bool gpu_graph_seed_streaming_expert_cache_from_prefill(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    const uint32_t seed_tokens = g ? g->prefill_seed_tokens : 0;
    if (!gpu_graph_streaming_prefill_cache_seed_enabled(g)) return true;
    if (!model || !weights || !g->prefill_seed_router_selected || seed_tokens == 0) {
        return false;
    }

    int32_t selected[DS4_MAX_LAYER *
                     DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                     DS4_MAX_EXPERT_USED];
    const uint64_t bytes = (uint64_t)DS4_N_LAYER *
                           DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                           DS4_N_EXPERT_USED * sizeof(selected[0]);
    if (ds4_gpu_tensor_read(g->prefill_seed_router_selected,
                            0,
                            selected,
                            bytes) == 0) {
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_PREFILL_CACHE_SEED_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_rows = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!gpu_graph_decode_iq2_selected_slots_expected(g, layer)) continue;

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: Metal prefill expert-cache seed byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        for (uint32_t row = 0; row < seed_tokens; row++) {
            const size_t sel_off = ((size_t)il *
                                    DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS +
                                    row) * DS4_N_EXPERT_USED;
            if (ds4_gpu_stream_expert_cache_seed_selected(
                        &table,
                        selected + sel_off,
                        DS4_N_EXPERT_USED) == 0) {
                return false;
            }
            seeded_rows++;
        }
        seeded_layers++;
    }
    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming prefill expert-cache seed k=%u layers=%u rows=%u time=%.3f ms\n",
                seed_tokens,
                seeded_layers,
                seeded_rows,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}



bool gpu_graph_seed_streaming_expert_cache_from_hotlist(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!gpu_graph_streaming_expert_hotlist_enabled(g)) return true;
    if (!model || !weights) return false;

    uint32_t cache_budget = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!gpu_graph_decode_iq2_selected_slots_expected(g, layer)) continue;

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: Metal expert hotlist budget byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        cache_budget = ds4_gpu_stream_expert_cache_budget_for_expert_size(
                gate_expert_bytes,
                down_expert_bytes);
        break;
    }
    if (cache_budget == 0) return true;
    const uint32_t preload_count =
        gpu_graph_streaming_expert_preload_count(g, cache_budget);
    if (preload_count == 0) return true;
    const uint32_t current_count =
        ds4_gpu_stream_expert_cache_current_count();
    const bool profile =
        getenv("DS4_CUDA_STREAMING_EXPERT_HOTLIST_PROFILE") != NULL;
    if (current_count >= preload_count) {
        if (profile) {
            fprintf(stderr,
                    "ds4: Metal streaming expert hotlist seed skipped preload=%u current=%u\n",
                    preload_count,
                    current_count);
        }
        return true;
    }

    int32_t experts[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t counts[DS4_MAX_LAYER];
    bool seen[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    memset(experts, 0, sizeof(experts));
    memset(priorities, 0, sizeof(priorities));
    memset(counts, 0, sizeof(counts));
    memset(seen, 0, sizeof(seen));

    const char *path = getenv("DS4_CUDA_STREAMING_EXPERT_HOTLIST");
    uint32_t loaded = 0;
    const bool from_file = path && path[0];
    if (from_file) {
        if (!gpu_graph_streaming_expert_hotlist_load_file(path,
                                                           preload_count,
                                                           experts,
                                                           priorities,
                                                           counts,
                                                           seen,
                                                           &loaded)) {
            return false;
        }
    } else if (!gpu_graph_streaming_expert_hotlist_load_default(preload_count,
                                                                  experts,
                                                                  priorities,
                                                                  counts,
                                                                  seen,
                                                                  &loaded)) {
        return false;
    }
    if (loaded == 0) return true;

    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_experts = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t n = counts[il];
        if (n == 0) continue;
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!gpu_graph_decode_iq2_selected_slots_expected(g, layer)) continue;

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: Metal expert hotlist seed byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        if (ds4_gpu_stream_expert_cache_seed_experts(
                    &table,
                    experts[il],
                    priorities[il],
                    n) == 0) {
            return false;
        }
        seeded_layers++;
        seeded_experts += n;
    }
    if (profile) {
        const char *source_name = NULL;
        if (from_file) {
            source_name = path;
        } else if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {
            source_name = "built-in-flash";
        } else if (g_ds4_shape.variant == DS4_VARIANT_PRO) {
            source_name = "built-in-pro";
        } else {
            source_name = "built-in";
        }
        fprintf(stderr,
                "ds4: Metal streaming expert hotlist seed source=%s preload=%u loaded=%u layers=%u experts=%u time=%.3f ms\n",
                source_name,
                preload_count,
                loaded,
                seeded_layers,
                seeded_experts,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}



/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
bool gpu_graph_eval_token_raw_swa_top(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        int                   *top_id,
        float                 *logits) {
    if (!top_id) return false;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_token_raw_swa(g, model, weights,
                                                  token, pos, true, true);
    if (ok) {
        ok = ds4_gpu_argmax_tensor(g->comp_selected,
                                   g->logits,
                                   DS4_N_VOCAB) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top_id, sizeof(*top_id)) != 0;
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after top-only graph eval failure also failed\n");
        }
    }
    return ok;
}



bool gpu_graph_eval_mtp_draft_from_hc(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        ds4_gpu_tensor      *prev_hc,
        ds4_gpu_tensor      *out_hc,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    if (!mtp || !mtp->block.attn_q_a || !g->mtp_raw_cache || !prev_hc || !out_hc) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint32_t raw_row = pos % g->raw_cap;
    uint32_t n_raw = g->mtp_n_raw + 1u;
    if (n_raw > g->raw_window) n_raw = g->raw_window;
    if (n_raw > g->raw_cap) n_raw = g->raw_cap;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after = g->after_ffn_hc;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g->mtp_embed,
                                                  base_model->map,
                                                  base_model->size,
                                                  base_weights->token_embd->abs_offset,
                                                  (uint32_t)base_weights->token_embd->dim[1],
                                                  (uint32_t)token,
                                                  DS4_N_EMBD,
                                                  1) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->mtp_enorm,
                                                  g->mtp_embed,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->enorm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->mtp_eproj,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->e_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_enorm,
                                              1) != 0;
    if (ok) ok = ds4_gpu_repeat_hc_tensor(g->mtp_eproj_hc,
                                            g->mtp_eproj,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->mtp_hnorm_hc,
                                                       prev_hc,
                                                       mtp_model->map,
                                                       mtp_model->size,
                                                       mtp->hnorm->abs_offset,
                                                       DS4_N_EMBD,
                                                       DS4_N_HC,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->mtp_hproj_hc,
                                              mtp_model->map,
                                              mtp_model->size,
                                              mtp->h_proj->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD,
                                              g->mtp_hnorm_hc,
                                              DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_add_tensor(g->mtp_input_hc,
                                      g->mtp_eproj_hc,
                                      g->mtp_hproj_hc,
                                      (uint32_t)hc_dim) != 0;
    if (ok) {
        g->cur_hc = g->mtp_input_hc;
        g->after_ffn_hc = out_hc;
        ok = gpu_graph_encode_decode_layer(g,
                                             mtp_model,
                                             &mtp->block,
                                             1,
                                             pos,
                                             g->mtp_raw_cache,
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
    }
    if (ok) g->cur_hc = out_hc;
    if (ok) ok = gpu_graph_encode_output_head_mtp(g,
                                                    base_model,
                                                    base_weights,
                                                    mtp_model,
                                                    mtp,
                                                    base_weights->output->dim[1]);
    if (ok && top_id) {
        ok = ds4_gpu_argmax_tensor(g->comp_selected,
                                   g->logits,
                                   DS4_N_VOCAB) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && top_id) {
        ok = ds4_gpu_tensor_read(g->comp_selected, 0, top_id, sizeof(*top_id)) != 0;
    }
    if (ok && g->mtp_n_raw < g->raw_window) g->mtp_n_raw++;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        g->after_ffn_hc = saved_after;
    }
    return ok;
}



bool gpu_graph_eval_mtp_draft(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        int                    token,
        uint32_t               pos,
        float                 *logits,
        int                   *top_id) {
    return gpu_graph_eval_mtp_draft_from_hc(g,
                                              base_model,
                                              base_weights,
                                              mtp_model,
                                              mtp,
                                              g->cur_hc,
                                              g->mtp_state_hc,
                                              token,
                                              pos,
                                              logits,
                                              top_id);
}




/* Stage-B no-replay rollback for the fused spec loop: after restoring the
 * pre-batch frontier snapshot, roll ONLY the recurrent compressor/indexer pool
 * state forward through the committed batch positions using the projections
 * saved during the verify batch (spec_comp_*_save). Bit-identical to what a
 * transformer replay would produce: the same per-token update kernels run on
 * the same input rows in the same order -- minus the 43-layer forward. The
 * comp-cache rows and raw KV need no work (the batch already wrote the
 * committed positions' rows from identical state; rejected rows are position-
 * addressed and get overwritten). Counters are set by formula. The pooled-row
 * emit goes to a scratch sink (cache rows are already correct). */
bool gpu_graph_dspark_compressor_rollforward(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        const ds4_weights *weights,
        uint32_t          pos0,
        uint32_t          n_positions) {
    if (!g || !model || !weights) return false;
    if (n_positions == 0) return true;
    if (n_positions > 17u || !g->spec_comp_scratch_row) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const ds4_layer_weights *layer = &weights->layer[il];
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const uint32_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
        const float freq_base = layer_rope_freq_base(il);
        const float freq_scale = layer_rope_freq_scale(il);
        const float ext_factor = DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
        float attn_factor = 1.0f;
        if (ext_factor != 0.0f && freq_scale > 0.0f) {
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        if (!g->spec_comp_kv_save[il] || !g->spec_comp_sc_save[il]) return false;
        for (uint32_t t = 0; t < n_positions; t++) {
            const uint32_t pos = pos0 + t;
            ds4_gpu_tensor *kv_view = gpu_graph_tensor_row_view(g->spec_comp_kv_save[il], t, comp_width);
            ds4_gpu_tensor *sc_view = gpu_graph_tensor_row_view(g->spec_comp_sc_save[il], t, comp_width);
            bool ok = kv_view && sc_view &&
                ds4_gpu_compressor_update_tensor(kv_view, sc_view,
                        g->layer_attn_state_kv[il], g->layer_attn_state_score[il],
                        g->spec_comp_scratch_row,
                        model->map, model->size,
                        layer->attn_compressor_ape->abs_offset,
                        layer->attn_compressor_ape->type,
                        layer->attn_compressor_norm->abs_offset,
                        layer->attn_compressor_norm->type,
                        DS4_N_HEAD_DIM, ratio, pos, 0,
                        DS4_N_ROT, (uint32_t)DS4_ROPE_ORIG_CTX,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                        DS4_RMS_EPS) != 0;
            ds4_gpu_tensor_free(sc_view);
            ds4_gpu_tensor_free(kv_view);
            if (!ok) return false;
            if (ratio == 4 && g->spec_icomp_kv_save[il]) {
                ds4_gpu_tensor *ikv = gpu_graph_tensor_row_view(g->spec_icomp_kv_save[il], t, index_width);
                ds4_gpu_tensor *isc = gpu_graph_tensor_row_view(g->spec_icomp_sc_save[il], t, index_width);
                ok = ikv && isc &&
                    ds4_gpu_compressor_update_tensor(ikv, isc,
                            g->layer_index_state_kv[il], g->layer_index_state_score[il],
                            g->spec_comp_scratch_row,
                            model->map, model->size,
                            layer->indexer_compressor_ape->abs_offset,
                            layer->indexer_compressor_ape->type,
                            layer->indexer_compressor_norm->abs_offset,
                            layer->indexer_compressor_norm->type,
                            DS4_N_INDEXER_HEAD_DIM, ratio, pos, 0,
                            DS4_N_ROT, (uint32_t)DS4_ROPE_ORIG_CTX,
                            freq_base, freq_scale, ext_factor, attn_factor,
                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                ds4_gpu_tensor_free(isc);
                ds4_gpu_tensor_free(ikv);
                if (!ok) return false;
            }
        }
        g->layer_n_comp[il] = (pos0 + n_positions) / ratio;
        if (ratio == 4) g->layer_n_index_comp[il] = (pos0 + n_positions) / ratio;
    }
    return true;
}
