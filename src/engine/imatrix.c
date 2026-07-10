#include "ds4_engine_internal.h"



bool imatrix_collector_init(ds4_imatrix_collector *c, uint32_t cap_tokens, const char *dataset_path) {
    memset(c, 0, sizeof(*c));
    c->cap_tokens = cap_tokens ? cap_tokens : 1u;
    c->dataset_path = dataset_path;
    const size_t gate_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_EMBD;
    const size_t down_n = (size_t)DS4_N_LAYER * DS4_N_EXPERT * DS4_N_FF_EXP;
    c->gate_up_sum2 = xcalloc(gate_n, sizeof(c->gate_up_sum2[0]));
    c->down_sum2 = xcalloc(down_n, sizeof(c->down_sum2[0]));
    c->ffn_norm_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EMBD * sizeof(c->ffn_norm_buf[0]));
    c->routed_mid_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_buf[0]));
    c->routed_mid_f16_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(c->routed_mid_f16_buf[0]));
    c->selected_buf = xmalloc((size_t)c->cap_tokens * DS4_N_EXPERT_USED * sizeof(c->selected_buf[0]));
    c->sq_tmp = xmalloc((size_t)DS4_N_EMBD * sizeof(c->sq_tmp[0]));
    return c->gate_up_sum2 && c->down_sum2 && c->ffn_norm_buf &&
           c->routed_mid_buf && c->routed_mid_f16_buf && c->selected_buf && c->sq_tmp;
}



void imatrix_collector_free(ds4_imatrix_collector *c) {
    if (!c) return;
    free(c->gate_up_sum2);
    free(c->down_sum2);
    free(c->ffn_norm_buf);
    free(c->routed_mid_buf);
    free(c->routed_mid_f16_buf);
    free(c->selected_buf);
    free(c->sq_tmp);
    memset(c, 0, sizeof(*c));
}



static float *imatrix_gate_up_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->gate_up_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_EMBD;
}



static float *imatrix_down_ptr(ds4_imatrix_collector *c, uint32_t il, uint32_t expert) {
    return c->down_sum2 + ((size_t)il * DS4_N_EXPERT + expert) * DS4_N_FF_EXP;
}



static bool imatrix_collect_layer_batch(
        ds4_imatrix_collector *c,
        ds4_gpu_graph         *g,
        uint32_t               il,
        uint32_t               n_tokens) {
    if (!c || n_tokens == 0) return true;
    if (n_tokens > c->cap_tokens) return false;

    const uint64_t norm_bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    const uint64_t mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t mid_bytes = mid_elems * (g->batch_routed_mid_is_f16 ? sizeof(uint16_t) : sizeof(float));
    const uint64_t sel_bytes = (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int);
    void *mid_dst = g->batch_routed_mid_is_f16
        ? (void *)c->routed_mid_f16_buf
        : (void *)c->routed_mid_buf;
    if (ds4_gpu_tensor_read(g->batch_ffn_norm, 0, c->ffn_norm_buf, norm_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_routed_mid, 0, mid_dst, mid_bytes) == 0 ||
        ds4_gpu_tensor_read(g->batch_router_selected, 0, c->selected_buf, sel_bytes) == 0)
    {
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *x = c->ffn_norm_buf + (size_t)t * DS4_N_EMBD;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) c->sq_tmp[i] = x[i] * x[i];

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const int expert = c->selected_buf[(size_t)t * DS4_N_EXPERT_USED + slot];
            if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) continue;

            float *gate_up = imatrix_gate_up_ptr(c, il, (uint32_t)expert);
            for (uint32_t i = 0; i < DS4_N_EMBD; i++) gate_up[i] += c->sq_tmp[i];
            c->gate_up_count[il][expert]++;

            float *down = imatrix_down_ptr(c, il, (uint32_t)expert);
            const size_t mid_off = ((size_t)t * DS4_N_EXPERT_USED + slot) * DS4_N_FF_EXP;
            if (g->batch_routed_mid_is_f16) {
                const uint16_t *mid = c->routed_mid_f16_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) {
                    const float v = f16_to_f32(mid[i]);
                    down[i] += v * v;
                }
            } else {
                const float *mid = c->routed_mid_buf + mid_off;
                for (uint32_t i = 0; i < DS4_N_FF_EXP; i++) down[i] += mid[i] * mid[i];
            }
            c->down_count[il][expert]++;
            c->observed_routes++;
        }
    }
    c->observed_tokens += n_tokens;
    c->chunks++;
    return true;
}



static void imatrix_write_i32(FILE *fp, int32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) ds4_die("failed to write imatrix");
}



static void imatrix_write_entry(
        FILE       *fp,
        const char *name,
        const float *sum2,
        const uint32_t *counts,
        uint32_t n_expert,
        uint32_t n_col) {
    const int32_t len = (int32_t)strlen(name);
    const int32_t ncall = 1;
    const int32_t nval = (int32_t)((uint64_t)n_expert * n_col);
    imatrix_write_i32(fp, len);
    if (fwrite(name, 1, (size_t)len, fp) != (size_t)len) ds4_die("failed to write imatrix name");
    imatrix_write_i32(fp, ncall);
    imatrix_write_i32(fp, nval);

    float *tmp = xmalloc((size_t)n_col * sizeof(tmp[0]));
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        const float *src = sum2 + (size_t)e * n_col;
        if (count == 0) {
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = 1.0f;
        } else {
            const float inv = 1.0f / (float)count;
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = src[i] * inv;
        }
        if (fwrite(tmp, sizeof(tmp[0]), n_col, fp) != n_col) ds4_die("failed to write imatrix values");
    }
    free(tmp);
}



bool imatrix_collector_save(
        const ds4_imatrix_collector *c,
        const ds4_weights           *weights,
        const char                  *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }

    const int32_t entries = (int32_t)(DS4_N_LAYER * 3);
    imatrix_write_i32(fp, entries);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        char name[256];
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_gate_exps->name.len, layer->ffn_gate_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_up_exps->name.len, layer->ffn_up_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->gate_up_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_EMBD,
                            c->gate_up_count[il],
                            DS4_N_EXPERT,
                            DS4_N_EMBD);
        snprintf(name, sizeof(name), "%.*s", (int)layer->ffn_down_exps->name.len, layer->ffn_down_exps->name.ptr);
        imatrix_write_entry(fp, name,
                            c->down_sum2 + (size_t)il * DS4_N_EXPERT * DS4_N_FF_EXP,
                            c->down_count[il],
                            DS4_N_EXPERT,
                            DS4_N_FF_EXP);
    }

    const int32_t chunks = (int32_t)c->chunks;
    imatrix_write_i32(fp, chunks);
    const char *dataset = c->dataset_path ? c->dataset_path : "";
    const int32_t dataset_len = (int32_t)strlen(dataset);
    imatrix_write_i32(fp, dataset_len);
    if (dataset_len && fwrite(dataset, 1, (size_t)dataset_len, fp) != (size_t)dataset_len) {
        ds4_die("failed to write imatrix dataset name");
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix output %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}



bool gpu_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    g->mtp_n_raw = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!gpu_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!gpu_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}



/* Execute graph-backend prefill in layer-major order so intermediate
 * activations stay on the GPU and cache state is built exactly once. */
static void gpu_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                start,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        int                     total) {
    if (!display_progress) return;
    if (layer_done > (uint32_t)DS4_N_LAYER) layer_done = (uint32_t)DS4_N_LAYER;
    uint64_t done = (uint64_t)n_tokens * layer_done / (uint32_t)DS4_N_LAYER;
    if (layer_done == (uint32_t)DS4_N_LAYER) done = n_tokens;
    display_progress(display_progress_ud, "prefill_display",
                     (int)(start + (uint32_t)done), total);
}



/* Bulk anchor-hidden dump for drafter retraining: after a chunk syncs, append
 * one record to DS4_DSPARK_PREFILL_DUMP -- {u32 n, u32 start, i32 ids[n],
 * f32 h0[n*4096], f32 h1[...], f32 h2[...]} -- and clear the arm flag. Records
 * with start==0 mark request boundaries for the trainer. Failure paths clear
 * the arm without writing so the verify batch path never sees a stale arm. */
static void dspark_bulk_drain(ds4_gpu_graph *g, const token_vec *prompt,
                              uint32_t start, uint32_t n, bool ok) {
    if (!g->dspark_bulk_n) return;
    const uint32_t cap_n = g->dspark_bulk_n < n ? g->dspark_bulk_n : n;
    g->dspark_bulk_n = 0;
    if (!ok) return;
    /* Prompt-window capture: keep the last <=128 positions' anchor hiddens in
     * the position%128 ring so generation can seed the drafter's context
     * window (contiguous positions -> at most two segment copies per layer). */
    if (g->dspark_prompt_h[0] && cap_n > 0) {
        const uint32_t win = DS4_DSPARK_DRAFT_WINDOW;
        const uint32_t take = cap_n < win ? cap_n : win;
        const uint32_t p0 = start + cap_n - take;
        for (int s2 = 0; s2 < 3; s2++) {
            uint32_t done = 0;
            while (done < take) {
                const uint32_t pos = p0 + done;
                const uint32_t slot = pos % win;
                uint32_t run = win - slot;
                if (run > take - done) run = take - done;
                (void)ds4_gpu_tensor_copy(g->dspark_prompt_h[s2],
                                          (uint64_t)slot * DS4_N_EMBD * sizeof(float),
                                          g->dspark_bulk_h[s2],
                                          (uint64_t)(pos - start) * DS4_N_EMBD * sizeof(float),
                                          (uint64_t)run * DS4_N_EMBD * sizeof(float));
                done += run;
            }
        }
        /* Contiguity guard: after a rewind + partial re-prefill the ring may
         * hold rows from an older prompt below `start`; mark them invalid. */
        if (start == 0 || start != g->dspark_prompt_n) g->dspark_prompt_lo = start;
        g->dspark_prompt_n = start + cap_n;
        /* Any prefill replaces the generation context, so the drafter's
         * committed-row window is stale by definition: empty it and let the
         * next generation start reseed from the prompt ring. (The server's
         * between-request reset does not go through invalidate/rewind, so
         * this is the invariant that catches every new prompt.) */
        for (int i2 = 0; i2 < 3; i2++) g->dspark_n_raw[i2] = 0;
    }
    const char *path = getenv("DS4_DSPARK_PREFILL_DUMP");
    if (!path || !path[0] || !g->dspark_bulk_h[0]) return;
    static FILE *f = NULL;
    static float *host = NULL;
    if (!f) f = fopen(path, "wb");
    if (!f) return;
    if (!host) host = xmalloc((size_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    uint32_t hdr[2] = { cap_n, start };
    fwrite(hdr, sizeof(uint32_t), 2, f);
    fwrite(prompt->v + start, sizeof(int32_t), cap_n, f);
    for (int s = 0; s < 3; s++) {
        if (!ds4_gpu_tensor_read(g->dspark_bulk_h[s], 0, host,
                                 (uint64_t)cap_n * DS4_N_EMBD * sizeof(float)))
            return;
        fwrite(host, sizeof(float), (size_t)cap_n * DS4_N_EMBD, f);
    }
    fflush(f);
}

static bool gpu_graph_prefill_layer_major_inner(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud);

bool gpu_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    const bool ok = gpu_graph_prefill_layer_major_inner(g, model, weights, prompt,
                                                        start, n_tokens, logits,
                                                        show_progress, imatrix,
                                                        display_progress,
                                                        display_progress_ud);
    dspark_bulk_drain(g, prompt, start, n_tokens, ok);
    return ok;
}

static bool gpu_graph_prefill_layer_major_inner(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    if (display_progress)
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);

    bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (!ok) return false;


    if (!gpu_graph_warmup_prefill_kernels(g, model, weights, n_tokens)) return false;

    /* Bulk anchor-hidden capture (drafter retraining): armed per chunk, after
     * warmup so warmup encodes don't pollute the buffers; drained (and cleared)
     * by gpu_graph_prefill_chunked_range after the chunk syncs. */
    g->dspark_bulk_n = g->dspark_bulk_h[0] ? n_tokens : 0;

    const bool split_profile = getenv("DS4_CUDA_GRAPH_PREFILL_SPLIT_PROFILE") != NULL;
    /*
     * A full long-prompt prefill can keep the GPU busy for a long time. Split
     * non-tiny prefills when a frontend asked for display progress: completed
     * layer command buffers are real scheduling/keepalive points, while
     * callbacks emitted while encoding one huge command buffer would only be
     * cosmetic.
     */
    const bool throttle = graph_power_throttle_enabled(g);
    /* Do NOT split (per-layer cudaDeviceSynchronize) just to fire a display
     * progress callback: on this backend end_commands() is a full device sync,
     * so per-layer progress drains the async launch pipeline 43x/chunk. Chunk-
     * level progress (gpu_graph_prefill_chunked_range) still fires; intra-chunk
     * progress is dropped in favor of throughput. */
    const bool split_commands = g->ssd_streaming ||
                                split_profile || throttle ||
                                n_tokens > 2048 || imatrix != NULL;
    const bool profile = getenv("DS4_CUDA_GRAPH_PREFILL_PROFILE") != NULL || split_profile;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

    if (!split_commands) {
        ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     model,
                                                     weights,
                                                     prompt,
                                                     start,
                                                     n_tokens);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = gpu_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                start,
                                                n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu whole-prefill layer %u encode failed\n", il);
            }
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);
        if (display_progress)
            display_progress(display_progress_ud, "prefill_display",
                             (int)(start + n_tokens), prompt->len);

        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = getenv("DS4_CUDA_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
        ds4_gpu_tensor *saved_cur = g->cur_hc;
        ds4_gpu_tensor *last_hc = NULL;
        if (ok && logits) {
            last_hc = gpu_graph_tensor_row_view(g->batch_cur_hc, output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok && logits) {
            g->cur_hc = last_hc;
            ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc = saved_cur;
        }

        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "ds4: gpu graph prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    if (g->ssd_streaming) {
        g->streaming_static_decode_map_current = false;
        if (!gpu_graph_stream_map_token(model, weights)) return false;
    }
    gpu_graph_stream_prefill_selected_profile_reset(g);
    gpu_graph_stream_prepare_slot layer_prepare_slots[DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD];
    memset(layer_prepare_slots, 0, sizeof(layer_prepare_slots));
    const bool layer_pagein =
        gpu_graph_stream_prefill_layer_pagein_enabled(g);
    const bool layer_readahead =
        !layer_pagein &&
        gpu_graph_stream_prefill_layer_readahead_enabled(g);
    const bool layer_pread =
        !layer_pagein && !layer_readahead &&
        gpu_graph_stream_prefill_layer_pread_enabled(g);
    const bool layer_madvise =
        !layer_pagein && !layer_pread && !layer_readahead &&
        gpu_graph_stream_prefill_layer_madvise_enabled(g);
    const bool layer_prepare =
        layer_pagein || layer_pread || layer_readahead || layer_madvise;
    const bool layer_prepare_overlap =
        layer_prepare && gpu_graph_stream_prefill_layer_pagein_overlap_enabled();
    const uint32_t layer_prepare_ahead =
        layer_prepare && layer_prepare_overlap ?
        gpu_graph_stream_prefill_layer_prepare_ahead() : 1u;
    const bool batch_selected_addr =
        gpu_graph_stream_prefill_batch_selected_addr_enabled(g, weights, n_tokens) ||
        gpu_graph_cuda_stream_prefill_batch_selected_addr_enabled(g, weights, n_tokens);
    if (g->ssd_streaming && DS4_N_LAYER > 0) {
        if (layer_prepare) {
            if (!gpu_graph_stream_prepare_start_if_needed(g,
                                                            model,
                                                            weights,
                                                            0,
                                                            n_tokens,
                                                            layer_madvise,
                                                            layer_pread,
                                                            layer_readahead,
                                                            batch_selected_addr,
                                                            layer_prepare_slots,
                                                            layer_prepare_ahead)) {
                return false;
            }
        } else {
            if (batch_selected_addr) {
                gpu_graph_stream_readahead_layer_decode(model, weights, 0);
            } else {
                gpu_graph_stream_readahead_layer(model, weights, 0);
            }
        }
    }

    double t_layer0 = (profile || throttle) ? now_sec() : 0.0;
    ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                 g->prefill_tokens,
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    const double t_embed_encoded = (profile || throttle) ? now_sec() : 0.0;
    const double t_embed_done = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: metal layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
        if (layer_prepare) {
            (void)gpu_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        double layer_elapsed = 0.0;
        if (layer_prepare &&
            !gpu_graph_stream_prepare_join_layer(g,
                                                   model,
                                                   weights,
                                                   il,
                                                   n_tokens,
                                                   layer_madvise,
                                                   layer_pread,
                                                   layer_readahead,
                                                   batch_selected_addr,
                                                   layer_prepare_slots,
                                                   layer_prepare_ahead)) {
            ok = false;
            break;
        }
        if (g->ssd_streaming) {
            g->streaming_static_decode_map_current = false;
            bool decode_only_map = batch_selected_addr;
            const bool map_ok = decode_only_map ?
                gpu_graph_stream_map_layer_decode(model, weights, il) :
                gpu_graph_stream_map_layer(model, weights, il);
            if (!map_ok) {
                ok = false;
                break;
            }
        }
        if (g->ssd_streaming) {
            if (layer_prepare && layer_prepare_overlap) {
                bool started_future = false;
                for (uint32_t ahead = 1; ahead <= layer_prepare_ahead; ahead++) {
                    if (il + ahead >= DS4_N_LAYER) break;
                    started_future = true;
                    if (!gpu_graph_stream_prepare_start_if_needed(g,
                                                                    model,
                                                                    weights,
                                                                    il + ahead,
                                                                    n_tokens,
                                                                    layer_madvise,
                                                                    layer_pread,
                                                                    layer_readahead,
                                                                    batch_selected_addr,
                                                                    layer_prepare_slots,
                                                                    layer_prepare_ahead)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
                if (!started_future && logits) {
                    gpu_graph_stream_readahead_output(model, weights);
                }
            } else if (!layer_prepare && il + 1 < DS4_N_LAYER) {
                if (batch_selected_addr) {
                    gpu_graph_stream_readahead_layer_decode(model, weights, il + 1);
                } else {
                    gpu_graph_stream_readahead_layer(model, weights, il + 1);
                }
            } else if (logits) {
                gpu_graph_stream_readahead_output(model, weights);
            }
        }
        if (split_profile) {
            const double t_attn0 = now_sec();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u attention encode failed\n", il);
            }
            const double t_attn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            start,
                                                            n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u ffn encode failed\n", il);
            }
            if (ok) {
                ds4_gpu_tensor *tmp = g->batch_cur_hc;
                g->batch_cur_hc = g->batch_next_hc;
                g->batch_next_hc = tmp;
            }
            if (ok) ok = gpu_graph_capture_prefill_seed_router_selected(g,
                                                                          il,
                                                                          n_tokens);
            const double t_ffn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();
            if (ok) {
                ok = gpu_graph_stream_prefill_selected_profile_layer(
                        g,
                        &weights->layer[il],
                        il,
                        n_tokens);
            }
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = (t_attn_done - t_attn0) + (t_ffn_done - t_ffn0);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "ds4: metal layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = (profile || throttle) ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        start,
                                                        n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u encode failed\n", il);
            }
            if (ok) ok = gpu_graph_capture_prefill_seed_router_selected(g,
                                                                          il,
                                                                          n_tokens);
            const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = (profile || throttle) ? now_sec() : 0.0;
            if (ok) {
                ok = gpu_graph_stream_prefill_selected_profile_layer(
                        g,
                        &weights->layer[il],
                        il,
                        n_tokens);
            }
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = t_done - t_chunk0;
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
        if (ok &&
            g->ssd_streaming &&
            layer_prepare &&
            !layer_prepare_overlap) {
            if (il + 1 < DS4_N_LAYER) {
                if (!gpu_graph_stream_prepare_start_if_needed(g,
                                                                model,
                                                                weights,
                                                                il + 1,
                                                                n_tokens,
                                                                layer_madvise,
                                                                layer_pread,
                                                                layer_readahead,
                                                                batch_selected_addr,
                                                                layer_prepare_slots,
                                                                layer_prepare_ahead)) {
                    ok = false;
                }
            } else if (logits) {
                gpu_graph_stream_readahead_output(model, weights);
            }
        }
        if (!ok) {
            if (layer_prepare) {
                (void)gpu_graph_stream_prepare_join_all(layer_prepare_slots,
                                                          layer_prepare_ahead);
            }
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        graph_power_note_prefill_layer(g, il, layer_elapsed);
        gpu_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  start,
                                                  n_tokens,
                                                  il + 1,
                                                  prompt->len);
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (!ok) {
        if (layer_prepare) {
            (void)gpu_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
        }
        return false;
    }
    if (show_progress) fputc('\n', stderr);
    gpu_graph_stream_prefill_selected_profile_summary(g);
    if (!gpu_graph_seed_streaming_expert_cache_from_hotlist(g, model, weights)) {
        return false;
    }
    if (!gpu_graph_seed_streaming_expert_cache_from_prefill(g, model, weights)) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = getenv("DS4_CUDA_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *last_hc = NULL;

    const double t_head0 = profile ? now_sec() : 0.0;
    if (logits) {
        last_hc = gpu_graph_tensor_row_view(g->batch_cur_hc,
                                              output_row,
                                              hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits && g->ssd_streaming) {
        const bool static_decode_map =
            gpu_graph_stream_decode_static_map_enabled();
        const bool static_map_state_cache =
            static_decode_map &&
            gpu_graph_stream_decode_static_map_state_cache_enabled();
        g->streaming_static_decode_map_current = false;
        if (static_map_state_cache) {
            ok = gpu_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = true;
        } else {
            ok = gpu_graph_stream_map_output(model, weights);
        }
    }
    if (ok && logits) {
        g->cur_hc = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "ds4: gpu layer-major prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}



bool gpu_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    if (gpu_graph_use_streaming_decode_prefill_range(g, weights, 0,
                                                       (uint32_t)n_tokens)) {
        return gpu_graph_prefill_decode_streaming_range(g,
                                                          model,
                                                          weights,
                                                          prompt,
                                                          0,
                                                          (uint32_t)n_tokens,
                                                          logits,
                                                          show_progress,
                                                          NULL,
                                                          NULL,
                                                          display_progress,
                                                          display_progress_ud,
                                                          cancel,
                                                          cancel_ud,
                                                          cancelled);
    }
    /* The layer-major fallback below may submit the whole short prefill as one
     * Metal command buffer.  Once that command is in flight there is no useful
     * safe prefix to expose: by the time cancellation can be observed again,
     * the prompt has already been fully read and the KV is valid.  Let the
     * caller observe the pending interrupt at generation time instead. */
    (void)cancel;
    (void)cancel_ud;
    (void)cancelled;
    return gpu_graph_prefill_layer_major(g,
                                           model,
                                           weights,
                                           prompt,
                                           0,
                                           (uint32_t)n_tokens,
                                           logits,
                                           show_progress,
                                           NULL,
                                           display_progress,
                                           display_progress_ud);
}



/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
bool gpu_graph_prefill_chunked_range(
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
        ds4_imatrix_collector *imatrix,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;
    if (g->ssd_streaming && start == 0) {
        ds4_gpu_stream_expert_cache_reset_route_hotness();
    }
    if (!imatrix &&
        gpu_graph_use_streaming_decode_prefill_range(g, weights,
                                                       start, n_tokens)) {
        return gpu_graph_prefill_decode_streaming_range(g,
                                                          model,
                                                          weights,
                                                          prompt,
                                                          start,
                                                          n_tokens,
                                                          logits,
                                                          show_progress,
                                                          progress,
                                                          progress_ud,
                                                          display_progress,
                                                          display_progress_ud,
                                                          cancel,
                                                          cancel_ud,
                                                          cancelled);
    }

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    const bool profile = getenv("DS4_CUDA_GRAPH_PREFILL_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    const uint32_t end = start + n_tokens;

    if (progress) {
        progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    }
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    for (uint32_t pos0 = start; pos0 < end; ) {
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        const uint32_t chunk_end = pos0 + chunk;
        /* Only the final chunk's logits are consumed (the progress callback below
         * reports position only, never reads logits). Running the full output
         * head + vocab GEMM + readback on every non-final chunk is wasted work
         * whose result is immediately overwritten. */
        float *chunk_logits = (chunk_end == end) ? logits : NULL;
        bool ok = gpu_graph_prefill_layer_major(g,
                                                  model,
                                                  weights,
                                                  prompt,
                                                  pos0,
                                                  chunk,
                                                  chunk_logits,
                                                  show_progress,
                                                  imatrix,
                                                  display_progress,
                                                  display_progress_ud);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        if (progress) {
            progress(progress_ud, "prefill_chunk", (int)chunk_end, prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)chunk_end, prompt->len);
        }
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        pos0 = chunk_end;
    }
    if (show_progress) fputc('\n', stderr);
    if (profile) {
        const double t_read = now_sec();
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                (t_read - t0) * 1000.0);
    }
    return true;
}



/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
bool gpu_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0) return false;
    return gpu_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress,
                                             progress,
                                             progress_ud,
                                             display_progress,
                                             display_progress_ud,
                                             NULL,
                                             cancel,
                                             cancel_ud,
                                             cancelled);
}



/* Layer-major speculative target verifier for tiny MTP suffixes.
 *
 * This is the first production-shaped verifier attempt: unlike repeated decode
 * it runs the target model layer-by-layer for the whole speculative suffix, and
 * unlike the diagnostic path it does not read back full logits for every row.
 * The verifier returns the row top-1 ids needed for acceptance.  The caller
 * then reads exactly one logits row: the row that becomes the new continuation
 * state.  It still reuses the existing batch layer kernels, so it is not yet
 * the final hand-written N=2/N=4 decode microbatch, but it exercises the right
 * verifier contract and removes the obvious diagnostic overheads first. */
bool gpu_graph_verify_suffix_tops(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        int                   *row_tops,
        float                 *row_logits) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (start > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - start) return false;
    const uint32_t top_rows = n_tokens > 1 ? n_tokens - 1 : 0;
    if (top_rows && !row_tops) return false;

    const bool timing = getenv("DS4_MTP_VERIFY_TIMING") != NULL;
    const double t0 = timing ? now_sec() : 0.0;
    bool ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, prompt, start, n_tokens);
    if (ok) ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                         g->prefill_tokens,
                                                         model,
                                                         weights,
                                                         prompt,
                                                         start,
                                                         n_tokens);
    const double t_uploaded = timing ? now_sec() : 0.0;
    if (!ok) return false;

    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = capture_prefix1 && n_tokens == 2;

    const double t_layers_encode0 = timing ? now_sec() : 0.0;
    ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = gpu_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            start,
                                            n_tokens);
    }
    const double t_layers_encoded = timing ? now_sec() : 0.0;
    if (ok) {
        ok = ds4_gpu_end_commands() != 0;
    } else {
        (void)ds4_gpu_synchronize();
    }
    const double t_layers_done = timing ? now_sec() : 0.0;
    g->spec_capture_prefix1 = saved_capture;
    if (!ok) return false;

    const double t_head_encode0 = timing ? now_sec() : 0.0;
    ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_output_head_batch(g,
                                                      model,
                                                      weights,
                                                      n_tokens,
                                                      weights->output->dim[1]);
    if (ok) {
        if (top_rows == 1) {
            /* Common K=2 verify case: top_k=1 over n_vocab → use the dedicated
             * argmax kernel (single-block tree-reduce) instead of the legacy
             * indexer_topk_kernel's single-thread O(n_vocab * top_k) fall-through. */
            ok = ds4_gpu_argmax_tensor(g->comp_selected,
                                       g->spec_logits,
                                       DS4_N_VOCAB) != 0;
        } else if (top_rows) {
            /* top-1 of each of the top_rows rows. The legacy
             * indexer_topk_tensor fall-through for this shape launches ONE
             * THREAD per row scanning the whole vocab (~14ms at 4 rows);
             * per-row argmax launches (1024-thread tree-reduce each) do the
             * same job in <1ms and write the same [top_rows] i32 layout. */
            for (uint32_t r = 0; ok && r < top_rows; r++) {
                ds4_gpu_tensor *row = ds4_gpu_tensor_view(
                        g->spec_logits,
                        (uint64_t)r * DS4_N_VOCAB * sizeof(float),
                        (uint64_t)DS4_N_VOCAB * sizeof(float));
                ds4_gpu_tensor *dst = ds4_gpu_tensor_view(
                        g->comp_selected,
                        (uint64_t)r * sizeof(int32_t),
                        sizeof(int32_t));
                ok = row && dst &&
                     ds4_gpu_argmax_tensor(dst, row, DS4_N_VOCAB) != 0;
                ds4_gpu_tensor_free(row);
                ds4_gpu_tensor_free(dst);
            }
        }
    }
    const double t_head_encoded = timing ? now_sec() : 0.0;
    if (ok) {
        ok = ds4_gpu_end_commands() != 0;
    } else {
        (void)ds4_gpu_synchronize();
    }
    const double t_head_done = timing ? now_sec() : 0.0;
    const double t_read0 = timing ? now_sec() : 0.0;
    if (ok && top_rows) {
        ok = ds4_gpu_tensor_read(g->comp_selected,
                                   0,
                                   row_tops,
                                   (uint64_t)top_rows * sizeof(row_tops[0])) != 0;
    }
    if (ok && row_logits) {
        ok = ds4_gpu_tensor_read(g->spec_logits,
                                   0,
                                   row_logits,
                                   (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    if (timing) {
        const double t_done = now_sec();
        fprintf(stderr,
                "ds4: mtp verify suffix tokens=%u start=%u capture_prefix1=%d upload=%.3f ms layers_encode=%.3f ms layers_execute=%.3f ms head_encode=%.3f ms head_execute=%.3f ms read=%.3f ms total=%.3f ms ok=%d\n",
                n_tokens,
                start,
                capture_prefix1 ? 1 : 0,
                (t_uploaded - t0) * 1000.0,
                (t_layers_encoded - t_layers_encode0) * 1000.0,
                (t_layers_done - t_layers_encoded) * 1000.0,
                (t_head_encoded - t_head_encode0) * 1000.0,
                (t_head_done - t_head_encoded) * 1000.0,
                (t_done - t_read0) * 1000.0,
                (t_done - t0) * 1000.0,
                ok ? 1 : 0);
    }
    return ok;
}



bool gpu_graph_read_spec_logits_row(ds4_gpu_graph *g, uint32_t row, float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->prefill_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                                 (uint64_t)row * row_bytes,
                                 logits,
                                 row_bytes) != 0;
}



/* Exact N=2 target verifier for MTP.
 *
 * The generic batch prefill path is fast, but it is not a safe substitute for
 * autoregressive decode: small row-wise differences in HC/MoE/output kernels
 * are enough to flip future greedy tokens.  This verifier keeps the exact
 * decode kernels and cache update order, but encodes the two proposed tokens
 * layer-by-layer in one command stream.  It returns the exact target top after
 * token0, and exact logits after token1. */
bool gpu_graph_verify_decode2_exact(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token0,
        int                    token1,
        uint32_t               start,
        int                   *top0,
        float                 *logits0,
        float                 *logits1) {
    if (!g || !top0 || !logits1 || g->raw_cap == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *cur0 = gpu_graph_tensor_row_view(g->batch_cur_hc, 0, hc_dim);
    ds4_gpu_tensor *cur1 = gpu_graph_tensor_row_view(g->batch_cur_hc, 1, hc_dim);
    ds4_gpu_tensor *next0 = gpu_graph_tensor_row_view(g->batch_next_hc, 0, hc_dim);
    ds4_gpu_tensor *next1 = gpu_graph_tensor_row_view(g->batch_next_hc, 1, hc_dim);
    bool ok = cur0 && cur1 && next0 && next1;

    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur0,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token0,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur1,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  (uint32_t)weights->token_embd->dim[1],
                                                  (uint32_t)token1,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;

    ds4_gpu_tensor *saved_cur = g->cur_hc;
    ds4_gpu_tensor *saved_after = g->after_ffn_hc;
    const bool saved_capture = g->spec_capture_prefix1;
    g->spec_capture_prefix1 = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t pos0 = start;
        const uint32_t pos1 = start + 1u;

        g->cur_hc = cur0;
        g->after_ffn_hc = next0;
        ok = gpu_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos0,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos0 % g->raw_cap,
                                             gpu_graph_raw_span_for_batch(g, pos0, 1),
                                             token0);
        if (!ok) break;
        ok = gpu_graph_capture_prefix1_attn_state(g, il) &&
             gpu_graph_capture_prefix1_index_state(g, il);
        if (!ok) break;

        g->cur_hc = cur1;
        g->after_ffn_hc = next1;
        ok = gpu_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos1,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos1 % g->raw_cap,
                                             gpu_graph_raw_span_for_batch(g, pos1, 1),
                                             token1);
        if (!ok) break;

        ds4_gpu_tensor *tmp = cur0; cur0 = next0; next0 = tmp;
        tmp = cur1; cur1 = next1; next1 = tmp;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefix1 = saved_capture;
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;

    if (ok) {
        g->cur_hc = cur0;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_argmax_tensor(g->comp_selected,
                                           g->logits,
                                           DS4_N_VOCAB) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        if (ok) ok = ds4_gpu_tensor_read(g->comp_selected, 0, top0, sizeof(*top0)) != 0;
        if (ok && logits0) {
            ok = ds4_gpu_tensor_read(g->logits,
                                       0,
                                       logits0,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits0[0])) != 0;
        }
    }

    if (ok) {
        g->cur_hc = cur1;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = gpu_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        g->cur_hc = saved_cur;
        if (ok) {
            ok = ds4_gpu_tensor_read(g->logits,
                                       0,
                                       logits1,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits1[0])) != 0;
        }
    }
    g->cur_hc = saved_cur;
    g->after_ffn_hc = saved_after;
    g->spec_capture_prefix1 = saved_capture;

    ds4_gpu_tensor_free(next1);
    ds4_gpu_tensor_free(next0);
    ds4_gpu_tensor_free(cur1);
    ds4_gpu_tensor_free(cur0);
    return ok;
}



/* Pick a raw SWA cache size for Metal.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
uint32_t gpu_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    const char *env = getenv("DS4_CUDA_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }

    return raw_cap;
}



/* Choose the prefill ubatch size. Whole-batch is fastest for normal prompts.
 * Long Flash prompts default to 4096-token chunks; PRO defaults to 8192. */
uint32_t gpu_graph_prefill_cap_for_prompt(int prompt_len,
                                                   uint32_t prefill_chunk) {
    return ds4_prefill_cap_for_prompt(prompt_len, prefill_chunk);
}



/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  On an M3
 * Max, prefill is faster from 2-token suffixes upward; keep the default at 4
 * as a conservative crossover.  The env knob remains useful for retuning. */
uint32_t gpu_graph_resume_prefill_min_tokens(void) {
    const char *env = getenv("DS4_CUDA_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return 4u;
}



ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk) {
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    if (ds4_backend_uses_graph(backend)) {
        m.prefill_cap = gpu_graph_prefill_cap_for_prompt((int)ctx,
                                                           prefill_chunk);
        m.raw_cap = gpu_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

        uint32_t min_ratio = UINT32_MAX;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
        }
        if (min_ratio == UINT32_MAX) min_ratio = ctx;
        m.comp_cap = ctx / min_ratio + 2u;
        if (m.comp_cap < 2u) m.comp_cap = 2u;

        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t layer_comp_cap = ctx / ratio + 2u;
            m.compressed_bytes += (uint64_t)layer_comp_cap *
                                  DS4_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        uint64_t attn_stage_cap = (uint64_t)(m.prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        /* indexer_scores/comp_mask token rows shrink to the slice size under
         * DS4_PREFILL_SLICE (see gpu_graph_prefill_slice / gpu_diag alloc). */
        uint64_t score_rows = (uint64_t)m.prefill_cap;
        if (gpu_graph_prefill_slice() != 0u &&
            (uint64_t)gpu_graph_prefill_slice() < score_rows) {
            score_rows = (uint64_t)gpu_graph_prefill_slice();
        }
        m.scratch_bytes = 2ull *
                          m.comp_cap *
                          score_rows *
                          sizeof(float) +
                          attn_stage_cap * DS4_N_HEAD_DIM * sizeof(float);
    } else {
        m.raw_cap = ds4_default_raw_cap(ctx);
        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t comp_cap = ctx / ratio + 2u;
            if (ratio == 4) m.comp_cap = comp_cap;
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
        m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(bool));
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}



ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int         ctx_size) {
    return ds4_context_memory_estimate_with_prefill(backend, ctx_size, 0);
}



int gpu_graph_prompt_logits_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                ctx_size) {
    int n_test = prompt->len;
    const char *n_test_env = getenv("DS4_CUDA_GRAPH_PROMPT_TOKENS");
    if (n_test_env && n_test_env[0]) {
        char *endp = NULL;
        const long v = strtol(n_test_env, &endp, 10);
        if (endp != n_test_env && v > 0 && v <= prompt->len) n_test = (int)v;
    }

    if (n_test <= 0 || n_test > ctx_size) {
        fprintf(stderr, "ds4: Metal graph prompt test needs 1..%d prompt tokens\n", ctx_size);
        return 1;
    }

    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, (uint32_t)n_test);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, (uint32_t)n_test, false);
    if (!ok) {
        gpu_graph_free(&g);
        fprintf(stderr, "ds4: failed to initialize Metal graph prompt test runtime\n");
        return 1;
    }
    const bool memory_report = getenv("DS4_CUDA_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    ds4_kv_cache cpu_cache;
    kv_cache_init(&cpu_cache, (uint32_t)ctx_size, raw_cap);
    float *cpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *gpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *oracle_logits = NULL;

    const char *oracle_path = getenv("DS4_ORACLE_LOGITS");
    if (oracle_path && oracle_path[0]) {
        oracle_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!read_f32_binary_file(oracle_path, oracle_logits, DS4_N_VOCAB)) {
            free(oracle_logits);
            oracle_logits = NULL;
        }
    }

    for (int t = 0; t < n_test; t++) {
        const bool last = t == n_test - 1;
        forward_token_raw_swa_cpu(last ? cpu_logits : NULL,
                                  model,
                                  weights,
                                  &cpu_cache,
                                  prompt->v[t],
                                  (uint32_t)t);
    }
    ok = gpu_graph_prefill_raw_swa(&g, model, weights, prompt, n_test,
                                     gpu_logits, true, NULL, NULL,
                                     NULL, NULL, NULL);
    if (memory_report) ds4_gpu_print_memory_report("after prompt graph");

    if (ok) {
        const char *dump_gpu = getenv("DS4_CUDA_GRAPH_DUMP_LOGITS");
        if (dump_gpu && dump_gpu[0]) {
            if (write_f32_binary_file(dump_gpu, gpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote Metal graph logits to %s\n", dump_gpu);
            }
        }
        const char *dump_cpu = getenv("DS4_CPU_DUMP_LOGITS");
        if (dump_cpu && dump_cpu[0]) {
            if (write_f32_binary_file(dump_cpu, cpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote CPU logits to %s\n", dump_cpu);
            }
        }
        if (getenv("DS4_CUDA_GRAPH_TRACE_CACHE") != NULL ||
            getenv("DS4_CUDA_GRAPH_TRACE_COMP") != NULL) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                const uint32_t n_raw = cpu_cache.layer[il].n_raw;
                if (n_raw != 0) {
                    const uint64_t raw_phys_n = (uint64_t)raw_cap * DS4_N_HEAD_DIM;
                    const uint64_t raw_logical_n = (uint64_t)n_raw * DS4_N_HEAD_DIM;
                    const uint32_t raw_start = n_raw < raw_cap ? 0u : ((uint32_t)n_test % raw_cap);
                    float *gpu_raw_phys = xmalloc((size_t)raw_phys_n * sizeof(float));
                    float *gpu_raw_logical = xmalloc((size_t)raw_logical_n * sizeof(float));
                    int raw_read_ok;
                    if (gpu_graph_raw_f16_enabled()) {
                        uint16_t *raw_h = xmalloc((size_t)raw_phys_n * sizeof(uint16_t));
                        raw_read_ok = ds4_gpu_tensor_read(g.layer_raw_cache[il], 0, raw_h,
                                                          raw_phys_n * sizeof(uint16_t));
                        if (raw_read_ok != 0) {
                            for (uint64_t i2 = 0; i2 < raw_phys_n; i2++) {
                                gpu_raw_phys[i2] = f16_to_f32(raw_h[i2]);
                            }
                        }
                        free(raw_h);
                    } else {
                        raw_read_ok = ds4_gpu_tensor_read(g.layer_raw_cache[il], 0, gpu_raw_phys,
                                                          raw_phys_n * sizeof(float));
                    }
                    if (raw_read_ok != 0) {
                        for (uint32_t r = 0; r < n_raw; r++) {
                            const uint32_t phys = (raw_start + r) % raw_cap;
                            memcpy(gpu_raw_logical + (uint64_t)r * DS4_N_HEAD_DIM,
                                   gpu_raw_phys + (uint64_t)phys * DS4_N_HEAD_DIM,
                                   (size_t)DS4_N_HEAD_DIM * sizeof(float));
                        }
                        fprintf(stderr,
                                "ds4: cache trace layer %u raw_n=%u raw_start=%u raw_max=%g raw_rms=%g\n",
                                il, n_raw, raw_start,
                                max_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n),
                                rms_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n));
                    }
                    free(gpu_raw_logical);
                    free(gpu_raw_phys);
                }

                const uint32_t n_comp = cpu_cache.layer[il].n_comp;
                if (n_comp == 0) continue;
                const uint64_t n = (uint64_t)n_comp * DS4_N_HEAD_DIM;
                float *gpu_comp = xmalloc((size_t)n * sizeof(float));
                bool comp_read = false;
                if (gpu_graph_attn_pack_enabled()) {
                    /* Packed cache: read through the bit-exact f32 dequant
                     * shadow (mirrors the prefill readers). */
                    ds4_gpu_tensor *shadow = gpu_graph_attn_comp_read_cache(&g, il, n_comp);
                    comp_read = shadow &&
                                ds4_gpu_tensor_read(shadow, 0, gpu_comp, n * sizeof(float)) != 0;
                } else {
                    comp_read = ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0,
                                                    gpu_comp, n * sizeof(float)) != 0;
                }
                if (comp_read) {
                    fprintf(stderr,
                            "ds4: comp trace layer %u n=%u attn_max=%g attn_rms=%g\n",
                            il, n_comp,
                            max_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n),
                            rms_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n));
                }
                free(gpu_comp);

                const uint32_t n_index = cpu_cache.layer[il].n_index_comp;
                if (n_index != 0 && g.layer_index_comp_cache[il]) {
                    const uint64_t ni = (uint64_t)n_index * DS4_N_INDEXER_HEAD_DIM;
                    float *gpu_index = xmalloc((size_t)ni * sizeof(float));
                    ds4_gpu_tensor *index_src = g.layer_index_comp_cache[il];
                    if (gpu_graph_idx_fp4_enabled()) {
                        /* Packed cache: dequant into the f32 staging or skip the
                         * trace — reading packed bytes as f32 would print garbage. */
                        index_src = (g.idx_comp_stage &&
                                     ds4_gpu_mxkv_dequant_tensor(g.layer_index_comp_cache[il],
                                                                 g.idx_comp_stage,
                                                                 DS4_ENGINE_MXKV_FMT_FP4,
                                                                 n_index,
                                                                 DS4_N_INDEXER_HEAD_DIM) != 0)
                            ? g.idx_comp_stage
                            : NULL;
                    }
                    if (index_src &&
                        ds4_gpu_tensor_read(index_src, 0, gpu_index, ni * sizeof(float)) != 0) {
                        fprintf(stderr,
                                "ds4: comp trace layer %u n=%u index_max=%g index_rms=%g\n",
                                il, n_index,
                                max_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni),
                                rms_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni));
                    }
                    free(gpu_index);
                }
            }
        }
        const uint64_t cpu_top = argmax_f32(cpu_logits, DS4_N_VOCAB);
        const uint64_t gpu_top = argmax_f32(gpu_logits, DS4_N_VOCAB);
        fprintf(stderr,
                "ds4: Metal prompt graph logits: tokens=%d logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                n_test,
                max_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                rms_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
        if (oracle_logits) {
            const uint64_t oracle_top = argmax_f32(oracle_logits, DS4_N_VOCAB);
            fprintf(stderr,
                    "ds4: oracle logits: tokens=%d oracle_top=%llu oracle_top_logit=%g cpu_max=%g cpu_rms=%g gpu_max=%g gpu_rms=%g\n",
                    n_test,
                    (unsigned long long)oracle_top,
                    oracle_logits[oracle_top],
                    max_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    max_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB));
        }
    } else {
        fprintf(stderr, "ds4: Metal prompt graph logits test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after prompt graph failure also failed\n");
        }
    }

    free(gpu_logits);
    free(cpu_logits);
    free(oracle_logits);
    kv_cache_free(&cpu_cache);
    gpu_graph_free(&g);
    return ok ? 0 : 1;
}



void embed_prompt(
        const ds4_model   * model,
        const ds4_weights * weights,
        const token_vec   * tokens,
        uint32_t            n_embd,
        float             * out) {
    for (int i = 0; i < tokens->len; i++) {
        embed_token_f16(model, weights, tokens->v[i], out + (uint64_t)i * n_embd);
    }
}

