#include "ds4_engine_internal.h"



#include "ds4_streaming_hotlist.inc"



bool gpu_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_HC_FUSION", &cache);
}



static bool gpu_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_KV_FUSION", &cache);
}



bool gpu_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_QKV_NORM_FUSION", &cache);
}



static bool gpu_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}



static bool gpu_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_HC_NORM_FUSION", &cache);
}



bool gpu_graph_enable_batch_hc_norm_fusion(void) {
    static int cache = -1;
    return !gpu_graph_use_reference_hc_norm_decode() &&
           gpu_graph_env_flag("DS4_CUDA_ENABLE_BATCH_HC_NORM_FUSION", &cache);
}



static bool gpu_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}



static bool gpu_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

/* Evaluated every layer on the decode path; cache the flag reads (like the
 * fusion toggles above) instead of scanning environ per layer. */
static bool gpu_graph_disable_shared_gate_up_swiglu(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION", &cache);
}

static bool gpu_graph_moe_replay_selected_ids(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_MOE_REPLAY_SELECTED_IDS", &cache);
}



static bool gpu_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (gpu_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}



static bool gpu_graph_hc_norm_fusion_check_enabled(void) {
    static int cache = -1;
    return gpu_graph_env_flag("DS4_CUDA_HC_NORM_FUSION_CHECK", &cache);
}



static float gpu_graph_hc_norm_fusion_check_tolerance(void) {
    static int initialized;
    static float tolerance;
    if (initialized) return tolerance;
    tolerance = 2.0e-4f;
    const char *env = getenv("DS4_CUDA_HC_NORM_FUSION_CHECK_TOL");
    if (env && env[0]) {
        char *end = NULL;
        const float v = strtof(env, &end);
        if (end != env && isfinite(v) && v > 0.0f) tolerance = v;
    }
    initialized = 1;
    return tolerance;
}



static bool gpu_graph_check_hc_norm_fusion(
        const char            *label,
        ds4_gpu_tensor        *fused_out,
        ds4_gpu_tensor        *fused_norm,
        const ds4_gpu_tensor  *mix,
        const ds4_gpu_tensor  *residual_hc,
        const ds4_model       *model,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint64_t               norm_weight_offset,
        uint32_t               il,
        uint32_t               pos) {
    if (!gpu_graph_hc_norm_fusion_check_enabled()) return true;
    if (!fused_out || !fused_norm || !mix || !residual_hc || !model) return false;

    const uint64_t n_embd = DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    bool ok = ref_split && ref_out && ref_norm;

    if (ok) {
        ok = ds4_gpu_hc_split_sinkhorn_tensor(ref_split,
                                              mix,
                                              model->map,
                                              model->size,
                                              scale_offset,
                                              base_offset,
                                              DS4_N_HC,
                                              DS4_N_HC_SINKHORN_ITER,
                                              DS4_HC_EPS) != 0 &&
             ds4_gpu_hc_weighted_sum_tensor(ref_out,
                                            residual_hc,
                                            ref_split,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0 &&
             ds4_gpu_rms_norm_weight_tensor(ref_norm,
                                            ref_out,
                                            model->map,
                                            model->size,
                                            norm_weight_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }

    if (ok) ok = ds4_gpu_end_commands() != 0;

    float *fused_out_cpu = NULL;
    float *ref_out_cpu = NULL;
    float *fused_norm_cpu = NULL;
    float *ref_norm_cpu = NULL;
    if (ok) {
        fused_out_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ref_out_cpu = xmalloc((size_t)n_embd * sizeof(float));
        fused_norm_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ref_norm_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ok = ds4_gpu_tensor_read(fused_out, 0, fused_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_out, 0, ref_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(fused_norm, 0, fused_norm_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_norm, 0, ref_norm_cpu, n_embd * sizeof(float)) != 0;
    }

    if (ok) {
        const float out_max = max_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float out_rms = rms_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float norm_max = max_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float norm_rms = rms_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float tol = gpu_graph_hc_norm_fusion_check_tolerance();
        fprintf(stderr,
                "ds4: Metal HC norm fusion check %s layer=%u pos=%u "
                "out_max=%g out_rms=%g norm_max=%g norm_rms=%g tol=%g\n",
                label ? label : "hc",
                il,
                pos,
                out_max,
                out_rms,
                norm_max,
                norm_rms,
                tol);
        if (out_max > tol || norm_max > tol) {
            fprintf(stderr,
                    "ds4: Metal HC norm fusion check failed for %s layer=%u pos=%u\n",
                    label ? label : "hc",
                    il,
                    pos);
            ok = false;
        }
    }

    free(fused_out_cpu);
    free(ref_out_cpu);
    free(fused_norm_cpu);
    free(ref_norm_cpu);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_split);

    const bool restart_ok = ds4_gpu_begin_commands() != 0;
    return ok && restart_ok;
}



static bool gpu_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (gpu_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}



int gpu_graph_attn_mx_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("DS4_ATTN_MX") != NULL ? 1 : 0;
    return cached;
}

uint64_t gpu_graph_attn_comp_cache_row_bytes(void) {
    if (gpu_graph_attn_mx_enabled()) return DS4_ENGINE_MXKV_FP8_ROWBYTES;
    return (uint64_t)DS4_N_HEAD_DIM *
           (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
}

uint32_t gpu_graph_attn_comp_read_is_f16(void) {
    return gpu_graph_attn_mx_enabled() ? 0u : (DS4_GPU_ATTN_COMP_CACHE_F16 ? 1u : 0u);
}

uint32_t gpu_graph_attn_comp_read_is_fp8(void) {
    return gpu_graph_attn_mx_enabled() ? 0u : (DS4_GPU_ATTN_COMP_CACHE_FP8 ? 1u : 0u);
}

/* Comp cache to hand the f32/f16/fp8 prefill attention consumers. Normally the
 * persistent cache itself; under MX storage, dequantize the first n_rows MXFP8
 * rows into the f32 shadow and return that (consumers then see plain f32 with
 * gpu_graph_attn_comp_read_is_f16/_is_fp8() == 0). */
ds4_gpu_tensor *gpu_graph_attn_comp_read_cache(ds4_gpu_graph *g, uint32_t il, uint32_t n_rows) {
    if (!g || il >= DS4_N_LAYER) return NULL;
    if (!gpu_graph_attn_mx_enabled()) return g->layer_attn_comp_cache[il];
    if (!g->attn_comp_dequant) return NULL;
    if (n_rows == 0) return g->attn_comp_dequant;
    if (n_rows > g->layer_comp_cap[il]) return NULL;
    if (ds4_gpu_mxkv_dequant_tensor(g->layer_attn_comp_cache[il], g->attn_comp_dequant,
                                    DS4_ENGINE_MXKV_FMT_FP8, n_rows, DS4_N_HEAD_DIM) == 0) {
        return NULL;
    }
    return g->attn_comp_dequant;
}



uint32_t gpu_graph_attn_comp_cache_is_f16(void) {
    return DS4_GPU_ATTN_COMP_CACHE_F16 ? 1u : 0u;
}

uint32_t gpu_graph_attn_comp_cache_is_fp8(void) {
    return DS4_GPU_ATTN_COMP_CACHE_FP8 ? 1u : 0u;
}
static bool gpu_graph_weight_is_plain_or_mxfp8(const ds4_tensor *w) {
    return w->type == DS4_TENSOR_F16 || w->type == DS4_TENSOR_FP8_E4M3;
}




static bool gpu_graph_store_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!g || il >= DS4_N_LAYER) return false;
    if (rows == 0) return true;
    if (!g->layer_attn_comp_cache[il] || !g->attn_comp_stage) return false;
    if (rows > g->attn_comp_stage_cap || first_row > g->layer_comp_cap[il] ||
        rows > g->layer_comp_cap[il] - first_row) {
        return false;
    }

    const uint64_t count = (uint64_t)rows * DS4_N_HEAD_DIM;
    const uint64_t dst_offset = (uint64_t)first_row *
                                gpu_graph_attn_comp_cache_row_bytes();
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        return ds4_gpu_tensor_copy_f32_to_f16(g->layer_attn_comp_cache[il],
                                               dst_offset,
                                               g->attn_comp_stage,
                                               0,
                                               count) != 0;
    }

    return ds4_gpu_tensor_copy(g->layer_attn_comp_cache[il],
                               dst_offset,
                               g->attn_comp_stage,
                               0,
                               count * sizeof(float)) != 0;
}



ds4_gpu_tensor *gpu_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return (DS4_GPU_ATTN_COMP_CACHE_F16 || gpu_graph_attn_mx_enabled())
        ? g->attn_comp_stage
        : g->layer_attn_comp_cache[il];
}



uint32_t gpu_graph_attn_comp_update_row(uint32_t row) {
    return (DS4_GPU_ATTN_COMP_CACHE_F16 || gpu_graph_attn_mx_enabled()) ? 0u : row;
}



bool gpu_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (gpu_graph_attn_mx_enabled()) {
        /* Pack the `rows` f32 rows the compressor staged into attn_comp_stage
         * into the persistent MXFP8 comp cache at first_row. */
        if (rows == 0) return true;
        if (!g || il >= DS4_N_LAYER || !g->layer_attn_comp_cache[il] || !g->attn_comp_stage) {
            return false;
        }
        if (first_row > g->layer_comp_cap[il] || rows > g->layer_comp_cap[il] - first_row) {
            return false;
        }
        ds4_gpu_tensor *dst = ds4_gpu_tensor_view(
            g->layer_attn_comp_cache[il],
            (uint64_t)first_row * DS4_ENGINE_MXKV_FP8_ROWBYTES,
            (uint64_t)rows * DS4_ENGINE_MXKV_FP8_ROWBYTES);
        bool packed = dst && ds4_gpu_mxkv_pack_tensor(g->attn_comp_stage, dst,
                                                      DS4_ENGINE_MXKV_FMT_FP8,
                                                      rows, DS4_N_HEAD_DIM) != 0;
        if (dst) ds4_gpu_tensor_free(dst);
        return packed;
    }
    if (DS4_GPU_ATTN_COMP_CACHE_FP8) return true;
    if (!DS4_GPU_ATTN_COMP_CACHE_F16) return true;
    return gpu_graph_store_attn_comp_stage(g, il, first_row, rows);
}



ds4_gpu_tensor *gpu_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16 || DS4_GPU_ATTN_COMP_CACHE_FP8 ||
        gpu_graph_attn_mx_enabled()) {
        return ds4_gpu_tensor_view(g->attn_comp_stage,
                                   0,
                                   (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}



ds4_gpu_tensor *gpu_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16 || DS4_GPU_ATTN_COMP_CACHE_FP8 ||
        gpu_graph_attn_mx_enabled()) {
        return g->attn_comp_stage;
    }
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(float));
}



void gpu_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    /* Only the pure-f32 path returns a fresh view; every staged path (F16, FP8,
     * MX) returns the persistent attn_comp_stage, which must not be freed. */
    if (!DS4_GPU_ATTN_COMP_CACHE_F16 && !DS4_GPU_ATTN_COMP_CACHE_FP8 &&
        !gpu_graph_attn_mx_enabled()) {
        ds4_gpu_tensor_free(t);
    }
}



/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
bool gpu_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);


bool gpu_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);


bool gpu_graph_decode_stage_profile_enabled(uint32_t il);


bool gpu_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);



static bool gpu_graph_use_streaming_iq2_cpu_router(void) {
    return getenv("DS4_CUDA_ENABLE_STREAMING_IQ2_CPU_ROUTER") != NULL &&
           getenv("DS4_CUDA_DISABLE_STREAMING_IQ2_CPU_ROUTER") == NULL;
}



static bool gpu_graph_use_cuda_selected_shared_overlap(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_CUDA_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP") == NULL;
}



static bool gpu_graph_use_iq2_selected_shared_overlap(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_CUDA_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP") == NULL &&
           getenv("DS4_CUDA_DISABLE_IQ2_SELECTED_SHARED_OVERLAP") == NULL;
}



static bool gpu_graph_use_iq2_selected_async_load(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_CUDA_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") == NULL;
}



static bool gpu_graph_use_iq2_selected_async_early_commit(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_CUDA_DISABLE_STREAMING_SELECTED_ASYNC_EARLY_COMMIT") == NULL;
}



static bool gpu_graph_decode_cpu_router_applicable(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    const bool streaming_iq2 =
        g &&
        g->ssd_streaming &&
        !g->quality &&
        gpu_graph_use_streaming_iq2_cpu_router() &&
        layer->ffn_gate_tid2eid == NULL &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        DS4_N_EXPERT_USED == 6 &&
        DS4_N_EXPERT >= 128 &&
        getenv("DS4_CUDA_MOE_WRITE_CLAMPED_ACT") == NULL &&
        getenv("DS4_CUDA_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
        getenv("DS4_CUDA_DISABLE_IQ2_SELECTED_EXPERT_VIEWS") == NULL;
    return streaming_iq2;
}



bool gpu_graph_decode_iq2_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
           DS4_N_EXPERT_USED == 6 &&
           DS4_N_EXPERT >= 128 &&
           getenv("DS4_CUDA_MOE_WRITE_CLAMPED_ACT") == NULL &&
           getenv("DS4_CUDA_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
           getenv("DS4_CUDA_DISABLE_IQ2_SELECTED_EXPERT_VIEWS") == NULL;
}



static bool gpu_graph_decode_cuda_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !layer ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps ||
        DS4_N_EXPERT_USED != 6 ||
        DS4_N_EXPERT < 128 ||
        getenv("DS4_CUDA_MOE_WRITE_CLAMPED_ACT") != NULL ||
        getenv("DS4_CUDA_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") != NULL) {
        return false;
    }
    const bool iq2 =
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        getenv("DS4_CUDA_DISABLE_IQ2_SELECTED_EXPERT_VIEWS") == NULL;
    return iq2;
}



uint32_t gpu_graph_streaming_prefill_cache_seed_k(const ds4_gpu_graph *g) {
    if (!g ||
        !g->ssd_streaming ||
        getenv("DS4_CUDA_ENABLE_STREAMING_PREFILL_CACHE_SEED") == NULL) {
        return 0;
    }

    uint32_t k = 1;
    const char *env = getenv("DS4_CUDA_STREAMING_PREFILL_CACHE_SEED_K");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0') {
            if (v == 0) return 0;
            k = v > DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS ?
                DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS : (uint32_t)v;
        }
    }
    return k;
}



bool gpu_graph_streaming_prefill_cache_seed_enabled(const ds4_gpu_graph *g) {
    return gpu_graph_streaming_prefill_cache_seed_k(g) != 0;
}



bool gpu_graph_streaming_expert_hotlist_enabled(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !g->ssd_streaming_cold &&
           getenv("DS4_CUDA_DISABLE_STREAMING_EXPERT_HOTLIST") == NULL;
}



static bool gpu_graph_streaming_expert_hotlist_add(
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    priority,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded) {
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return true;
    if (layer >= DS4_MAX_LAYER || expert >= DS4_MAX_EXPERT) return true;
    if (seen[layer][expert]) return true;
    if (counts[layer] >= DS4_MAX_EXPERT) return false;
    seen[layer][expert] = true;
    if (priority == 0) priority = 1;
    priorities[layer][counts[layer]] = priority;
    experts[layer][counts[layer]++] = (int32_t)expert;
    (*loaded)++;
    return true;
}



bool gpu_graph_streaming_expert_hotlist_load_file(
        const char *path,
        uint32_t    max_entries,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded_out) {
    if (!path || !path[0] || max_entries == 0 ||
        !experts || !priorities || !counts || !seen || !loaded_out) {
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr,
                "ds4: failed to open streaming expert hotlist %s: %s\n",
                path,
                strerror(errno));
        return false;
    }

    char line[256];
    uint64_t lineno = 0;
    uint32_t loaded = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (loaded >= max_entries) break;
        lineno++;
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        errno = 0;
        char *end = NULL;
        unsigned long layer = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        unsigned long expert = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        unsigned long long hits = strtoull(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        if (hits == 0) continue;
        const uint32_t priority =
            hits > UINT32_MAX ? UINT32_MAX : (uint32_t)hits;
        if (!gpu_graph_streaming_expert_hotlist_add((uint32_t)layer,
                                                      (uint32_t)expert,
                                                      priority,
                                                      experts,
                                                      priorities,
                                                      counts,
                                                      seen,
                                                      &loaded)) {
            goto bad_line;
        }
        continue;

bad_line:
        fprintf(stderr,
                "ds4: invalid streaming expert hotlist line %" PRIu64 " in %s\n",
                lineno,
                path);
        fclose(fp);
        return false;
    }
    if (ferror(fp)) {
        fprintf(stderr,
                "ds4: failed to read streaming expert hotlist %s: %s\n",
                path,
                strerror(errno));
        fclose(fp);
        return false;
    }
    fclose(fp);

    if (loaded == 0) {
        fprintf(stderr, "ds4: streaming expert hotlist %s had no usable nonzero entries\n", path);
    }
    *loaded_out = loaded;
    return true;
}



bool gpu_graph_streaming_expert_hotlist_load_default(
        uint32_t    max_entries,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded_out) {
    if (max_entries == 0 || !experts || !priorities || !counts || !seen || !loaded_out) {
        return false;
    }
    const uint16_t (*hotlist)[2] = NULL;
    uint32_t hotlist_count = 0;
    if (g_ds4_shape.variant == DS4_VARIANT_PRO) {
        hotlist = ds4_default_streaming_hotlist_pro;
        hotlist_count = ds4_default_streaming_hotlist_pro_count;
    } else if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {
        hotlist = ds4_default_streaming_hotlist_flash;
        hotlist_count = ds4_default_streaming_hotlist_flash_count;
    } else {
        *loaded_out = 0;
        return true;
    }
    uint32_t loaded = 0;
    for (uint32_t i = 0;
         i < hotlist_count && loaded < max_entries;
         i++) {
        if (!gpu_graph_streaming_expert_hotlist_add(
                hotlist[i][0],
                hotlist[i][1],
                max_entries - loaded,
                experts,
                priorities,
                counts,
                seen,
                &loaded)) {
            return false;
        }
    }
    *loaded_out = loaded;
    return true;
}



uint32_t gpu_graph_streaming_expert_preload_count(
        const ds4_gpu_graph *g,
        uint32_t             cache_budget) {
    if (!g || cache_budget == 0) return 0;
    uint32_t preload = g->streaming_preload_experts;
    if (preload == 0) {
        preload = cache_budget;
        /* Auto mode is a hot seed, not a request to synchronously fill the
         * whole cache. Large Flash caches can otherwise spend startup doing
         * thousands of preads into shared Metal buffers and trip the system
         * watchdog before decode begins. Explicit CLI preload counts bypass
         * this cap. */
        const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_AUTO_PRELOAD_CAP");
        uint32_t cap = 4096;
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            if (end != env && *end == '\0') {
                cap = v > UINT32_MAX ? UINT32_MAX : (uint32_t)v;
            }
        }
        if (cap != 0 && preload > cap) preload = cap;
    }
    if (preload > cache_budget) preload = cache_budget;
    const uint64_t max_possible = (uint64_t)DS4_N_LAYER * DS4_N_EXPERT;
    if ((uint64_t)preload > max_possible) preload = (uint32_t)max_possible;
    return preload;
}



static bool gpu_graph_decode_set_hash_selected_override(
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 token,
        uint64_t                 gate_tensor_bytes,
        uint64_t                 down_tensor_bytes,
        const ds4_gpu_graph     *g) {
    if (!layer->ffn_gate_tid2eid) return true;

    (void)gate_tensor_bytes;
    (void)down_tensor_bytes;
    const bool iq2_selected =
        gpu_graph_decode_iq2_selected_slots_expected(g, layer);
    if (!iq2_selected) {
        return true;
    }

    int selected[DS4_MAX_EXPERT_USED];
    int32_t selected_i32[DS4_MAX_EXPERT_USED];
    layer_hash_selected_experts(selected, model, layer, (int)token);
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        selected_i32[i] = (int32_t)selected[i];
    }
    if (g && g->ssd_streaming) {
        if (DS4_N_EXPERT == 0 ||
            gate_tensor_bytes % DS4_N_EXPERT != 0 ||
            down_tensor_bytes % DS4_N_EXPERT != 0) {
            return false;
        }
        const uint64_t gate_expert_bytes = gate_tensor_bytes / DS4_N_EXPERT;
        const uint64_t down_expert_bytes = down_tensor_bytes / DS4_N_EXPERT;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        if (ds4_gpu_stream_expert_cache_begin_selected_load(
                    &table,
                    selected_i32,
                    DS4_N_EXPERT_USED) == 0) {
            return false;
        }
    }
    return ds4_gpu_routed_moe_set_selected_override(selected_i32, DS4_N_EXPERT_USED) != 0;
}



static bool gpu_graph_decode_cpu_router(
        ds4_gpu_graph          *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                token) {
    const bool profile =
        getenv("DS4_CUDA_STREAMING_IQ2_CPU_ROUTER_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;
    if (ds4_gpu_tensor_read(g->ffn_norm,
                            0,
                            g->cpu_router_norm,
                            (uint64_t)DS4_N_EMBD * sizeof(g->cpu_router_norm[0])) == 0) {
        return false;
    }
    const double t_read = profile ? now_sec() : 0.0;

    float logits[DS4_MAX_EXPERT];
    float probs[DS4_MAX_EXPERT];
    int selected[DS4_MAX_EXPERT_USED];
    int32_t selected_i32[DS4_MAX_EXPERT_USED];
    float weights[DS4_MAX_EXPERT_USED];

    matvec_any(logits, model, layer->ffn_gate_inp, g->cpu_router_norm);
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, (int)token);
        layer_hash_router_weights_from_probs(weights, probs, selected);
    } else {
        layer_topk_selected_experts_from_probs(selected, weights, model, layer, probs);
    }
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        selected_i32[i] = (int32_t)selected[i];
    }
    const double t_cpu = profile ? now_sec() : 0.0;

    if (ds4_gpu_tensor_write(g->router_logits,
                             0,
                             logits,
                             (uint64_t)DS4_N_EXPERT * sizeof(logits[0])) == 0 ||
        ds4_gpu_tensor_write(g->router_probs,
                             0,
                             probs,
                             (uint64_t)DS4_N_EXPERT * sizeof(probs[0])) == 0 ||
        ds4_gpu_tensor_write(g->router_selected,
                             0,
                             selected_i32,
                             (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_i32[0])) == 0 ||
        ds4_gpu_tensor_write(g->router_weights,
                             0,
                             weights,
                             (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) == 0) {
        return false;
    }
    const double t_write = profile ? now_sec() : 0.0;
    if (ds4_gpu_begin_commands() == 0) return false;
    if (ds4_gpu_routed_moe_set_selected_override(selected_i32, DS4_N_EXPERT_USED) == 0) return false;

    if (profile) {
        fprintf(stderr,
                "ds4: Metal CPU router layer=%u gate=%s down=%s sync=%.3f ms read=%.3f ms cpu=%.3f ms write=%.3f ms total=%.3f ms\n",
                il,
                tensor_type_name(layer->ffn_gate_exps->type),
                tensor_type_name(layer->ffn_down_exps->type),
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_cpu - t_read) * 1000.0,
                (t_write - t_cpu) * 1000.0,
                (t_write - t0) * 1000.0);
    }
    return true;
}



static bool gpu_graph_use_iq2_selected_readahead_shared_delay(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_CUDA_ENABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY") != NULL &&
           getenv("DS4_CUDA_DISABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY") == NULL;
}



static bool gpu_graph_decode_selected_readahead_override(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
    if (!g || !model || !layer || !g->router_selected ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_SELECTED_READAHEAD_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t selected_ids[DS4_MAX_EXPERT_USED] = {0};
    if (ds4_gpu_tensor_read(g->router_selected,
                            0,
                            selected_ids,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_ids[0])) == 0) {
        return false;
    }
    const double t_read = profile ? now_sec() : 0.0;

    bool seen[DS4_MAX_EXPERT] = {0};
    uint32_t unique = 0;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming selected readahead expert id %d is outside 0..%u at layer %u\n",
                    selected_ids[i],
                    DS4_N_EXPERT,
                    il);
            return false;
        }
        const uint32_t expert = (uint32_t)selected_ids[i];
        if (seen[expert]) continue;
        seen[expert] = true;
        unique++;

        const uint64_t expert_id = (uint64_t)expert;
        if (expert_id > UINT64_MAX / gate_expert_bytes ||
            expert_id > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: Metal streaming selected readahead offset overflow\n");
            return false;
        }
        const uint64_t gate_rel = expert_id * gate_expert_bytes;
        const uint64_t down_rel = expert_id * down_expert_bytes;
        if (gate_rel > UINT64_MAX - layer->ffn_gate_exps->abs_offset ||
            gate_rel > UINT64_MAX - layer->ffn_up_exps->abs_offset ||
            down_rel > UINT64_MAX - layer->ffn_down_exps->abs_offset) {
            fprintf(stderr, "ds4: Metal streaming selected readahead offset overflow\n");
            return false;
        }

        gpu_graph_stream_readahead_range_impl(model,
                                                layer->ffn_gate_exps->abs_offset + gate_rel,
                                                gate_expert_bytes,
                                                true);
        gpu_graph_stream_readahead_range_impl(model,
                                                layer->ffn_up_exps->abs_offset + gate_rel,
                                                gate_expert_bytes,
                                                true);
        gpu_graph_stream_readahead_range_impl(model,
                                                layer->ffn_down_exps->abs_offset + down_rel,
                                                down_expert_bytes,
                                                true);
    }
    const double t_hint = profile ? now_sec() : 0.0;

    if (ds4_gpu_routed_moe_set_selected_override(selected_ids,
                                                 DS4_N_EXPERT_USED) == 0) {
        return false;
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_begin_selected_load(
                &table,
                selected_ids,
                DS4_N_EXPERT_USED) == 0) {
        return false;
    }
    if (ds4_gpu_begin_commands() == 0) return false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming selected readahead layer=%u unique=%u sync=%.3f ms read=%.3f ms hint=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                unique,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_hint - t_read) * 1000.0,
                (t_done - t_hint) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return true;
}



static bool gpu_graph_decode_cuda_selected_load(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
    if (!gpu_graph_decode_cuda_selected_slots_expected(g, layer) ||
        !model ||
        !g->router_selected ||
        DS4_N_EXPERT == 0 ||
        DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t selected_ids[DS4_MAX_EXPERT_USED] = {0};
    bool ok = ds4_gpu_tensor_read(g->router_selected,
                                  0,
                                  selected_ids,
                                  (uint64_t)DS4_N_EXPERT_USED *
                                      sizeof(selected_ids[0])) != 0;
    const double t_read = profile ? now_sec() : 0.0;

    if (ok) {
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                    &table,
                    selected_ids,
                    DS4_N_EXPERT_USED) != 0;
    }
    const double t_load = profile ? now_sec() : 0.0;

    if (ds4_gpu_begin_commands() == 0) ok = false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: CUDA streaming selected load layer=%u sync=%.3f ms read=%.3f ms load=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_load - t_read) * 1000.0,
                (t_done - t_load) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return ok;
}



bool gpu_graph_cuda_stream_prefill_batch_selected_load(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint32_t                  n_tokens,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
    if (!gpu_graph_decode_cuda_selected_slots_expected(g, layer) ||
        !model ||
        !g->batch_router_selected ||
        n_tokens <= 1 ||
        DS4_N_EXPERT == 0 ||
        DS4_N_EXPERT_USED == 0 ||
        getenv("DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_LOAD") != NULL) {
        return true;
    }

    if ((uint64_t)n_tokens > UINT64_MAX / (uint64_t)DS4_N_EXPERT_USED) {
        fprintf(stderr, "ds4: CUDA streaming prefill selected-id count overflow at layer %u\n", il);
        return false;
    }
    const uint64_t n_ids64 = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids64 == 0 || n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "ds4: CUDA streaming prefill selected-id byte size overflow at layer %u\n", il);
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t *selected_ids = xmalloc((size_t)n_ids64 * sizeof(selected_ids[0]));
    bool ok = ds4_gpu_tensor_read(g->batch_router_selected,
                                  0,
                                  selected_ids,
                                  n_ids64 * sizeof(selected_ids[0])) != 0;
    const double t_read = profile ? now_sec() : 0.0;
    if (ok) {
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        ok = ds4_gpu_stream_expert_cache_prepare_selected_batch(
                    &table,
                    selected_ids,
                    n_tokens,
                    DS4_N_EXPERT_USED) != 0;
    }
    free(selected_ids);
    const double t_load = profile ? now_sec() : 0.0;

    if (ds4_gpu_begin_commands() == 0) ok = false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: CUDA streaming prefill batch selected load layer=%u tokens=%u sync=%.3f ms read=%.3f ms load=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                n_tokens,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_load - t_read) * 1000.0,
                (t_done - t_load) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return ok;
}



static pthread_mutex_t g_gpu_graph_selected_async_load_mutex =
    PTHREAD_MUTEX_INITIALIZER;


static pthread_cond_t g_gpu_graph_selected_async_load_cond =
    PTHREAD_COND_INITIALIZER;


static pthread_cond_t g_gpu_graph_selected_async_load_done_cond =
    PTHREAD_COND_INITIALIZER;


static pthread_t g_gpu_graph_selected_async_load_thread;


static bool g_gpu_graph_selected_async_load_thread_started = false;


static bool g_gpu_graph_selected_async_load_has_job = false;


static bool g_gpu_graph_selected_async_load_done = false;


static gpu_graph_selected_async_load g_gpu_graph_selected_async_load_job;



static void gpu_graph_selected_async_load_run(
        gpu_graph_selected_async_load *job) {
    job->ok = false;

    if (!job->g || !job->model || !job->layer || !job->g->router_selected ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return;
    }
    if (job->event_value != 0) {
        if (ds4_gpu_wait_selected_readback_ready(job->event_value,
                                                 "selected-id async expert load") == 0) {
            return;
        }
        if (ds4_gpu_tensor_read(job->g->router_selected,
                                0,
                                job->selected_ids,
                                (uint64_t)DS4_N_EXPERT_USED *
                                    sizeof(job->selected_ids[0])) == 0) {
            return;
        }
    }
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (job->selected_ids[i] < 0 ||
            (uint32_t)job->selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming async selected expert id %d is outside 0..%u at layer %u\n",
                    job->selected_ids[i],
                    DS4_N_EXPERT,
                    job->il);
            return;
        }
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(job->model,
                                       job->layer,
                                       job->il,
                                       job->gate_expert_bytes,
                                       job->down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_begin_selected_load(
                &table,
                job->selected_ids,
                DS4_N_EXPERT_USED) == 0) {
        return;
    }

    job->ok = true;
}



static void *gpu_graph_selected_async_load_worker_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_gpu_graph_selected_async_load_mutex);
        while (!g_gpu_graph_selected_async_load_has_job) {
            pthread_cond_wait(&g_gpu_graph_selected_async_load_cond,
                              &g_gpu_graph_selected_async_load_mutex);
        }
        gpu_graph_selected_async_load job =
            g_gpu_graph_selected_async_load_job;
        pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);

        gpu_graph_selected_async_load_run(&job);

        pthread_mutex_lock(&g_gpu_graph_selected_async_load_mutex);
        g_gpu_graph_selected_async_load_job = job;
        g_gpu_graph_selected_async_load_has_job = false;
        g_gpu_graph_selected_async_load_done = true;
        pthread_cond_signal(&g_gpu_graph_selected_async_load_done_cond);
        pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);
    }
    return NULL;
}



static bool gpu_graph_selected_async_load_ensure_worker(void) {
    pthread_mutex_lock(&g_gpu_graph_selected_async_load_mutex);
    if (g_gpu_graph_selected_async_load_thread_started) {
        pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);
        return true;
    }
    const int rc = pthread_create(&g_gpu_graph_selected_async_load_thread,
                                  NULL,
                                  gpu_graph_selected_async_load_worker_main,
                                  NULL);
    if (rc != 0) {
        pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);
        fprintf(stderr,
                "ds4: failed to start Metal streaming async selected load worker: %s\n",
                strerror(rc));
        return false;
    }
    g_gpu_graph_selected_async_load_thread_started = true;
    pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);
    return true;
}



static DS4_MAYBE_UNUSED bool gpu_graph_selected_async_load_start(
        gpu_graph_selected_async_load *job,
        ds4_gpu_graph                   *g,
        const ds4_model                 *model,
        const ds4_layer_weights         *layer,
        uint32_t                         il,
        uint64_t                         event_value,
        uint64_t                         gate_expert_bytes,
        uint64_t                         down_expert_bytes) {
    if (!job || event_value == 0) return false;
    if (!gpu_graph_selected_async_load_ensure_worker()) return false;
    memset(job, 0, sizeof(*job));
    job->g = g;
    job->model = model;
    job->layer = layer;
    job->il = il;
    job->event_value = event_value;
    job->gate_expert_bytes = gate_expert_bytes;
    job->down_expert_bytes = down_expert_bytes;

    pthread_mutex_lock(&g_gpu_graph_selected_async_load_mutex);
    if (g_gpu_graph_selected_async_load_has_job ||
        g_gpu_graph_selected_async_load_done) {
        pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);
        return false;
    }
    g_gpu_graph_selected_async_load_job = *job;
    g_gpu_graph_selected_async_load_job.ok = false;
    g_gpu_graph_selected_async_load_has_job = true;
    pthread_cond_signal(&g_gpu_graph_selected_async_load_cond);
    pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);
    job->active = true;
    return true;
}



static bool gpu_graph_selected_async_load_finish(
        gpu_graph_selected_async_load *job) {
    if (!job || !job->active) return false;
    pthread_mutex_lock(&g_gpu_graph_selected_async_load_mutex);
    while (!g_gpu_graph_selected_async_load_done) {
        pthread_cond_wait(&g_gpu_graph_selected_async_load_done_cond,
                          &g_gpu_graph_selected_async_load_mutex);
    }
    *job = g_gpu_graph_selected_async_load_job;
    g_gpu_graph_selected_async_load_done = false;
    pthread_mutex_unlock(&g_gpu_graph_selected_async_load_mutex);
    job->active = false;
    if (!job->ok) return false;
    return ds4_gpu_routed_moe_set_selected_override(job->selected_ids,
                                                   DS4_N_EXPERT_USED) != 0;
}



bool gpu_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const bool compressed = g->comp_ratio_override >= 0
        ? g->comp_ratio_override != 0
        : ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !gpu_graph_use_reference_qkv_norm();

    bool ok = true;
    const bool decode_stage_profile = gpu_graph_decode_stage_profile_enabled(il);
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
#define DS4_CUDA_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = gpu_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_attn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    const bool fuse_hc_norm =
        DS4_N_HC == 4 &&
        !gpu_graph_use_reference_hc_decode() &&
        !gpu_graph_use_reference_hc_norm_decode();
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->attn_cur,
                                                         g->attn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->cur_hc,
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
        if (ok) {
            ok = gpu_graph_check_hc_norm_fusion("attn",
                                                  g->attn_cur,
                                                  g->attn_norm,
                                                  g->hc_mix,
                                                  g->cur_hc,
                                                  model,
                                                  layer->hc_attn_scale->abs_offset,
                                                  layer->hc_attn_base->abs_offset,
                                                  layer->attn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = gpu_graph_decode_hc_pre(g->attn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->cur_hc,
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_attn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_pre", g->attn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm, g->attn_cur,
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_norm", g->attn_norm, DS4_N_EMBD, il, pos);
    }
    bool qkv_pair_projected = false;
    if (ok && qkv_rms_fused) {
        qkv_pair_projected = ds4_gpu_matmul_mxfp8_pair_tensor(g->qr,
                                                             g->kv_raw,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a->abs_offset,
                                                             layer->attn_kv->abs_offset,
                                                             DS4_N_EMBD,
                                                             q_rank,
                                                             DS4_N_HEAD_DIM,
                                                             g->attn_norm,
                                                             1) != 0;
    }
    if (ok && !qkv_pair_projected) ok = ds4_gpu_matmul_mxfp8_tensor(g->qr,
                                                                    model->map,
                                                                    model->size,
                                                                    layer->attn_q_a->abs_offset,
                                                                    DS4_N_EMBD,
                                                                    q_rank,
                                                                    g->attn_norm,
                                                                    1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora", g->qr, q_rank, il, pos);
    }
    if (qkv_rms_fused) {
        if (ok && !qkv_pair_projected) ok = ds4_gpu_matmul_mxfp8_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->qr_norm,
                                                             g->qr,
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             g->kv,
                                                             g->kv_raw,
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             1,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->qr_norm, g->qr,
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("q_lora_norm", g->qr_norm, q_rank, il, pos);
    }
    if (qkv_rms_fused && ok) {
        gpu_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->q, model->map, model->size,
                                              layer->attn_q_b->abs_offset,
                                              q_rank, q_dim,
                                              g->qr_norm, 1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("Qraw", g->q, q_dim, il, pos);
    }
    const bool decode_q_norm_debug = gpu_graph_debug_wants("Qnorm", il, pos);
    bool decode_q_norm_rope_fused = false;
    if (ok && !decode_q_norm_debug) {
        decode_q_norm_rope_fused =
            ds4_gpu_head_rms_norm_rope_tail_tensor(g->q,
                                                   1,
                                                   DS4_N_HEAD,
                                                   DS4_N_HEAD_DIM,
                                                   DS4_N_ROT,
                                                   pos,
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
    if (!decode_q_norm_rope_fused) {
        if (ok) ok = ds4_gpu_head_rms_norm_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("Qnorm", g->q, q_dim, il, pos);
        }
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->q, 1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                                DS4_N_ROT, pos,
                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                false, freq_base, freq_scale, ext_factor, attn_factor,
                                                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("Qcur", g->q, q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->kv_raw, model->map, model->size,
                                                  layer->attn_kv->abs_offset,
                                                  DS4_N_EMBD, DS4_N_HEAD_DIM,
                                                  g->attn_norm, 1) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVraw", g->kv_raw, DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->kv, g->kv_raw,
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            gpu_graph_debug_dump_tensor("KVnorm", g->kv, DS4_N_HEAD_DIM, il, pos);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->kv, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false, freq_base, freq_scale, ext_factor, attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("KVrope", g->kv, DS4_N_HEAD_DIM, il, pos);
    }
    /* RoPE stays as the exact standalone kernel above.  The decode fusion
     * starts after that, where FP8 KV quantization and raw-cache storage can
     * share one pass without changing the trigonometric path. */
    if (ok) ok = gpu_graph_decode_kv_store(g->kv, raw_cache, raw_cap, raw_row);
    DS4_CUDA_PROFILE_DECODE_STAGE("kv_path");
    if (ok) {
        gpu_graph_debug_dump_tensor("KVcur", g->kv, DS4_N_HEAD_DIM, il, pos);
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = getenv("DS4_CUDA_INDEXER_STAGE_PROFILE") != NULL;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            !gpu_graph_weight_is_plain_or_mxfp8(layer->attn_compressor_kv) ||
            !gpu_graph_weight_is_plain_or_mxfp8(layer->attn_compressor_gate) ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: Metal graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        if (ok && !gpu_graph_use_reference_compressor_pair_proj()) {
            if (layer->attn_compressor_kv->type == DS4_TENSOR_F16) {
                ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                      g->comp_sc_cur,
                                                      model->map,
                                                      model->size,
                                                      layer->attn_compressor_kv->abs_offset,
                                                      layer->attn_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      comp_width,
                                                      g->attn_norm,
                                                      1) != 0;
            } else {
                ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                    layer->attn_compressor_kv,
                                                    DS4_N_EMBD, comp_width,
                                                    g->attn_norm, 1) &&
                     gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                    layer->attn_compressor_gate,
                                                    DS4_N_EMBD, comp_width,
                                                    g->attn_norm, 1);
            }
        } else {
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                         layer->attn_compressor_kv,
                                                         DS4_N_EMBD, comp_width,
                                                         g->attn_norm, 1);
            if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                         layer->attn_compressor_gate,
                                                         DS4_N_EMBD, comp_width,
                                                         g->attn_norm, 1);
        }
        const uint32_t comp_row = g->layer_n_comp[il];
        if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                        g->comp_sc_cur,
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
            if (!comp_row_view) {
                ok = false;
            } else if (gpu_graph_attn_mx_enabled()) {
                /* comp_row_view aliases the f32 stage; commit packs it to MXFP8. */
                gpu_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
            } else if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
                ds4_gpu_tensor *packed_dst = ds4_gpu_tensor_view(
                    g->layer_attn_comp_cache[il],
                    (uint64_t)comp_row * DS4_FP8_KV_ROWBYTES(DS4_N_HEAD_DIM),
                    (uint64_t)DS4_N_HEAD_DIM);
                ds4_gpu_tensor *scales_dst = ds4_gpu_tensor_view(
                    g->layer_attn_comp_cache[il],
                    (uint64_t)comp_row * DS4_FP8_KV_ROWBYTES(DS4_N_HEAD_DIM) + DS4_N_HEAD_DIM,
                    (uint64_t)DS4_FP8_KV_NBLK(DS4_N_HEAD_DIM) * sizeof(float));
                ok = packed_dst && scales_dst &&
                     ds4_gpu_dsv4_fp8_kv_pack_tensor(comp_row_view, packed_dst, scales_dst,
                                                     1, DS4_N_HEAD_DIM) != 0;
                if (packed_dst) ds4_gpu_tensor_free(packed_dst);
                if (scales_dst) ds4_gpu_tensor_free(scales_dst);
            } else {
                ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                if (ok) {
                    gpu_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
            }
            ds4_gpu_tensor_free(comp_row_view);
            if (ok) ok = gpu_graph_commit_attn_comp_stage(g, il, comp_row, 1);
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_compressor_kv) ||
                !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_compressor_gate) ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: Metal graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && !gpu_graph_use_reference_compressor_pair_proj()) {
                if (layer->indexer_compressor_kv->type == DS4_TENSOR_F16) {
                    ok = ds4_gpu_matmul_f16_pair_tensor(g->comp_kv_cur,
                                                          g->comp_sc_cur,
                                                          model->map,
                                                          model->size,
                                                          layer->indexer_compressor_kv->abs_offset,
                                                          layer->indexer_compressor_gate->abs_offset,
                                                          DS4_N_EMBD,
                                                          index_width,
                                                          g->attn_norm,
                                                          1) != 0;
                } else {
                    ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                        layer->indexer_compressor_kv,
                                                        DS4_N_EMBD, index_width,
                                                        g->attn_norm, 1) &&
                         gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                        layer->indexer_compressor_gate,
                                                        DS4_N_EMBD, index_width,
                                                        g->attn_norm, 1);
                }
            } else {
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_kv_cur, model,
                                                             layer->indexer_compressor_kv,
                                                             DS4_N_EMBD, index_width,
                                                             g->attn_norm, 1);
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->comp_sc_cur, model,
                                                             layer->indexer_compressor_gate,
                                                             DS4_N_EMBD, index_width,
                                                              g->attn_norm, 1);
            }
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = ds4_gpu_compressor_update_tensor(g->comp_kv_cur,
                                                            g->comp_sc_cur,
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            g->layer_index_comp_cache[il],
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
                        g->layer_index_comp_cache[il],
                        (uint64_t)index_row * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                        (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                          1,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                    ds4_gpu_tensor_free(index_row_view);
                }
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_sparse_threshold =
                gpu_graph_decode_indexer_sparse_threshold(g);
            if (ok &&
                g->layer_n_comp[il] > decode_sparse_threshold &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_attn_q_b) ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: Metal graph indexer q projection expects F16 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           !gpu_graph_weight_is_plain_or_mxfp8(layer->indexer_proj) ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: Metal graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->indexer_q,
                                                              model,
                                                              layer->indexer_attn_q_b,
                                                              q_rank,
                                                              indexer_q_dim,
                                                              g->qr_norm,
                                                              1);
                if (ok) ok = ds4_gpu_rope_tail_tensor(g->indexer_q, 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(g->indexer_q,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                if (ok) ok = gpu_graph_matmul_plain_tensor(g->indexer_weights, model,
                                                             layer->indexer_proj,
                                                             DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                             g->attn_norm, 1);
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_score_one_tensor(g->indexer_scores,
                                                                g->indexer_q,
                                                                g->indexer_weights,
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(g->comp_selected,
                                                           g->indexer_scores,
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           DS4_N_INDEXER_TOP_K) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = gpu_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = g->comp_selected;
                    /*
                     * Contract: the indexer top-k is fixed by the model config
                     * and must remain the full 512 rows.  Do not reduce this for
                     * throughput benchmarks.
                     *
                     * Why: the indexer is not just an implementation detail.  It
                     * decides which compressed memory rows are visible to the
                     * attention kernel.  If we keep only 128/256 rows, the later
                     * indexed-attention math may be perfectly computed, but it is
                     * computed over the wrong candidate set: rows ranked 257-512
                     * are removed before softmax/PV can use them.  Those rows may
                     * carry weak-but-necessary evidence for retrieval, name/number
                     * recall, or long-context disambiguation.  The error is
                     * therefore semantic/algorithmic, not the acceptable kind of
                     * local numerical drift caused by a different reduction order
                     * or Tensor/NAX precision.
                     *
                     * Short prompt tests, first-token agreement, or even a small
                     * official-vector set can miss this because many prompts do
                     * not need the tail of the 512 selected compressed rows.  The
                     * failure appears only when the model needs information that
                     * fell below the reduced cutoff.  Optimizations belong inside
                     * the score/top-k/attention implementation while preserving
                     * DS4_N_INDEXER_TOP_K.
                     */
                    n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                        ? DS4_N_INDEXER_TOP_K
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("compressor_indexer");

    if (ok) {
        const uint32_t raw_start = gpu_graph_raw_start_for_span(g, pos, n_raw);
        if (gpu_graph_attn_mx_enabled()) {
            /* MXFP8 decode attention (CUTLASS Sm120 MX GEMM). Compute-path swap
             * over the current f32 caches; requires a sm_120f build. */
            ok = ds4_gpu_attn_mx_decode(
                    g->heads, g->q,
                    raw_cache, raw_cap, raw_start, n_raw,
                    comp_cache,
                    (comp_selected != NULL && n_selected != 0) ? comp_selected : NULL,
                    n_comp,
                    (comp_selected != NULL) ? n_selected : 0,
                    model->map, model->size, layer->attn_sinks->abs_offset,
                    pos, g->raw_window, ds4_layer_compress_ratio(il),
                    DS4_N_HEAD, DS4_N_HEAD_DIM,
                    &g->attn_mx_scratch, &g->attn_mx_scratch_bytes) == 0;
        } else if (n_comp != 0 && comp_selected != NULL && n_selected != 0) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    g->heads,
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset,
                    g->q,
                    raw_cache,
                    g->layer_attn_comp_cache[il],
                    gpu_graph_attn_comp_cache_is_f16(), gpu_graph_attn_comp_cache_is_fp8(),
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = gpu_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            ok = ds4_gpu_attention_decode_heads_tensor(g->heads,
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset,
                                                         g->q, raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         gpu_graph_attn_comp_cache_is_f16(), gpu_graph_attn_comp_cache_is_fp8(),
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attention");
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_out", g->heads, q_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(g->heads,
                                            1, DS4_N_HEAD, DS4_N_HEAD_DIM,
                                            DS4_N_ROT, pos,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("kqv_back", g->heads, q_dim, il, pos);
    }
    const bool fuse_attn_out_hc =
        !gpu_graph_directional_steering_attn_enabled(g) &&
        !gpu_graph_use_reference_attn_out_hc();
    if (ok && fuse_attn_out_hc) {
        ok = ds4_gpu_attention_output_low_tensor(g->attn_low,
                                                   model->map,
                                                   model->size,
                                                   layer->attn_output_a->abs_offset,
                                                   group_dim,
                                                   rank,
                                                   n_groups,
                                                   g->heads) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_fp8_hc_expand_tensor(g->after_attn_hc,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        g->attn_low,
                                                        g->cur_hc,
                                                        g->hc_split,
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok) {
        ok = ds4_gpu_attention_output_batch_tensor(g->attn_out,
                                                     g->attn_low,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_output_a->abs_offset,
                                                     layer->attn_output_b->abs_offset,
                                                     group_dim, rank,
                                                     n_groups, DS4_N_EMBD,
                                                     g->heads, 1) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_low", g->attn_low, (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("attn_out", g->attn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_attn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_attn(g, g->attn_out, il, 1);
    }
    if (ok && !fuse_attn_out_hc) {
        ok = ds4_gpu_hc_expand_tensor(g->after_attn_hc, g->attn_out, g->cur_hc,
                                        g->hc_post, g->hc_comb, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_attn_post", g->after_attn_hc, hc_dim, il, pos);
    }
    if (ok) ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->after_attn_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->hc_mix, model, layer->hc_ffn_fn,
                                                 hc_dim, mix_hc, g->flat_hc, 1);
    if (ok && fuse_hc_norm) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(g->ffn_cur,
                                                         g->ffn_norm,
                                                         g->hc_split,
                                                         g->hc_mix,
                                                         g->after_attn_hc,
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
        if (ok) {
            ok = gpu_graph_check_hc_norm_fusion("ffn",
                                                  g->ffn_cur,
                                                  g->ffn_norm,
                                                  g->hc_mix,
                                                  g->after_attn_hc,
                                                  model,
                                                  layer->hc_ffn_scale->abs_offset,
                                                  layer->hc_ffn_base->abs_offset,
                                                  layer->ffn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = gpu_graph_decode_hc_pre(g->ffn_cur,
                                       g->hc_split,
                                       g->hc_mix,
                                       g->after_attn_hc,
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
        gpu_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_CUDA_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, DS4_N_EMBD, il, pos);
    }
    uint64_t gate_expert_bytes = 0, gate_row_bytes = 0;
    uint64_t down_expert_bytes = 0, down_row_bytes = 0;
    if (ok) {
        ok = routed_expert_gate_down_layout(layer->ffn_gate_exps, layer->ffn_down_exps,
                                            &gate_expert_bytes, &gate_row_bytes,
                                            &down_expert_bytes, &down_row_bytes);
    }
    if (ok && gpu_graph_decode_cpu_router_applicable(g, layer)) {
        ok = gpu_graph_decode_cpu_router(g, model, layer, il, (uint32_t)token);
    } else {
        if (ok) ok = gpu_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
                                                     DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
        if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
                                                    model->map, model->size,
                                                    layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                    layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                    layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                    (uint32_t)token,
                                                    DS4_N_EXPERT,
                                                    DS4_N_EXPERT_USED,
                                                    DS4_EXPERT_WEIGHT_SCALE,
                                                    0,
                                                    0,
                                                    layer->ffn_exp_probs_b != NULL,
                                                    layer->ffn_gate_tid2eid != NULL,
                                                    g->router_logits) != 0;
        if (ok) ok = gpu_graph_decode_set_hash_selected_override(model,
                                                                   layer,
                                                                   il,
                                                                   (uint32_t)token,
                                                                   layer->ffn_gate_exps->bytes,
                                                                   layer->ffn_down_exps->bytes,
                                                                   g);
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("router");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, DS4_N_EXPERT, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, DS4_N_EXPERT, il, pos);
        gpu_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, DS4_N_EXPERT_USED, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);
    }
    const bool keep_ffn_out = gpu_graph_needs_ffn_out(g, il, pos);
    const bool fuse_shared_gate_up =
        !g->quality &&
        !gpu_graph_disable_shared_gate_up_swiglu();
    const bool fuse_shared_down_hc =
        !keep_ffn_out && !gpu_graph_use_reference_shared_down_hc();
    const bool iq2_selected_shared_overlap =
        gpu_graph_use_iq2_selected_shared_overlap(g) &&
        gpu_graph_decode_iq2_selected_slots_expected(g, layer);
    const bool cuda_selected_shared_overlap =
        gpu_graph_use_cuda_selected_shared_overlap(g) &&
        gpu_graph_decode_cuda_selected_slots_expected(g, layer);
    const bool overlap_selected_shared =
        ok &&
        !decode_stage_profile &&
        !gpu_graph_decode_cpu_router_applicable(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        !gpu_graph_moe_replay_selected_ids() &&
        (iq2_selected_shared_overlap ||
         cuda_selected_shared_overlap);
    const bool async_selected_load =
        overlap_selected_shared &&
        ((iq2_selected_shared_overlap &&
          gpu_graph_use_iq2_selected_async_load(g)) ||
         cuda_selected_shared_overlap);
    const bool selected_readahead_shared_delay =
        ok &&
        !overlap_selected_shared &&
        !decode_stage_profile &&
        gpu_graph_use_iq2_selected_readahead_shared_delay(g) &&
        gpu_graph_decode_iq2_selected_slots_expected(g, layer) &&
        !gpu_graph_decode_cpu_router_applicable(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        !gpu_graph_moe_replay_selected_ids();
    const bool cuda_stream_selected_load =
        ok &&
        !overlap_selected_shared &&
        !selected_readahead_shared_delay &&
        g->ssd_streaming &&
        gpu_graph_decode_cuda_selected_slots_expected(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        !gpu_graph_moe_replay_selected_ids();
    if (cuda_stream_selected_load) {
        ok = gpu_graph_decode_cuda_selected_load(g,
                                                   model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
    }
    if (selected_readahead_shared_delay) {
        if (ok) {
            ok = gpu_graph_decode_selected_readahead_override(g,
                                                                model,
                                                                layer,
                                                                il,
                                                                gate_expert_bytes,
                                                                down_expert_bytes);
        }
        if (ok && fuse_shared_gate_up) {
            ok = ds4_gpu_shared_gate_up_swiglu_mxfp8_tensor(g->shared_gate,
                                                             g->shared_up,
                                                             g->shared_mid,
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_gate_shexp->abs_offset,
                                                             layer->ffn_up_shexp->abs_offset,
                                                             DS4_N_EMBD,
                                                             shared_dim,
                                                             g->ffn_norm,
                                                             DS4_SWIGLU_CLAMP_EXP) != 0;
        } else if (ok) {
            if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_gate, model->map, model->size,
                                                      layer->ffn_gate_shexp->abs_offset,
                                                      DS4_N_EMBD, shared_dim,
                                                      g->ffn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_up, model->map, model->size,
                                                      layer->ffn_up_shexp->abs_offset,
                                                      DS4_N_EMBD, shared_dim,
                                                      g->ffn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up,
                                               shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
        DS4_CUDA_PROFILE_DECODE_STAGE("shared_gate_up");
        if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                     g->routed_gate,
                                                     g->routed_up,
                                                     g->routed_mid,
                                                     g->routed_down,
                                                     tensor_map_base(model, layer->ffn_gate_exps),
                                                     tensor_map_size(model, layer->ffn_gate_exps),
                                                     layer->ffn_gate_exps->abs_offset,
                                                     layer->ffn_up_exps->abs_offset,
                                                     layer->ffn_down_exps->abs_offset,
                                                     layer->ffn_gate_exps->type,
                                                     layer->ffn_down_exps->type,
                                                     gate_expert_bytes, gate_row_bytes,
                                                     down_expert_bytes, down_row_bytes,
                                                     (uint32_t)expert_in_dim,
                                                     (uint32_t)down_in_dim,
                                                     (uint32_t)routed_out_dim,
                                                     g->router_selected, g->router_weights,
                                                     ds4_layer_n_expert(il),
                                                     DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm,
                                                     il) != 0;
        DS4_CUDA_PROFILE_DECODE_STAGE("routed_moe");
        if (ok) {
            gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                          (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
        }
        if (ok && fuse_shared_down_hc) {
            ok = ds4_gpu_shared_down_hc_expand_mxfp8_tensor(g->after_ffn_hc,
                                                             g->shared_out,
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_down_shexp->abs_offset,
                                                             shared_dim,
                                                             DS4_N_EMBD,
                                                             g->shared_mid,
                                                             g->routed_out,
                                                             g->after_attn_hc,
                                                             g->hc_split,
                                                             DS4_N_EMBD,
                                                             DS4_N_HC) != 0;
        } else if (ok) {
            ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_out, model->map, model->size,
                                              layer->ffn_down_shexp->abs_offset,
                                              shared_dim, DS4_N_EMBD,
                                              g->shared_mid, 1) != 0;
        }
        DS4_CUDA_PROFILE_DECODE_STAGE("shared_down");
        if (ok) {
            gpu_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
        }
        if (ok && keep_ffn_out) {
            ok = gpu_graph_ensure_ffn_out(g) &&
                 ds4_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
        }
        if (ok && keep_ffn_out) {
            gpu_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
        }
        if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
            ok = gpu_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
        }
        if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
            ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                            g->ffn_out,
                                            g->after_attn_hc,
                                            g->hc_post,
                                            g->hc_comb,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        } else if (ok && !fuse_shared_down_hc) {
            ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                      g->routed_out,
                                                      g->shared_out,
                                                      g->after_attn_hc,
                                                      g->hc_split,
                                                      DS4_N_EMBD,
                                                      DS4_N_HC) != 0;
        }
        DS4_CUDA_PROFILE_DECODE_STAGE("ffn_hc_post");
        if (ok) {
            gpu_graph_debug_dump_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
        }
        return ok;
    }
    if (overlap_selected_shared) {
        uint64_t selected_event = 0;
        if (ok) ok = ds4_gpu_signal_selected_readback_ready(&selected_event) != 0;
        gpu_graph_selected_async_load async_load = {0};
        bool async_load_started = false;
        const bool async_early_commit =
            async_selected_load &&
            gpu_graph_use_iq2_selected_async_early_commit(g);
        if (ok && async_selected_load) {
            ok = gpu_graph_selected_async_load_start(&async_load,
                                                       g,
                                                       model,
                                                       layer,
                                                       il,
                                                       selected_event,
                                                       gate_expert_bytes,
                                                       down_expert_bytes);
            async_load_started = ok;
        }
        if (ok && async_early_commit) {
            ok = ds4_gpu_flush_commands() != 0;
        }
        if (ok && fuse_shared_gate_up) {
            ok = ds4_gpu_shared_gate_up_swiglu_mxfp8_tensor(g->shared_gate,
                                                             g->shared_up,
                                                             g->shared_mid,
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_gate_shexp->abs_offset,
                                                             layer->ffn_up_shexp->abs_offset,
                                                             DS4_N_EMBD,
                                                             shared_dim,
                                                             g->ffn_norm,
                                                             DS4_SWIGLU_CLAMP_EXP) != 0;
        } else if (ok) {
            if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_gate, model->map, model->size,
                                                      layer->ffn_gate_shexp->abs_offset,
                                                      DS4_N_EMBD, shared_dim,
                                                      g->ffn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_up, model->map, model->size,
                                                      layer->ffn_up_shexp->abs_offset,
                                                      DS4_N_EMBD, shared_dim,
                                                      g->ffn_norm, 1) != 0;
            if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up,
                                               shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
        DS4_CUDA_PROFILE_DECODE_STAGE("shared_gate_up");
        if (ok && !fuse_shared_down_hc) {
            ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_out, model->map, model->size,
                                              layer->ffn_down_shexp->abs_offset,
                                              shared_dim, DS4_N_EMBD,
                                              g->shared_mid, 1) != 0;
        }
        DS4_CUDA_PROFILE_DECODE_STAGE("shared_down");
        if (async_load_started) {
            const bool flush_ok = ds4_gpu_flush_commands() != 0;
            const bool finish_ok =
                gpu_graph_selected_async_load_finish(&async_load);
            ok = ok && flush_ok && finish_ok;
        } else if (ok) {
            ok = ds4_gpu_commit_and_wait_selected_readback(selected_event,
                                                           "selected-id shared-overlap") != 0;
        }
        if (ok && !async_load_started) {
            int32_t selected_ids[DS4_MAX_EXPERT_USED];
            ok = ds4_gpu_tensor_read(g->router_selected,
                                     0,
                                     selected_ids,
                                     (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_ids[0])) != 0 &&
                 ds4_gpu_routed_moe_set_selected_override(selected_ids,
                                                          DS4_N_EXPERT_USED) != 0;
            if (ok) {
                const ds4_gpu_stream_expert_table table =
                    graph_stream_expert_table_make(model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
                ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                            &table,
                            selected_ids,
                            DS4_N_EXPERT_USED) != 0;
            }
        }
        if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                     g->routed_gate,
                                                     g->routed_up,
                                                     g->routed_mid,
                                                     g->routed_down,
                                                     tensor_map_base(model, layer->ffn_gate_exps),
                                                     tensor_map_size(model, layer->ffn_gate_exps),
                                                     layer->ffn_gate_exps->abs_offset,
                                                     layer->ffn_up_exps->abs_offset,
                                                     layer->ffn_down_exps->abs_offset,
                                                     layer->ffn_gate_exps->type,
                                                     layer->ffn_down_exps->type,
                                                     gate_expert_bytes, gate_row_bytes,
                                                     down_expert_bytes, down_row_bytes,
                                                     (uint32_t)expert_in_dim,
                                                     (uint32_t)down_in_dim,
                                                     (uint32_t)routed_out_dim,
                                                     g->router_selected, g->router_weights,
                                                     ds4_layer_n_expert(il),
                                                     DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm,
                                                     il) != 0;
        DS4_CUDA_PROFILE_DECODE_STAGE("routed_moe");
        if (ok) {
            gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                          (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
            gpu_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
        }
        if (ok && fuse_shared_down_hc) {
            ok = ds4_gpu_shared_down_hc_expand_mxfp8_tensor(g->after_ffn_hc,
                                                             g->shared_out,
                                                             model->map,
                                                             model->size,
                                                             layer->ffn_down_shexp->abs_offset,
                                                             shared_dim,
                                                             DS4_N_EMBD,
                                                             g->shared_mid,
                                                             g->routed_out,
                                                             g->after_attn_hc,
                                                             g->hc_split,
                                                             DS4_N_EMBD,
                                                             DS4_N_HC) != 0;
        }
        DS4_CUDA_PROFILE_DECODE_STAGE("shared_down");
        if (ok) {
            gpu_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
        }
        if (ok && keep_ffn_out) {
            ok = gpu_graph_ensure_ffn_out(g) &&
                 ds4_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
        }
        if (ok && keep_ffn_out) {
            gpu_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
        }
        if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
            ok = gpu_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
        }
        if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
            ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                            g->ffn_out,
                                            g->after_attn_hc,
                                            g->hc_post,
                                            g->hc_comb,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        } else if (ok && !fuse_shared_down_hc) {
            ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                      g->routed_out,
                                                      g->shared_out,
                                                      g->after_attn_hc,
                                                      g->hc_split,
                                                      DS4_N_EMBD,
                                                      DS4_N_HC) != 0;
        }
        DS4_CUDA_PROFILE_DECODE_STAGE("ffn_hc_post");
        if (ok) {
            gpu_graph_debug_dump_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
        }
        return ok;
    }
    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
                                                 g->routed_gate,
                                                 g->routed_up,
                                                 g->routed_mid,
                                                 g->routed_down,
                                                 tensor_map_base(model, layer->ffn_gate_exps),
                                                 tensor_map_size(model, layer->ffn_gate_exps),
                                                 layer->ffn_gate_exps->abs_offset,
                                                 layer->ffn_up_exps->abs_offset,
                                                 layer->ffn_down_exps->abs_offset,
                                                 layer->ffn_gate_exps->type,
                                                 layer->ffn_down_exps->type,
                                                 gate_expert_bytes, gate_row_bytes,
                                                 down_expert_bytes, down_row_bytes,
                                                 (uint32_t)expert_in_dim,
                                                 (uint32_t)down_in_dim,
                                                 (uint32_t)routed_out_dim,
                                                 g->router_selected, g->router_weights,
                                                 ds4_layer_n_expert(il),
                                                 DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm,
                                                 il) != 0;
    DS4_CUDA_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_gate_clamped", g->routed_gate,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        gpu_graph_debug_dump_tensor("ffn_moe_up_clamped", g->routed_up,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", g->routed_mid,
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_down", g->routed_down,
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_moe_out", g->routed_out, DS4_N_EMBD, il, pos);
    }
    if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_mxfp8_tensor(g->shared_gate,
                                                         g->shared_up,
                                                         g->shared_mid,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_gate_shexp->abs_offset,
                                                         layer->ffn_up_shexp->abs_offset,
                                                         DS4_N_EMBD,
                                                         shared_dim,
                                                         g->ffn_norm,
                                                         DS4_SWIGLU_CLAMP_EXP) != 0;
    } else {
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_gate, model->map, model->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_up, model->map, model->size,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  DS4_N_EMBD, shared_dim,
                                                  g->ffn_norm, 1) != 0;
        if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up,
                                           shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("shared_gate_up");
    if (ok && fuse_shared_down_hc) {
        ok = ds4_gpu_shared_down_hc_expand_mxfp8_tensor(g->after_ffn_hc,
                                                         g->shared_out,
                                                         model->map,
                                                         model->size,
                                                         layer->ffn_down_shexp->abs_offset,
                                                         shared_dim,
                                                         DS4_N_EMBD,
                                                         g->shared_mid,
                                                         g->routed_out,
                                                         g->after_attn_hc,
                                                         g->hc_split,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_matmul_mxfp8_tensor(g->shared_out, model->map, model->size,
                                          layer->ffn_down_shexp->abs_offset,
                                          shared_dim, DS4_N_EMBD,
                                          g->shared_mid, 1) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        gpu_graph_debug_dump_tensor("ffn_shexp", g->shared_out, DS4_N_EMBD, il, pos);
    }
    if (ok && keep_ffn_out) {
        ok = gpu_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(g->ffn_out, g->shared_out, g->routed_out, DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        gpu_graph_debug_dump_tensor("ffn_out", g->ffn_out, DS4_N_EMBD, il, pos);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = gpu_graph_apply_directional_steering_ffn(g, g->ffn_out, il, 1);
    }
    if (ok && gpu_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(g->after_ffn_hc,
                                        g->ffn_out,
                                        g->after_attn_hc,
                                        g->hc_post,
                                        g->hc_comb,
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
    } else if (ok && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(g->after_ffn_hc,
                                                  g->routed_out,
                                                  g->shared_out,
                                                  g->after_attn_hc,
                                                  g->hc_split,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_CUDA_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_CUDA_PROFILE_DECODE_STAGE
    if (ok) {
        gpu_graph_debug_dump_tensor("hc_ffn_post", g->after_ffn_hc, hc_dim, il, pos);
    }
    if (ok) gpu_graph_capture_dspark_target_hc(g, il);
    return ok;
}

void gpu_graph_capture_dspark_target_hc(ds4_gpu_graph *g, uint32_t il) {
    int slot = -1;
    for (int i = 0; i < 3; i++) {
        if (il == g->dspark_target_layer_ids[i]) { slot = i; break; }
    }
    if (slot < 0 || !g->dspark_target_h[slot]) return;

    ds4_gpu_dspark_hc_mean_reduce(g->dspark_target_h[slot],
                                   g->after_ffn_hc,
                                   DS4_N_EMBD, DS4_N_HC);
}



bool gpu_graph_dspark_project_main_x(
        ds4_gpu_graph          *g,
        const ds4_model         *dspark_model,
        const ds4_dspark_weights *w) {
    const uint64_t E = DS4_N_EMBD;
    const uint64_t concat_dim = 3ull * E;

    for (int i = 0; i < 3; i++) {
        if (!g->dspark_target_h[i]) return false;
    }

    /* Persistent scratch: this runs up to 5x per fused spec step and each
     * cudaMalloc/cudaFree pair serializes the device. */
    ds4_gpu_tensor *target_concat = g->dspark_concat;
    ds4_gpu_tensor *proj_out = g->dspark_proj_out;
    if (!target_concat || !proj_out) return false;

    bool ok = true;
    for (int i = 0; i < 3; i++) {
        ok = ds4_gpu_tensor_copy(target_concat, (uint64_t)i * E * sizeof(float),
                                 g->dspark_target_h[i], 0, E * sizeof(float)) != 0;
        if (!ok) break;
    }

    if (ok) {
        ok = ds4_gpu_matmul_mxfp8_tensor(proj_out,
                                          dspark_model->map,
                                          dspark_model->size,
                                          w->main_proj->abs_offset,
                                          concat_dim, E,
                                          target_concat, 1) != 0;
    }

    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            proj_out,
                                            dspark_model->map,
                                            dspark_model->size,
                                            w->main_norm->abs_offset,
                                            (uint32_t)E,
                                            DS4_RMS_EPS) != 0;
    }

    return ok;
}

void gpu_graph_dspark_seed_draft_kv(
        ds4_gpu_graph          *g,
        const ds4_model         *dspark_model,
        const ds4_dspark_weights *w,
        uint32_t                 n_rows) {
    const uint64_t kv_bytes = (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    /* Persistent scratch (dspark_seed_*): the fused loop seeds up to 5 rows per
     * step across 3 layers; per-call cudaMalloc/cudaFree here was ~9 device-
     * serializing pairs per call. */
    ds4_gpu_tensor *kv_out = g->dspark_seed_kv;
    ds4_gpu_tensor *kv_norm = g->dspark_seed_norm;
    ds4_gpu_tensor *kv_rot = g->dspark_seed_rot;
    if (!kv_out || !kv_norm || !kv_rot) return;
    for (int li = 0; li < 3; li++) {
        if (!ds4_gpu_matmul_mxfp8_tensor(kv_out,
                                          dspark_model->map,
                                          dspark_model->size,
                                          w->layer[li].attn_kv->abs_offset,
                                          DS4_N_EMBD, DS4_N_HEAD_DIM,
                                          g->dspark_main_x, 1)) {
            continue;
        }
        if (!ds4_gpu_rms_norm_weight_tensor(kv_norm, kv_out,
                                             dspark_model->map,
                                             dspark_model->size,
                                             w->layer[li].attn_kv_a_norm->abs_offset,
                                             DS4_N_HEAD_DIM, DS4_RMS_EPS)) {
            continue;
        }
        /* Seed one KV row per committed position.  Each row is RoPE'd at its OWN
         * sequence position (dspark_n_raw[li]) and fp8-rounded, matching the
         * draft-forward KV and the DSparkAttention reference (main_kv is rotated
         * at start_pos).  kv_norm holds the un-rotated vector; we rotate a fresh
         * copy per position so multi-row seeds (accepted drafts) land at distinct
         * positions rather than all sharing the first row's rotation. */
        for (uint32_t i = 0; i < n_rows; i++) {
            const uint32_t pos = g->dspark_n_raw[li];
            const uint32_t row = pos % DS4_DSPARK_DRAFT_WINDOW;
            ds4_gpu_tensor_copy(kv_rot, 0, kv_norm, 0, kv_bytes);
            ds4_gpu_rope_tail_tensor(kv_rot, 1, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
                                     pos, 0, false,
                                     (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
                                     DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW);
            ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv_rot, 1, DS4_N_HEAD_DIM, DS4_N_ROT);
            ds4_gpu_tensor_copy(g->dspark_raw_cache[li],
                                (uint64_t)row * kv_bytes,
                                kv_rot, 0, kv_bytes);
            g->dspark_n_raw[li]++;
        }
    }
}

bool gpu_graph_dspark_draft_forward(
        ds4_gpu_graph          *g,
        const ds4_model         *base_model,
        const ds4_weights       *base_weights,
        const ds4_model         *dspark_model,
        const ds4_dspark_weights *w,
        ds4_gpu_tensor         *base_logits_out,
        const int32_t            draft_ids[],
        uint32_t                n_draft) {
    if (!g || !base_model || !base_weights || !dspark_model || !w ||
        !base_logits_out || n_draft == 0 || n_draft > 16 ||
        n_draft > g->prefill_cap)
        return false;

    if (!base_weights->token_embd || !base_weights->output)
        return false;

    /* Embed N draft tokens via main model's F16 token_embd → HC-expand */
    ds4_gpu_tensor *tokens_t = ds4_gpu_tensor_alloc((uint64_t)n_draft * sizeof(int32_t));
    if (!tokens_t) return false;
    if (!ds4_gpu_tensor_write(tokens_t, 0, draft_ids, (uint64_t)n_draft * sizeof(int32_t))) {
        ds4_gpu_tensor_free(tokens_t);
        return false;
    }
    bool ok = ds4_gpu_embed_tokens_hc_tensor(g->batch_cur_hc,
                                              tokens_t,
                                              base_model->map,
                                              base_model->size,
                                              base_weights->token_embd->abs_offset,
                                              DS4_N_VOCAB,
                                              n_draft,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    ds4_gpu_tensor_free(tokens_t);
    if (!ok) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;

    const int prev_comp = g->comp_ratio_override;
    g->comp_ratio_override = 0;

    for (uint32_t li = 0; li < 3 && ok; li++) {
        const ds4_layer_weights *layer = &w->layer[li];
        const uint32_t raw_cap = DS4_DSPARK_DRAFT_WINDOW;
        const uint32_t q_rank = (uint32_t)layer->attn_q_a->dim[1];
        /* Draft queries/KV sit at the frontier (the current main_kv was seeded at
         * dspark_n_raw[li]-1 just before this forward), so RoPE them at the real
         * position -- NOT 0.  The reference DSparkAttention rotates draft Q/KV at
         * start_pos+1 while the seeded main_kv is at start_pos. */
        const uint32_t pos0 = g->dspark_n_raw[li];

        /* --- HC pre-processing --- */
        /* Create views from batch working set */
        ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            g->batch_hc_mix, 0, (uint64_t)n_draft * mix_hc * sizeof(float));
        ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            g->batch_hc_split, 0, (uint64_t)n_draft * mix_hc * sizeof(float));
        ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            g->batch_ffn_cur, 0, (uint64_t)n_draft * DS4_N_EMBD * sizeof(float));
        ok = hc_mix_view && hc_split_view && attn_cur_view;
        /* RMS norm: flat HC from batch_cur_hc */
        if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(
            g->batch_flat_hc, g->batch_cur_hc,
            (uint32_t)hc_dim, n_draft, DS4_RMS_EPS) != 0;
        /* HC → mix projection */
        if (ok) ok = gpu_graph_matmul_plain_tensor(
            hc_mix_view, dspark_model,
            layer->hc_attn_fn,
            hc_dim, mix_hc, g->batch_flat_hc, n_draft);
        /* HC split + weighted sum → attn_cur (E-dim) */
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(
            attn_cur_view, hc_split_view, hc_mix_view,
            g->batch_cur_hc,
            dspark_model->map, dspark_model->size,
            layer->hc_attn_scale->abs_offset,
            layer->hc_attn_base->abs_offset,
            DS4_N_EMBD, DS4_N_HC,
            DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS) != 0;
        /* Input RMS norm → batch_attn_norm */
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_attn_norm, attn_cur_view,
            dspark_model->map, dspark_model->size,
            layer->attn_norm->abs_offset,
            DS4_N_EMBD, n_draft, DS4_RMS_EPS) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_norm", g->batch_attn_norm,
                                             (uint64_t)n_draft * DS4_N_EMBD, li, pos0);
        /* --- Q projection --- */
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(
            g->batch_qr, dspark_model->map, dspark_model->size,
            layer->attn_q_a->abs_offset,
            DS4_N_EMBD, q_rank, g->batch_attn_norm, n_draft) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_qr_norm, g->batch_qr,
            dspark_model->map, dspark_model->size,
            layer->attn_q_a_norm->abs_offset,
            q_rank, n_draft, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(
            g->batch_q, dspark_model->map, dspark_model->size,
            layer->attn_q_b->abs_offset,
            q_rank, DS4_N_HEAD * DS4_N_HEAD_DIM,
            g->batch_qr_norm, n_draft) != 0;
        /* Q head-norm + RoPE */
        if (ok) ok = ds4_gpu_head_rms_norm_rope_tail_tensor(
            g->batch_q, n_draft,
            DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
            pos0, 0, false,
            (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_RMS_EPS) != 0;

        /* --- KV projection --- */
        if (ok) ok = ds4_gpu_matmul_mxfp8_tensor(
            g->batch_kv_raw, dspark_model->map, dspark_model->size,
            layer->attn_kv->abs_offset,
            DS4_N_EMBD, DS4_N_HEAD_DIM,
            g->batch_attn_norm, n_draft) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
            g->batch_kv, g->batch_kv_raw,
            dspark_model->map, dspark_model->size,
            layer->attn_kv_a_norm->abs_offset,
            DS4_N_HEAD_DIM, n_draft, DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(
            g->batch_kv, n_draft,
            DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT,
            pos0, 0, false,
            (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(
            g->batch_kv, n_draft, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;

        /* --- Store draft KV transiently in ring buffer for attention --- */
        const uint32_t saved_n_raw = g->dspark_n_raw[li];
        const uint32_t kv_store_pos = saved_n_raw % raw_cap;
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(
            g->dspark_raw_cache[li], g->batch_kv,
            raw_cap, kv_store_pos, n_draft, DS4_N_HEAD_DIM) != 0;
        const uint32_t vis_raw = saved_n_raw + n_draft;
        const uint32_t cap_raw = vis_raw < raw_cap ? vis_raw : raw_cap;
        const uint32_t raw_start = vis_raw > raw_cap
            ? (vis_raw - raw_cap) % raw_cap : 0;

        /* --- Non-causal raw batch attention ---
         * Queries are at positions [saved_n_raw, saved_n_raw+n_draft).
         * Visible raw entries span [0, vis_raw) — all cached + current draft rows. */
        if (ok) ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(
            g->batch_heads,
            dspark_model->map, dspark_model->size,
            layer->attn_sinks->abs_offset,
            g->batch_q, g->dspark_raw_cache[li],
            n_draft, saved_n_raw,
            cap_raw, raw_cap, raw_start,
            0,
            DS4_N_HEAD, DS4_N_HEAD_DIM,
            1) != 0;

        if (ok) gpu_graph_debug_dump_tensor("dsp_heads", g->batch_heads,
                                             (uint64_t)n_draft * DS4_N_HEAD * DS4_N_HEAD_DIM, li, pos0);
        /* Inverse-rotate the attention output's rope dims before the o
         * projection (reference: apply_rotary_emb(o, freqs_cis, inverse=True);
         * the verify/prefill path does the same via its "kqv_back" rope).
         * This call was MISSING here: wo_a/wo_b consumed position-rotated
         * rope dims -- 64/512 dims per head scrambled in every drafter block,
         * the ~30-point acceptance bug (#4). Verified vs the torch reference:
         * the engine attention output matches exactly WITHOUT inverse rope
         * (cos 0.99999) and diverges with it (cos 0.57), so everything
         * downstream of this line computed on corrupted features. */
        if (ok) ok = ds4_gpu_rope_tail_tensor(
            g->batch_heads, n_draft,
            DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT,
            pos0, 0, true,
            (float)DS4_ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f,
            DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
        /* --- Attention output projection (LoRA grouped) --- */
        if (ok) ok = ds4_gpu_attention_output_batch_tensor(
            g->batch_attn_out, g->batch_attn_low,
            dspark_model->map, dspark_model->size,
            layer->attn_output_a->abs_offset,
            layer->attn_output_b->abs_offset,
            group_dim, rank, n_groups, DS4_N_EMBD,
            g->batch_heads, n_draft) != 0;
        if (ok) gpu_graph_debug_dump_tensor("dsp_attn_out", g->batch_attn_out,
                                             (uint64_t)n_draft * DS4_N_EMBD, li, pos0);

        /* --- HC expand + split → batch_after_attn_hc --- */
        /* View sized to the real draft count: the CUDA side infers n_tokens from
         * out_hc->bytes, and the raw batch tensor is allocated at prefill capacity
         * (4096) -- passing it unviewed made every drafter block expand the FULL
         * capacity, ~2.8 ms per call x3 blocks = ~8 ms of pure waste per spec step. */
        if (ok) {
            ds4_gpu_tensor *after_attn_view = ds4_gpu_tensor_view(
                g->batch_after_attn_hc, 0,
                (uint64_t)n_draft * DS4_N_HC * DS4_N_EMBD * sizeof(float));
            ok = after_attn_view &&
                 ds4_gpu_hc_expand_split_tensor(
                     after_attn_view, g->batch_attn_out,
                     g->batch_cur_hc, g->batch_hc_split,
                     DS4_N_EMBD, DS4_N_HC) != 0;
            ds4_gpu_tensor_free(after_attn_view);
        }

        /* --- FFN batch (reuses existing function) --- */
        if (ok) ok = gpu_graph_encode_layer_ffn_batch(
            g, dspark_model, layer, li, pos0, n_draft);

        if (ok) gpu_graph_debug_dump_tensor("dsp_after_attn_hc", g->batch_after_attn_hc,
                                             (uint64_t)n_draft * DS4_N_HC * DS4_N_EMBD, li, pos0);
        /* --- HC swap for next layer --- */
        if (ok) {
            ds4_gpu_tensor *tmp = g->batch_cur_hc;
            g->batch_cur_hc = g->batch_next_hc;
            g->batch_next_hc = tmp;
        }
        if (ok) gpu_graph_debug_dump_tensor("dsp_block_out", g->batch_cur_hc,
                                             (uint64_t)n_draft * DS4_N_HC * DS4_N_EMBD, li, pos0);
        /* Draft KV is transient; dspark_n_raw remains at the persistent count.
         * Committed positions are seeded via gpu_graph_dspark_seed_draft_kv(). */

        /* Views over the batch working set are per-iteration host structs;
         * free them each layer or they leak on every speculative block. */
        ds4_gpu_tensor_free(hc_mix_view);
        ds4_gpu_tensor_free(hc_split_view);
        ds4_gpu_tensor_free(attn_cur_view);
    }

    g->comp_ratio_override = prev_comp;

    /* Batch output head → N-token logits in g->spec_logits.  Use the DSpark
     * drafter's OWN hc_head + norm (dspark.2.*) with the shared vocab head, NOT
     * the main model's output head (which was corrupting the draft logits). */
    if (ok) {
        ok = gpu_graph_encode_dspark_output_head_batch(
            g, dspark_model, w, base_model, base_weights, n_draft, DS4_N_VOCAB);
    }

    /* The output head already wrote into g->spec_logits.  Callers that pass a
     * distinct buffer get a copy; the session passes g->spec_logits itself, so
     * skip the self-copy (a same-buffer cudaMemcpy is undefined). */
    if (ok && base_logits_out && base_logits_out != g->spec_logits) {
        const uint64_t logits_bytes = (uint64_t)n_draft * DS4_N_VOCAB * sizeof(float);
        ok = ds4_gpu_tensor_copy(base_logits_out, 0,
                                  g->spec_logits, 0,
                                  logits_bytes) != 0;
    }

    return ok;
}

/* Encode the final HC collapse, output norm, and vocab projection on Metal. */
bool gpu_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->output_pre,
                                                 (const ds4_model *)model,
                                                 weights->output_hc_fn,
                                                 hc_dim,
                                                 DS4_N_HC,
                                                 g->flat_hc,
                                             1) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_pre", g->output_pre, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc_weights", g->output_weights, DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_hc", g->output_embd, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
        gpu_graph_debug_dump_tensor("result_norm", g->output_norm, DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) {
        if (weights->output->type == DS4_TENSOR_BF16)
            ok = ds4_gpu_matmul_bf16_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
        else
            ok = ds4_gpu_matmul_mxfp8_tensor(g->logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
    }
    if (ok) {
        gpu_graph_debug_dump_tensor("result_output", g->logits, vocab_dim, DS4_N_LAYER, 0);
    }
    return ok;
}



/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and MXFP8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
bool gpu_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;
    ds4_gpu_tensor *logits = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(g->batch_hc_mix,
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(g->batch_hc_split,
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm,
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    logits = ds4_gpu_tensor_view(g->spec_logits,
                                   0,
                                   (uint64_t)n_tokens * vocab_dim * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm && logits;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc,
                                                      g->batch_cur_hc,
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(output_pre,
                                                 (const ds4_model *)model,
                                                 weights->output_hc_fn,
                                             hc_dim,
                                             DS4_N_HC,
                                             g->batch_flat_hc,
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  g->batch_cur_hc,
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) {
        if (weights->output->type == DS4_TENSOR_BF16)
            ok = ds4_gpu_matmul_bf16_tensor(logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, output_norm, n_tokens) != 0;
        else
            ok = ds4_gpu_matmul_mxfp8_tensor(logits, model->map, model->size,
                                            weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, output_norm, n_tokens) != 0;
    }

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

/* DSpark drafter output head.  Collapses the drafter's final HC with the DSpark
 * block's OWN head (dspark.2.hc_head_fn/scale/base) and norm (dspark.2.norm),
 * then projects to vocab with the SHARED main output head (self.head in the
 * reference).  The plain gpu_graph_encode_output_head_batch used the MAIN model's
 * output_hc and output_norm weights for the drafter -- wrong weights that
 * corrupted the draft base logits (base0_hit was ~29%). */
bool gpu_graph_encode_dspark_output_head_batch(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        const ds4_model          *base_model,
        const ds4_weights        *bw,
        uint32_t                  n_tokens,
        uint64_t                  vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = ds4_gpu_tensor_view(g->batch_hc_mix, 0, (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights = ds4_gpu_tensor_view(g->batch_hc_split, 0, (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd = ds4_gpu_tensor_view(g->batch_ffn_cur, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm = ds4_gpu_tensor_view(g->batch_ffn_norm, 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits = ds4_gpu_tensor_view(g->spec_logits, 0, (uint64_t)n_tokens * vocab_dim * sizeof(float));
    bool ok = output_pre && output_weights && output_embd && output_norm && logits;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(g->batch_flat_hc, g->batch_cur_hc,
                                                     (uint32_t)hc_dim, n_tokens, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(output_pre, dspark_model, dw->hc_head_fn,
                                               hc_dim, DS4_N_HC, g->batch_flat_hc, n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights, output_pre,
                                                  dspark_model->map, dspark_model->size,
                                                  dw->hc_head_scale->abs_offset,
                                                  dw->hc_head_base->abs_offset,
                                                  DS4_N_HC, DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd, g->batch_cur_hc, output_weights,
                                                DS4_N_EMBD, DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm, output_embd,
                                                     dspark_model->map, dspark_model->size,
                                                     dw->final_norm->abs_offset,
                                                     DS4_N_EMBD, n_tokens, DS4_RMS_EPS) != 0;
    if (ok) {
        if (bw->output->type == DS4_TENSOR_BF16)
            ok = ds4_gpu_matmul_bf16_tensor(logits, base_model->map, base_model->size,
                                            bw->output->abs_offset, DS4_N_EMBD, vocab_dim,
                                            output_norm, n_tokens) != 0;
        else
            ok = ds4_gpu_matmul_mxfp8_tensor(logits, base_model->map, base_model->size,
                                             bw->output->abs_offset, DS4_N_EMBD, vocab_dim,
                                             output_norm, n_tokens) != 0;
    }
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}



bool gpu_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_FP8_E4M3) {
        return ds4_gpu_matmul_mxfp8_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    fprintf(stderr, "ds4: plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}



bool gpu_graph_matmul_mxfp8_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    const bool ok = ds4_gpu_matmul_mxfp8_tensor(out,
                                                 model->map,
                                                 model->size,
                                                 w->abs_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tok) != 0;
    return ok;
}



bool gpu_graph_encode_output_head_mtp(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(g->flat_hc, g->cur_hc, (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = gpu_graph_matmul_plain_tensor(g->output_pre, mtp_model, mtp->hc_head_fn,
                                                 hc_dim, DS4_N_HC, g->flat_hc, 1);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(g->output_weights,
                                                    g->output_pre,
                                                    mtp_model->map,
                                                    mtp_model->size,
                                                    mtp->hc_head_scale->abs_offset,
                                                    mtp->hc_head_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(g->output_embd,
                                                  g->cur_hc,
                                                  g->output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                  g->output_embd,
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) {
        if (base_weights->output->type == DS4_TENSOR_BF16)
            ok = ds4_gpu_matmul_bf16_tensor(g->logits, base_model->map, base_model->size,
                                            base_weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
        else
            ok = ds4_gpu_matmul_mxfp8_tensor(g->logits, base_model->map, base_model->size,
                                            base_weights->output->abs_offset, DS4_N_EMBD,
                                            vocab_dim, g->output_norm, 1) != 0;
    }
    return ok;
}



/* =========================================================================
 * Metal Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * Metal tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */

static void gpu_graph_trace_layer_stages(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        const float            *cpu_in_hc,
        uint32_t                il,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t shared_in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];

    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float post[4];
    float comb[16];
    float ffn_post[4];
    float ffn_comb[16];
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    const uint64_t shared_blocks = (shared_in_dim + 31) / 32;
    int8_t *shared_xq = xmalloc((size_t)shared_blocks * 32);
    float *shared_xscale = xmalloc((size_t)shared_blocks * sizeof(float));
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_in_hc, cpu_attn_cur, post, comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_one(cpu_heads, model, layer, cpu_q, cpu_kv);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_in_hc, post, comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, ffn_post, ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    quantize_q8_0_activation(cpu_ffn_norm, shared_xq, shared_xscale, shared_in_dim);
    matvec_q8_0_pair_prequant(cpu_shared_gate,
                              cpu_shared_up,
                              model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              shared_xq,
                              shared_xscale);
    swiglu(cpu_shared_mid, cpu_shared_gate, cpu_shared_up, shared_dim, DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0(cpu_shared, model, layer->ffn_down_shexp, cpu_shared_mid);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc, cpu_ffn_out, cpu_after_attn_hc, ffn_post, ffn_comb, DS4_N_EMBD, DS4_N_HC);

    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];

    bool ok = ds4_gpu_tensor_read(g->attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_gate, 0, gpu_shared_gate, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_up, 0, gpu_shared_up, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_mid, 0, gpu_shared_mid, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
              ds4_gpu_tensor_read(g->router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
              ds4_gpu_tensor_read(g->routed_mid, 0, gpu_routed_mid_all, (uint64_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(g->cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0;

    if (ok) {
        fprintf(stderr,
                "ds4: Metal stage layer %u attn_cur=%g/%g attn_norm=%g/%g q=%g/%g kv=%g/%g attn_out=%g/%g after_attn_hc=%g/%g ffn_cur=%g/%g ffn_norm=%g/%g shared=%g/%g router_w=%g routed=%g/%g ffn_out=%g/%g after_ffn_hc=%g/%g\n",
                il,
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD), rms_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD), rms_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim), rms_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM), rms_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD), rms_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim), rms_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD), rms_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD), rms_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD), rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD), rms_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim), rms_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim));
        fprintf(stderr,
                "ds4: Metal shared layer %u gate=%g/%g up=%g/%g mid=%g/%g down=%g/%g\n",
                il,
                max_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim), rms_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim),
                max_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim), rms_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim),
                max_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim), rms_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD));
        fprintf(stderr,
                "ds4: Metal routed layer %u mid=%g/%g out=%g/%g\n",
                il,
                max_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                rms_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal stage layer %u router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    il,
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
    }

    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_routed_mid_all);
    free(gpu_shared);
    free(gpu_shared_mid);
    free(gpu_shared_up);
    free(gpu_shared_gate);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(shared_xscale);
    free(shared_xq);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_routed);
    free(cpu_shared);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_ffn_norm);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_kv);
    free(cpu_qr_norm);
    free(cpu_q);
    free(cpu_attn_norm);
    free(cpu_attn_cur);
}



int gpu_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        bool               quality) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];

    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc(&g, weights, layer);
    g.quality = quality;
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = gpu_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        ds4_gpu_tensor *embedded_hc = g.cur_hc;
        g.cur_hc = g.after_ffn_hc;
        g.after_ffn_hc = embedded_hc;
    }
    if (ok) ok = gpu_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok) {
        ok = ds4_gpu_tensor_read(g.after_ffn_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_cur, 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_norm, 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.q, 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.kv, 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.layer_raw_cache[0], 0, gpu_raw, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.attn_out, 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.after_attn_hc, 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_cur, 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_norm, 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.shared_out, 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.router_selected, 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_gpu_tensor_read(g.router_weights, 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_gpu_tensor_read(g.routed_out, 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.ffn_out, 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.cur_hc, 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: Metal graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("metal graph q", gpu_q, q_dim);
        print_vec_stats("metal graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("metal graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: Metal graph test failed while encoding first decode stages\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph test failure also failed\n");
        }
    }

    gpu_graph_free(&g);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}



int gpu_graph_first_token_full_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        bool               quality) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: full Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = weights->output->dim[1];
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));

    forward_first_token_cpu(cpu_hc, model, weights, token);
    output_logits_one(cpu_logits, model, weights, cpu_hc);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc(&g, weights, &weights->layer[0]);
    g.quality = quality;
    const bool trace_layers = getenv("DS4_CUDA_GRAPH_TRACE_LAYERS") != NULL;
    if (trace_layers && ok) {
        g.materialize_ffn_out = true;
        const bool teacher_force = getenv("DS4_CUDA_GRAPH_TEACHER_FORCE") != NULL;
        const char *stage_layer_env = getenv("DS4_CUDA_GRAPH_TRACE_STAGE_LAYER");
        const long stage_layer = stage_layer_env ? strtol(stage_layer_env, NULL, 10) : -1;
        float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        float *cpu_cur = xmalloc((size_t)hc_dim * sizeof(float));
        float *cpu_next = xmalloc((size_t)hc_dim * sizeof(float));

        embed_token_f16(model, weights, token, plain);
        hc_from_plain_embedding(cpu_cur, plain, DS4_N_EMBD, DS4_N_HC);
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            if (teacher_force) {
                ok = ds4_gpu_tensor_write(g.cur_hc, 0, cpu_cur, hc_dim * sizeof(float)) != 0;
            }
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                       il, 0, g.layer_raw_cache[il], g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
            if (ok) ok = ds4_gpu_end_commands() != 0;

            layer_forward_self_one(cpu_next, model, &weights->layer[il], cpu_cur, il, 0, token);
            if (ok) ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0;
            if (ok) {
                fprintf(stderr,
                        "ds4: Metal full graph layer %u%s hc_max=%g hc_rms=%g\n",
                        il,
                        teacher_force ? " teacher" : "",
                        max_abs_diff(cpu_next, gpu_hc, hc_dim),
                        rms_abs_diff(cpu_next, gpu_hc, hc_dim));
                if (stage_layer == (long)il) {
                    gpu_graph_trace_layer_stages(&g, model, &weights->layer[il], cpu_cur, il, token);
                }
            }
            float *ctmp = cpu_cur;
            cpu_cur = cpu_next;
            cpu_next = ctmp;
        }

        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = gpu_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;

        free(cpu_next);
        free(cpu_cur);
        free(plain);
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(g.cur_hc,
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = gpu_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                 il, 0, g.layer_raw_cache[il],
                                                 g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = g.cur_hc;
            g.cur_hc = g.after_ffn_hc;
            g.after_ffn_hc = tmp;
        }

        if (ok) ok = gpu_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_read(g.cur_hc, 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.logits, 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        const uint64_t cpu_top = argmax_f32(cpu_logits, vocab_dim);
        const uint64_t gpu_top = argmax_f32(gpu_logits, vocab_dim);
        fprintf(stderr,
                "ds4: Metal full first-token graph diffs: final_hc_max=%g final_hc_rms=%g logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                rms_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                rms_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
    } else {
        fprintf(stderr, "ds4: Metal full first-token graph test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after full graph failure also failed\n");
        }
    }

    gpu_graph_free(&g);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_hc);
    free(cpu_hc);
    return ok ? 0 : 1;
}



/* =========================================================================
 * Metal Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing Metal backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

uint32_t gpu_graph_token_split_after_layers(void) {
    uint32_t split_after_layers = 4;
    const char *split_env = getenv("DS4_CUDA_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }
    return split_after_layers;
}

