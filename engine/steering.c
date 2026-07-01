#include "ds4_engine_internal.h"



bool gpu_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_attn_scale != 0.0f;
}



bool gpu_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
    return g && g->directional_steering_dirs && g->directional_steering_ffn_scale != 0.0f;
}



static bool gpu_graph_apply_directional_steering(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !g->directional_steering_dirs || scale == 0.0f) return true;
    return ds4_gpu_directional_steering_project_tensor(x,
                                            g->directional_steering_dirs,
                                            il,
                                            DS4_N_EMBD,
                                            rows,
                                            scale) != 0;
}



bool gpu_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}



bool gpu_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return gpu_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}



static uint64_t gpu_graph_kv_cache_bytes_for_context(uint32_t ctx_size, uint32_t raw_cap) {
    uint64_t bytes = (uint64_t)DS4_N_LAYER *
                     raw_cap *
                     DS4_N_HEAD_DIM *
                     sizeof(float);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * DS4_N_HEAD_DIM *
                 (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
        if (ratio == 4) {
            bytes += comp_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        }
    }
    return bytes;
}



uint64_t gpu_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    uint64_t comp_cap = (uint64_t)(ctx_size / min_ratio + 2u);
    if (comp_cap < 2u) comp_cap = 2u;
    const uint64_t kv_cache_bytes = gpu_graph_kv_cache_bytes_for_context(ctx_size, raw_cap);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    uint64_t bytes = kv_cache_bytes +
                     2ull * comp_cap * prefill_cap * sizeof(float);
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        uint64_t attn_stage_cap = (uint64_t)(prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        bytes += attn_stage_cap * DS4_N_HEAD_DIM * sizeof(float);
    }
    return bytes;
}



ds4_gpu_tensor *gpu_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? ds4_gpu_tensor_alloc_managed(bytes) : ds4_gpu_tensor_alloc(bytes);
}



/* =========================================================================
 * Metal Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_METAL_GRAPH_DUMP_PREFIX is set.  Dumping synchronizes and restarts
 * the command batch, so it is intentionally isolated here.
 */

bool gpu_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    const char *prefix = getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!prefix || !prefix[0]) return false;

    const char *name_env = getenv("DS4_METAL_GRAPH_DUMP_NAME");
    if (name_env && name_env[0] && strstr(name_env, name) == NULL) return false;

    const char *layer_env = getenv("DS4_METAL_GRAPH_DUMP_LAYER");
    if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0 &&
        (uint32_t)strtoul(layer_env, NULL, 10) != il) return false;

    const char *pos_env = getenv("DS4_METAL_GRAPH_DUMP_POS");
    if (pos_env && pos_env[0] && (uint32_t)strtoul(pos_env, NULL, 10) != pos) return false;

    return true;
}

