#include "ds4_engine_internal.h"



bool gpu_graph_alloc(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    return gpu_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA, 1, false);
}






uint32_t gpu_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}



uint32_t gpu_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}



/* Capture the verifier prefix after the first speculative token.
 *
 * Exact MTP speculation is only profitable if partial accepts are cheap.  The
 * target verifier computes two draft tokens together; if only the first token
 * is accepted, replaying a one-token verifier throws away most of the gain.
 * For compressed-attention layers the mutable frontier is just the small
 * compressor state plus append counters, so we save that prefix-1 state while
 * the N=2 verifier is already stepping the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
bool gpu_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}



bool gpu_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}



uint32_t gpu_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g) {
    (void)g;
    static int parsed = -1;
    static uint32_t cached = 0;
    if (parsed < 0) {
        parsed = 0;
        const char *env = getenv("DS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            while (end && isspace((unsigned char)*end)) end++;
            if (end != env && end && *end == '\0' &&
                (v == 64ul || v == 128ul || v == 256ul || v == 512ul ||
                 v == 1024ul || v == 2048ul || v == 4096ul)) {
                cached = (uint32_t)v;
                parsed = 1;
            } else {
                fprintf(stderr,
                        "ds4: invalid DS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD=%s; "
                        "expected 64, 128, 256, 512, 1024, 2048, or 4096\n",
                        env);
            }
        }
    }
    if (parsed > 0) return cached;

    /* Keep dense attention longer than the legacy 512-row window by default.
     * Around the 2K frontier the sparse path's score/top-k setup dominates
     * the smaller attention scan, while larger contexts benefit from sparse
     * indexed attention.  This threshold changes only the implementation used
     * to consume the compressed rows; it must not lower the 512-row indexer
     * selection defined by DS4_N_INDEXER_TOP_K. */
    return 1024u;
}



/* =========================================================================
 * GPU Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * DS4_CUDA_DISABLE_*_FUSION environment switches.
 */

bool gpu_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}

