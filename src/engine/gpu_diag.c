#include "ds4_engine_internal.h"



float rms_abs_diff(const float *a, const float *b, uint64_t n) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        ss += d * d;
    }
    return n ? (float)sqrt(ss / (double)n) : 0.0f;
}



uint64_t argmax_f32(const float *x, uint64_t n) {
    uint64_t best = 0;
    for (uint64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) best = i;
    }
    return best;
}




void print_vec_stats(const char *name, const float *x, uint64_t n) {
    float minv = DS4_POS_INF;
    float maxv = DS4_NEG_INF;
    double ss = 0.0;

    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        ss += (double)v * v;
    }

    printf("%s: min=%g max=%g rms=%g\n",
        name, minv, maxv, sqrt(ss / (double)n));
}



void gpu_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_f32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



void gpu_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_f16 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    uint16_t *hbuf = xmalloc((size_t)n_f16 * sizeof(hbuf[0]));
    float *fbuf = xmalloc((size_t)n_f16 * sizeof(fbuf[0]));
    if (ds4_gpu_tensor_read(t, 0, hbuf, n_f16 * sizeof(hbuf[0])) != 0) {
        for (uint64_t i = 0; i < n_f16; i++) fbuf[i] = f16_to_f32(hbuf[i]);
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, fbuf, n_f16)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(fbuf);
    free(hbuf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



void gpu_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_i32 == 0 || !gpu_graph_debug_wants(name, il, pos)) return;
    const char *prefix = getenv("DS4_CUDA_GRAPH_DUMP_PREFIX");

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume GPU command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}



bool gpu_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return gpu_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           gpu_graph_debug_wants("ffn_out", il, pos);
}



bool gpu_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    if (!g->ffn_out) {
        g->ffn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out != NULL;
}



bool gpu_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    if (!g->batch_ffn_out) {
        g->batch_ffn_out = ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out != NULL;
}



/* =========================================================================
 * GPU Release Graph Allocation.
 * ========================================================================= */

/* Derived capacities and tensor dimensions for one session's GPU graph.
 * Computed by gpu_graph_compute_dims and shared by the allocator
 * (gpu_graph_alloc_raw_cap) and the sizing estimate (gpu_graph_session_bytes)
 * so admission control and the allocator can never disagree about the derived
 * quantities.  The per-buffer byte expressions still appear in both functions;
 * the server's estimate-vs-actual reconciliation (>10% drift warning) is the
 * enforcement that they stay in sync. */
typedef struct {
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t ctx_size;
    uint32_t prefill_cap;
    uint32_t comp_cap;
    uint32_t attn_comp_stage_cap;          /* only meaningful under DS4_ATTN_PACK */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    uint64_t hc_dim;
    uint64_t mix_hc;
    uint64_t q_rank;
    uint64_t q_dim;
    uint64_t low_dim;
    uint64_t shared_dim;
    uint64_t routed_mid_dim;
    uint64_t vocab_dim;
    uint64_t comp_width_max;
    uint64_t indexer_q_dim;
} gpu_graph_dims;

static void gpu_graph_compute_dims(
        gpu_graph_dims *d,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap) {
    memset(d, 0, sizeof(*d));
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    d->raw_cap = raw_cap;
    d->raw_window = raw_window;
    d->ctx_size = ctx_size;
    d->prefill_cap = prefill_cap;

    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    d->comp_cap = ctx_size / min_ratio + 2u;
    if (d->comp_cap < 2u) d->comp_cap = 2u;
    d->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
    if (d->attn_comp_stage_cap < 2u) d->attn_comp_stage_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            d->layer_comp_cap[il] = 0;
        } else {
            d->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (d->layer_comp_cap[il] < 2u) d->layer_comp_cap[il] = 2u;
        }
    }

    d->hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    d->mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    d->q_rank = layer->attn_q_a->dim[1];
    d->q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    d->low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    d->shared_dim = layer->ffn_gate_shexp->dim[1];
    d->routed_mid_dim = layer->ffn_gate_exps->dim[1];
    d->vocab_dim = weights->output ? weights->output->dim[1] : DS4_N_VOCAB;
    d->comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    d->indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
}



/* TRUE total per-session GPU byte cost: the sum of every ds4_gpu_tensor that
 * gpu_graph_alloc_raw_cap (and, when enable_spec, gpu_graph_init_dspark_target)
 * allocates for one session.  This is what admission control must price a
 * session at — the packed KV estimate (ds4_context_memory_estimate_packed)
 * covers only the persistent KV rows plus the indexer score/mask scratch and
 * undercounts the real cost by roughly an order of magnitude (2026-07-13
 * incident: three ctx=65536 slots admitted at 0.5 GiB each consumed the whole
 * GB10 and hard-locked the machine).
 *
 * KEEP IN SYNC with the two allocators below (same order, same expressions).
 * The server reconciles this estimate against the measured allocation delta
 * after every session create and logs a loud warning on >10% drift, so a
 * missed buffer surfaces on the first live run instead of as an
 * under-admission OOM.
 *
 * EXCLUSION LIST — intentionally unaccounted per-session allocations (each
 * negligible, absorbed by DS4_SERVER_MEM_FLOOR_BYTES; do not re-derive):
 *   - the lazy gpu_graph_ensure_ffn_out/gpu_graph_ensure_batch_ffn_out
 *     buffers (only allocated under steering/diagnostics);
 *   - directional-steering dirs (gpu_graph_load_directional_steering,
 *     ~DS4_N_LAYER*DS4_N_EMBD floats — these ARE inside the measured create
 *     delta, so they show up in reconciliation but never in this estimate);
 *   - the spec snapshot/restore descriptor tables that
 *     ds4_gpu_batched_copy_prepare lazily cudaMallocs on the first fused
 *     spec step (~KBs; raw cudaMalloc, outside both this estimate and the
 *     ds4_gpu_tensor byte counter that reconciliation measures). */
uint64_t gpu_graph_session_bytes(
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 raw_cap,
        uint32_t                 ctx_size,
        uint32_t                 prefill_cap,
        bool                     enable_spec) {
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    const uint64_t pc = dz.prefill_cap;
    const uint64_t f32 = sizeof(float);

    /* Persistent KV caches (raw ring, packed attn comp, indexer comp) plus the
     * indexer_scores/comp_mask working pair — shared with the managed-vs-device
     * KV placement policy the allocator itself uses. */
    uint64_t kv_cache_bytes = 0;
    uint64_t total = gpu_graph_context_bytes_for_kv_policy(
            dz.ctx_size, dz.raw_cap, dz.prefill_cap, &kv_cache_bytes);

    /* Per-layer attention/indexer state (and their spec shadows). */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_state = (uint64_t)coff * DS4_N_HEAD_DIM *
                                    coff * ratio * f32;
        total += 2ull * attn_state;                       /* layer_attn_state_kv/score */
        if (enable_spec) total += 2ull * attn_state;      /* spec_attn_state_kv/score */
        if (ratio == 4) {
            const uint64_t index_state = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM *
                                         coff * ratio * f32;
            total += 2ull * index_state;                  /* layer_index_state_kv/score */
            if (enable_spec) total += 2ull * index_state; /* spec_index_state_kv/score */
        }
    }

    /* Single-token graph buffers. */
    total += 2ull * dz.hc_dim * f32;                      /* cur_hc, flat_hc */
    total += 2ull * dz.mix_hc * f32;                      /* hc_mix, hc_split (views free) */
    total += 2ull * DS4_N_EMBD * f32;                     /* attn_cur, attn_norm */
    total += 2ull * dz.q_rank * f32;                      /* qr, qr_norm */
    total += dz.q_dim * f32;                              /* q */
    total += 2ull * DS4_N_HEAD_DIM * f32;                 /* kv_raw, kv */
    total += 2ull * dz.comp_width_max * f32;              /* comp_kv_cur, comp_sc_cur */
    if (gpu_graph_attn_pack_enabled()) {
        total += (uint64_t)dz.attn_comp_stage_cap * DS4_N_HEAD_DIM * f32; /* attn_comp_stage */
        total += (uint64_t)dz.comp_cap * DS4_N_HEAD_DIM * f32;            /* attn_comp_dequant */
    }
    if (gpu_graph_idx_fp4_enabled()) {
        total += (uint64_t)dz.comp_cap * DS4_N_INDEXER_HEAD_DIM * f32;    /* idx_comp_stage */
    }
    total += dz.indexer_q_dim * f32;                      /* indexer_q */
    total += (uint64_t)DS4_N_INDEXER_HEAD * f32;          /* indexer_weights */
    total += (uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
             pc * sizeof(uint32_t);                       /* comp_selected */
    total += dz.q_dim * f32;                              /* heads */
    total += dz.low_dim * f32;                            /* attn_low */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* attn_out */
    total += dz.hc_dim * f32;                             /* after_attn_hc */
    total += 2ull * DS4_N_EMBD * f32;                     /* ffn_cur, ffn_norm */
    total += 3ull * dz.shared_dim * f32;                  /* shared_gate/up/mid */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* shared_out */
    total += 2ull * DS4_N_EXPERT * f32;                   /* router_logits, router_probs */
    total += (uint64_t)DS4_N_EXPERT_USED * (sizeof(int) + f32); /* router_selected/weights */
    total += 3ull * DS4_N_EXPERT_USED * dz.routed_mid_dim * f32; /* routed_gate/up/mid */
    total += (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * f32;     /* routed_down */
    total += (uint64_t)DS4_N_EMBD * f32;                  /* routed_out */
    total += dz.hc_dim * f32;                             /* after_ffn_hc */
    total += 2ull * DS4_N_HC * f32;                       /* output_pre, output_weights */
    total += 2ull * DS4_N_EMBD * f32;                     /* output_embd, output_norm */
    total += dz.vocab_dim * f32;                          /* logits */
    total += pc * sizeof(int32_t);                        /* prefill_tokens */

    /* Batch (prefill working set) buffers — the pc-scaled bulk that dominates
     * the non-KV cost (~4 GiB at pc=4096 on Flash). */
    total += 3ull * pc * dz.hc_dim * f32;                 /* batch_cur/next/flat_hc */
    total += 2ull * pc * dz.mix_hc * f32;                 /* batch_hc_mix/split */
    total += 2ull * pc * DS4_N_EMBD * f32;                /* batch_attn_cur/norm */
    total += 2ull * pc * dz.q_rank * f32;                 /* batch_qr/qr_norm */
    total += pc * dz.q_dim * f32;                         /* batch_q */
    total += 2ull * pc * DS4_N_HEAD_DIM * f32;            /* batch_kv_raw/kv */
    total += 2ull * pc * dz.comp_width_max * f32;         /* batch_comp_kv/sc */
    total += pc * dz.indexer_q_dim * f32;                 /* batch_indexer_q */
    total += pc * DS4_N_INDEXER_HEAD * f32;               /* batch_indexer_weights */
    total += pc * dz.q_dim * f32;                         /* batch_heads */
    total += pc * dz.low_dim * f32;                       /* batch_attn_low */
    total += pc * DS4_N_EMBD * f32;                       /* batch_attn_out */
    total += pc * dz.hc_dim * f32;                        /* batch_after_attn_hc */
    total += 2ull * pc * DS4_N_EMBD * f32;                /* batch_ffn_cur/norm */
    total += 3ull * pc * dz.shared_dim * f32;             /* batch_shared_gate/up/mid */
    total += pc * DS4_N_EMBD * f32;                       /* batch_shared_out */
    total += 2ull * pc * DS4_N_EXPERT * f32;              /* batch_router_logits/probs */
    total += pc * DS4_N_EXPERT_USED * (sizeof(int) + f32); /* batch_router_selected/weights */
    total += 3ull * pc * DS4_N_EXPERT_USED * dz.routed_mid_dim * f32; /* batch_routed_gate/up/mid */
    total += pc * DS4_N_EXPERT_USED * DS4_N_EMBD * f32;   /* batch_routed_down */
    total += pc * DS4_N_EMBD * f32;                       /* batch_routed_out */

    /* DSpark drafter graph state (gpu_graph_init_dspark_target), allocated by
     * ds4_session_create whenever the engine has a drafter loaded — the
     * production merged GGUF auto-enables it. */
    if (enable_spec) {
        total += 3ull * DS4_N_EMBD * f32;                 /* dspark_target_h[3] */
        total += 3ull * 17 * DS4_N_EMBD * f32;            /* dspark_target_h_batch[3] */
        total += 3ull * DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * f32; /* dspark_raw_cache[3] */
        total += (uint64_t)DS4_N_EMBD * f32;              /* dspark_main_x */
        total += 3ull * pc * DS4_N_EMBD * f32;            /* dspark_bulk_h[3] */
        total += 3ull * DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * f32; /* dspark_prompt_h[3] */
        const uint64_t attn_w = 2ull * DS4_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * DS4_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            total += 2ull * 17 * attn_w * f32;            /* spec_comp_kv/sc_save */
            if (ratio == 4) total += 2ull * 17 * idx_w * f32; /* spec_icomp_kv/sc_save */
        }
        total += (uint64_t)DS4_N_HEAD_DIM * f32;          /* spec_comp_scratch_row */
        total += 3ull * DS4_N_EMBD * f32;                 /* dspark_concat */
        total += (uint64_t)DS4_N_EMBD * f32;              /* dspark_proj_out */
        total += 3ull * DS4_N_HEAD_DIM * f32;             /* dspark_seed_kv/norm/rot */
        total += (uint64_t)DS4_N_VOCAB * f32;             /* dspark_markov_logits */
        total += 16ull * DS4_N_VOCAB * f32;               /* spec_logits */
    }
    return total;
}



/* DS4_KV_MANAGED: measurement override for the managed-vs-device placement of
 * the PERSISTENT KV caches (raw ring + attn/index comp caches), for the
 * multi-session UVM over-provisioning design (plan addendum 2026-07-14 —
 * first-touch fault cost must be measured before committing to
 * always-managed slot KV). Unset/empty → the size-based policy
 * (ds4_gpu_should_use_managed_kv_cache) decides, as before. "1"/"on" →
 * force cudaMallocManaged; "0"/"off"/"false" → force device cudaMalloc.
 * Read once per process (static cache) and consulted only at graph
 * allocation (session create) — never on a token/layer path. */
static int gpu_graph_kv_managed_override(void) {
    static int cached = -2;
    if (cached == -2) {
        const char *v = getenv("DS4_KV_MANAGED");
        if (!v || !v[0]) {
            cached = -1;
        } else if (strcmp(v, "0") == 0 || strcasecmp(v, "off") == 0 ||
                   strcasecmp(v, "false") == 0) {
            cached = 0;
        } else {
            cached = 1;
        }
    }
    return cached;
}



/* Allocate the GPU graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF. */
bool gpu_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_spec) {
    memset(g, 0, sizeof(*g));
    g->comp_ratio_override = -1;
    gpu_graph_dims dz;
    gpu_graph_compute_dims(&dz, weights, layer, raw_cap, ctx_size, prefill_cap);
    raw_cap = dz.raw_cap;
    ctx_size = dz.ctx_size;
    prefill_cap = dz.prefill_cap;
    g->raw_cap = dz.raw_cap;
    g->raw_window = dz.raw_window;
    g->prefill_cap = dz.prefill_cap;

    /* DS4_ATTN_PACK validation lives up here: no allocations have happened
     * yet, so these early returns need no cleanup. */
    if (getenv("DS4_ATTN_MX") != NULL) {
        /* Removed 2026-07-10: superseded by DS4_ATTN_PACK (bit-exact, smaller
         * rows; MX re-quantized the rope dims and cost drafter acceptance).
         * Refuse loudly instead of silently running a different format. */
        fprintf(stderr,
                "ds4: DS4_ATTN_MX has been removed (superseded by DS4_ATTN_PACK); "
                "unset DS4_ATTN_MX (use DS4_ATTN_PACK=0 for the classic f32 comp cache)\n");
        return false;
    }
    if (gpu_graph_attn_pack_enabled() &&
        (DS4_N_ROT != 64u || ((DS4_N_HEAD_DIM - DS4_N_ROT) % 64u) != 0u)) {
        fprintf(stderr,
                "ds4: DS4_ATTN_PACK requires n_rot 64 and 64-aligned nope dims "
                "(head_dim %u / n_rot %u)\n",
                (unsigned)DS4_N_HEAD_DIM, (unsigned)DS4_N_ROT);
        return false;
    }
    g->comp_cap = dz.comp_cap;
    if (gpu_graph_attn_pack_enabled()) {
        g->attn_comp_stage_cap = dz.attn_comp_stage_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_comp_cap[il] = dz.layer_comp_cap[il];
    }

    const uint64_t hc_dim = dz.hc_dim;
    const uint64_t mix_hc = dz.mix_hc;
    const uint64_t q_rank = dz.q_rank;
    const uint64_t q_dim = dz.q_dim;
    const uint64_t low_dim = dz.low_dim;
    const uint64_t shared_dim = dz.shared_dim;
    const uint64_t routed_mid_dim = dz.routed_mid_dim;
    const uint64_t vocab_dim = dz.vocab_dim;
    const uint64_t comp_width_max = dz.comp_width_max;
    const uint64_t indexer_q_dim = dz.indexer_q_dim;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        gpu_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const int managed_override = gpu_graph_kv_managed_override();
    const bool managed_kv_cache = managed_override >= 0
        ? managed_override != 0
        : ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_override >= 0) {
        fprintf(stderr,
                "ds4: DS4_KV_MANAGED override: KV caches use %s allocation "
                "(measurement flag; policy would have chosen %s)\n",
                managed_kv_cache ? "MANAGED (cudaMallocManaged)"
                                 : "DEVICE (cudaMalloc)",
                ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes,
                                                    context_bytes) != 0
                    ? "managed" : "device");
    }
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    g->cur_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->flat_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->hc_mix = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    g->hc_pre = ds4_gpu_tensor_view(g->hc_split, 0, (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_post = ds4_gpu_tensor_view(g->hc_split,
                                       (uint64_t)DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * sizeof(float));
    g->hc_comb = ds4_gpu_tensor_view(g->hc_split,
                                       2ull * DS4_N_HC * sizeof(float),
                                       (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    g->attn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->attn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->qr = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->qr_norm = ds4_gpu_tensor_alloc(q_rank * sizeof(float));
    g->q = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->kv_raw = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    g->kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool state_init_ok = true;
    const uint64_t raw_elem_bytes = gpu_graph_raw_f16_enabled() ? sizeof(uint16_t)
                                                                : sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_raw_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * raw_elem_bytes);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            const uint64_t comp_row_bytes = gpu_graph_attn_comp_cache_row_bytes();
            g->layer_attn_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                    managed_kv_cache,
                    (uint64_t)g->layer_comp_cap[il] * comp_row_bytes);
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            if (enable_spec) {
                g->spec_attn_state_kv[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] = ds4_gpu_tensor_alloc(attn_width * attn_rows * sizeof(float));
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                gpu_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                const uint64_t index_row_bytes = gpu_graph_idx_fp4_enabled()
                    ? DS4_ENGINE_IDXFP4_ROWBYTES
                    : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
                g->layer_index_comp_cache[il] = gpu_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * index_row_bytes);
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                if (enable_spec) {
                    g->spec_index_state_kv[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] = ds4_gpu_tensor_alloc(index_width * index_rows * sizeof(float));
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    gpu_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    g->comp_kv_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    g->comp_sc_cur = ds4_gpu_tensor_alloc(comp_width_max * sizeof(float));
    if (gpu_graph_attn_pack_enabled()) {
        /* f32 staging: the compressor writes real f32 rows here, then the commit
         * step packs them to the persistent DS4_ATTN_PACK comp cache. */
        g->attn_comp_stage = ds4_gpu_tensor_alloc((uint64_t)g->attn_comp_stage_cap *
                                                  DS4_N_HEAD_DIM * sizeof(float));
        /* f32 shadow the prefill attention consumers (and session save) read
         * after the persistent packed comp cache is dequantized (see
         * gpu_graph_attn_comp_read_cache). */
        g->attn_comp_dequant = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap *
                                                    DS4_N_HEAD_DIM * sizeof(float));
    }
    if (gpu_graph_idx_fp4_enabled()) {
        if (DS4_N_INDEXER_HEAD_DIM != 128u) {
            /* The packed loader and QAT+pack kernels hard-code the 68-byte
             * head_dim-128 row; fail loud here instead of deep in a launch. */
            fprintf(stderr,
                    "ds4: DS4_IDX_FP4 requires indexer head_dim 128 (model has %u)\n",
                    DS4_N_INDEXER_HEAD_DIM);
            gpu_graph_free(g);
            return false;
        }
        /* f32 emit/repack staging for the packed indexer cache: comp-cap rows so
         * the compressor writers can keep their absolute row indices. */
        g->idx_comp_stage = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap *
                                                 DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        ds4_gpu_indexer_set_fp4(1);
    }
    g->indexer_q = ds4_gpu_tensor_alloc(indexer_q_dim * sizeof(float));
    g->indexer_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
    /* DS4_PREFILL_SLICE: these two are the only ctx-scaling f32 work buffers
     * with a prefill_cap token dimension; under slicing they only ever hold
     * one <=slice-token span at a time. */
    const uint64_t score_rows = (gpu_graph_prefill_slice() != 0u &&
                                 (uint64_t)gpu_graph_prefill_slice() < pc)
        ? (uint64_t)gpu_graph_prefill_slice() : pc;
    g->indexer_scores = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * score_rows * sizeof(float));
    g->comp_mask = ds4_gpu_tensor_alloc((uint64_t)g->comp_cap * score_rows * sizeof(float));
    g->comp_selected = ds4_gpu_tensor_alloc((uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) *
                                              pc * sizeof(uint32_t));
    g->heads = ds4_gpu_tensor_alloc(q_dim * sizeof(float));
    g->attn_low = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    g->attn_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_attn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->ffn_cur = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->ffn_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->shared_gate = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_up = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_mid = ds4_gpu_tensor_alloc(shared_dim * sizeof(float));
    g->shared_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->router_logits = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_probs = ds4_gpu_tensor_alloc(DS4_N_EXPERT * sizeof(float));
    g->router_selected = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(int));
    g->router_weights = ds4_gpu_tensor_alloc(DS4_N_EXPERT_USED * sizeof(float));
    g->routed_gate = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_up = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_mid = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->routed_down = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->routed_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->after_ffn_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    g->output_pre = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits = ds4_gpu_tensor_alloc(vocab_dim * sizeof(float));
    g->prefill_tokens = ds4_gpu_tensor_alloc(pc * sizeof(int32_t));
    g->batch_cur_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_next_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_flat_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_hc_mix = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_hc_split = ds4_gpu_tensor_alloc(pc * mix_hc * sizeof(float));
    g->batch_attn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_attn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_qr = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_qr_norm = ds4_gpu_tensor_alloc(pc * q_rank * sizeof(float));
    g->batch_q = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_kv_raw = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_kv = ds4_gpu_tensor_alloc(pc * DS4_N_HEAD_DIM * sizeof(float));
    g->batch_comp_kv = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_comp_sc = ds4_gpu_tensor_alloc(pc * comp_width_max * sizeof(float));
    g->batch_indexer_q = ds4_gpu_tensor_alloc(pc * indexer_q_dim * sizeof(float));
    g->batch_indexer_weights = ds4_gpu_tensor_alloc(pc * DS4_N_INDEXER_HEAD * sizeof(float));
    g->batch_heads = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(float));
    g->batch_attn_low = ds4_gpu_tensor_alloc(pc * low_dim * sizeof(float));
    g->batch_attn_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_after_attn_hc = ds4_gpu_tensor_alloc(pc * hc_dim * sizeof(float));
    g->batch_ffn_cur = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_ffn_norm = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_shared_gate = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_up = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_mid = ds4_gpu_tensor_alloc(pc * shared_dim * sizeof(float));
    g->batch_shared_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));
    g->batch_router_logits = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_probs = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT * sizeof(float));
    g->batch_router_selected = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(int));
    g->batch_router_weights = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * sizeof(float));
    g->batch_routed_gate = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_up = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_mid = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
    g->batch_routed_down = ds4_gpu_tensor_alloc(pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
    g->batch_routed_out = ds4_gpu_tensor_alloc(pc * DS4_N_EMBD * sizeof(float));

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_spec ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             (!gpu_graph_idx_fp4_enabled() || g->idx_comp_stage != NULL) &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_spec ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL));
        }
    }

    const bool ok = state_init_ok && layer_cache_ok &&
                    g->cur_hc && g->flat_hc && g->hc_mix && g->hc_split &&
                    g->hc_pre && g->hc_post && g->hc_comb &&
                    g->attn_cur && g->attn_norm && g->qr && g->qr_norm &&
                    g->q && g->kv_raw && g->kv &&
                    g->comp_kv_cur && g->comp_sc_cur &&
                    (!gpu_graph_attn_pack_enabled() ||
                     (g->attn_comp_stage && g->attn_comp_dequant)) &&
                    g->indexer_q && g->indexer_weights && g->indexer_scores &&
                    g->comp_mask && g->comp_selected &&
                    g->heads && g->attn_low && g->attn_out &&
                    g->after_attn_hc && g->ffn_cur && g->ffn_norm &&
                    g->shared_gate && g->shared_up && g->shared_mid &&
                    g->shared_out &&
                    g->router_logits && g->router_probs && g->router_selected && g->router_weights &&
                    g->routed_gate && g->routed_up && g->routed_mid &&
                    g->routed_down && g->routed_out &&
                    g->after_ffn_hc &&
                    g->output_pre && g->output_weights && g->output_embd &&
                    g->output_norm && g->logits &&
                    g->prefill_tokens &&
                    g->batch_cur_hc && g->batch_next_hc && g->batch_flat_hc &&
                    g->batch_hc_mix && g->batch_hc_split &&
                    g->batch_attn_cur && g->batch_attn_norm &&
                    g->batch_qr && g->batch_qr_norm && g->batch_q &&
                    g->batch_kv_raw && g->batch_kv &&
                    g->batch_comp_kv && g->batch_comp_sc &&
                    g->batch_indexer_q && g->batch_indexer_weights &&
                    g->batch_heads && g->batch_attn_low && g->batch_attn_out &&
                    g->batch_after_attn_hc &&
                    g->batch_ffn_cur && g->batch_ffn_norm &&
                    g->batch_shared_gate && g->batch_shared_up &&
                    g->batch_shared_mid && g->batch_shared_out &&
                    g->batch_router_logits && g->batch_router_probs &&
                    g->batch_router_selected && g->batch_router_weights &&
                    g->batch_routed_gate && g->batch_routed_up &&
                    g->batch_routed_mid && g->batch_routed_down &&
                    g->batch_routed_out;
    if (!ok) gpu_graph_free(g);
    return ok;
}

bool gpu_graph_init_dspark_target(ds4_gpu_graph *g, const uint32_t target_layer_ids[3]) {
    bool ok = true;
    for (int i = 0; i < 3; i++) {
        g->dspark_target_layer_ids[i] = target_layer_ids[i];
        g->dspark_target_h[i] = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        /* Fused-loop batch capture: one anchor hidden per verify-batch position
         * (17 = the spec block clamp of 16 drafts + first_token). */
        g->dspark_target_h_batch[i] = ds4_gpu_tensor_alloc(
            (uint64_t)17 * DS4_N_EMBD * sizeof(float));
        g->dspark_raw_cache[i] = ds4_gpu_tensor_alloc(
            (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_n_raw[i] = 0;
        ok = ok && g->dspark_target_h[i] && g->dspark_target_h_batch[i] &&
             g->dspark_raw_cache[i];
    }
    g->dspark_capture_batch_n = 0;
    g->dspark_main_x = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    ok = ok && g->dspark_main_x;
    /* Bulk prefill capture buffers: always allocated when the drafter is
     * loaded (~200 MB at prefill_cap 4096) -- the prompt-window seeding
     * reduces every chunk through them; DS4_DSPARK_PREFILL_DUMP additionally
     * streams them to disk for retraining data. */
    g->dspark_bulk_n = 0;
    for (int i = 0; i < 3; i++) {
        g->dspark_bulk_h[i] = ds4_gpu_tensor_alloc(
            (uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
        ok = ok && g->dspark_bulk_h[i];
    }
    /* Prompt-window ring: last <=128 prompt positions' anchor hiddens. */
    g->dspark_prompt_n = 0;
    for (int i = 0; i < 3; i++) {
        g->dspark_prompt_h[i] = ds4_gpu_tensor_alloc(
            (uint64_t)DS4_DSPARK_DRAFT_WINDOW * DS4_N_EMBD * sizeof(float));
        ok = ok && g->dspark_prompt_h[i];
    }
    /* Stage-B no-replay rollback: per-position compressor projection saves for
     * every compressed layer (attn comp_width <= 2*DS4_N_HEAD_DIM; indexer width
     * = 2*DS4_N_INDEXER_HEAD_DIM) + one emit-sink scratch row. ~8 MB total. */
    {
        const uint64_t attn_w = 2ull * DS4_N_HEAD_DIM;
        const uint64_t idx_w = 2ull * DS4_N_INDEXER_HEAD_DIM;
        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            g->spec_comp_kv_save[il] = NULL;
            g->spec_comp_sc_save[il] = NULL;
            g->spec_icomp_kv_save[il] = NULL;
            g->spec_icomp_sc_save[il] = NULL;
            if (ratio == 0) continue;
            g->spec_comp_kv_save[il] = ds4_gpu_tensor_alloc(17ull * attn_w * sizeof(float));
            g->spec_comp_sc_save[il] = ds4_gpu_tensor_alloc(17ull * attn_w * sizeof(float));
            ok = ok && g->spec_comp_kv_save[il] && g->spec_comp_sc_save[il];
            if (ratio == 4) {
                g->spec_icomp_kv_save[il] = ds4_gpu_tensor_alloc(17ull * idx_w * sizeof(float));
                g->spec_icomp_sc_save[il] = ds4_gpu_tensor_alloc(17ull * idx_w * sizeof(float));
                ok = ok && g->spec_icomp_kv_save[il] && g->spec_icomp_sc_save[il];
            }
        }
        g->spec_comp_scratch_row = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        ok = ok && g->spec_comp_scratch_row;
        g->spec_comp_save_n = 0;
        g->dspark_concat = ds4_gpu_tensor_alloc(3ull * DS4_N_EMBD * sizeof(float));
        g->dspark_proj_out = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->dspark_seed_kv = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_norm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_seed_rot = ds4_gpu_tensor_alloc((uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->dspark_markov_logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_VOCAB * sizeof(float));
        ok = ok && g->dspark_concat && g->dspark_proj_out && g->dspark_seed_kv &&
             g->dspark_seed_norm && g->dspark_seed_rot && g->dspark_markov_logits;
    }
    /* Speculative-logits buffer for the N-token draft base logits and the
     * target verify pass. */
    if (!g->spec_logits) {
        g->spec_logits = ds4_gpu_tensor_alloc((uint64_t)16 * DS4_N_VOCAB * sizeof(float));
    }
    ok = ok && g->spec_logits;
    return ok;
}

