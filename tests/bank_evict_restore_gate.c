/* Tier-2 BANK EVICT/RESTORE gate (task #55 increment 2b, the memory-safety core).
 *
 * Proves the per-bank physical evict/restore cycle the proactive-eviction guard
 * relies on is (a) a real physical reclaim and (b) KV-BIT-IDENTICAL on return —
 * WITHOUT the box-lock risk of the full-server smoke (single session, controlled,
 * fills modest, never approaches OOM).
 *
 * Flow (mirrors the server guard's evict/restore, minus the disk I/O — the
 * snapshot lives in host RAM here; on the server it is the disk KV cache):
 *   1. prefill bank 0 to L tokens; capture its frontier; CHECKSUM its comp/index
 *      rows (raw D2H fold).
 *   2. save_snapshot(bank 0)   — the D2H KV snapshot (server: kv_cache "evict").
 *   3. repoint to bank 1       — bank 0 is now idle (free_physical refuses cur).
 *   4. free_physical(bank 0)   — DIRECT cudaFree of bank 0's split comp/index;
 *      assert cudaMemGetInfo free ROSE (physical returned), is_evicted==true.
 *   5. alloc_physical(bank 0)  — fresh cudaMallocManaged + base-table rebuild;
 *      assert the rebuilt comp_bases[0] entry == the new comp[il][0] ptr.
 *   6. repoint to bank 0; load_snapshot — H2D reload (server: kv_cache_try_load).
 *   7. CHECKSUM bank 0's comp/index again; assert == step 1 (bit-identical).
 *
 * MODEL-DEPENDENT, GPU-resident; run manually under the memory discipline (hold
 * temp/gpu.lock, drop_caches, no foreign ds4 process). NOT part of `make test`.
 *
 * usage: DS4_MSEQ_BANKS=2 ./tests/bank_evict_restore_gate MODEL [L]
 */
#include "ds4.h"
#include "ds4_engine_internal.h"
#include "ds4_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

static const double GIB = 1024.0 * 1024.0 * 1024.0;
static int g_fail;
#define CHECK(cond, ...) do { if(!(cond)){ fprintf(stderr,"EVICT-RESTORE FAIL: " __VA_ARGS__); fprintf(stderr,"\n"); g_fail=1; } } while(0)

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb"); if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    fclose(fp); buf[n] = '\0'; if (len_out) *len_out = (size_t)n; return buf;
}

/* FNV-1a fold of bank `bank`'s captured comp+index frontier rows (raw D2H). 0 on
 * a read failure (e.g. an evicted bank) — the caller only checksums resident banks. */
static uint64_t checksum_bank_kv(ds4_session *s, uint32_t bank) {
    ds4_gpu_graph *g = &s->graph;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    uint64_t h = 1469598103934665603ull;
    uint8_t *buf = malloc(64u * 1024u * 1024u);   /* per-layer row block scratch */
    if (!buf) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t ncomp = g->ms_n_comp[bank][il];
        if (ncomp) {
            ds4_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, bank);
            if (!v || ds4_gpu_tensor_read(v, 0, buf, (uint64_t)ncomp * attn_row) == 0) { ds4_gpu_tensor_free(v); free(buf); return 0; }
            ds4_gpu_tensor_free(v);
            for (uint64_t i = 0; i < (uint64_t)ncomp * attn_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
        }
        if (ratio == 4) {
            const uint32_t nidx = g->ms_n_index_comp[bank][il];
            if (nidx) {
                ds4_gpu_tensor *v = gpu_graph_bank_index_comp_view(g, il, bank);
                if (!v || ds4_gpu_tensor_read(v, 0, buf, (uint64_t)nidx * idx_row) == 0) { ds4_gpu_tensor_free(v); free(buf); return 0; }
                ds4_gpu_tensor_free(v);
                for (uint64_t i = 0; i < (uint64_t)nidx * idx_row; i++) { h ^= buf[i]; h *= 1099511628211ull; }
            }
        }
    }
    free(buf);
    return h;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL [L]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 24576;
    const int ctx = L + 4096;

    ds4_engine *e = NULL;
    ds4_engine_options opt; memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1]; opt.backend = DS4_BACKEND_CUDA;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "engine open failed\n"); return 1; }

    size_t tl = 0; char *text = read_file("tests/long_context_story_prompt.txt", &tl);
    if (!text) { fprintf(stderr, "prompt read failed\n"); return 1; }
    ds4_tokens base; memset(&base, 0, sizeof(base));
    ds4_tokenize_text(e, text, &base); free(text);
    if (base.len < 256) { fprintf(stderr, "prompt too short\n"); return 1; }

    ds4_session *s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0) { fprintf(stderr, "session create failed\n"); return 1; }
    const uint32_t pool = gpu_graph_bank_pool_count(&s->graph);
    fprintf(stderr, "evict_restore_gate: pool banks=%u ctx=%d L=%d\n", pool, ctx, L);
    if (pool < 2) { fprintf(stderr, "need DS4_MSEQ_BANKS>=2\n"); return 1; }

    /* 1. Prefill bank 0 (cur=0), capture its frontier, checksum its KV. */
    int *toks = malloc((size_t)L * sizeof(int));
    for (int i = 0; i < L; i++) toks[i] = base.v[i % base.len];
    ds4_tokens p; memset(&p, 0, sizeof(p)); p.v = toks; p.len = p.cap = L;
    char err[256];
    if (ds4_session_sync(s, &p, err, sizeof(err)) != 0) { fprintf(stderr, "sync failed: %s\n", err); return 1; }
    gpu_graph_bank_counters_capture(&s->graph, 0);
    (void)ds4_gpu_synchronize();
    const uint64_t sum_before = checksum_bank_kv(s, 0);
    const uint64_t touched0 = ds4_session_bank_touched_kv_bytes(s, 0);
    CHECK(sum_before != 0, "checksum_before failed");
    fprintf(stderr, "evict_restore_gate: bank0 prefilled pos=%d touched=%.3f GiB checksum=%016" PRIx64 "\n",
            ds4_session_pos(s), (double)touched0 / GIB, sum_before);

    /* 2. RAW snapshot of bank 0's comp/index frontier rows to host buffers
     * (guaranteed bit-identical by construction — isolates the free/alloc+table
     * primitives from the payload machinery; the server uses the disk KV cache). */
    ds4_gpu_graph *g = &s->graph;
    const uint64_t attn_row = gpu_graph_attn_comp_cache_row_bytes();
    const uint64_t idx_row = gpu_graph_idx_fp4_enabled()
        ? DS4_ENGINE_IDXFP4_ROWBYTES : (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    uint8_t *comp_snap[DS4_MAX_LAYER] = {0}, *idx_snap[DS4_MAX_LAYER] = {0};
    uint64_t comp_sz[DS4_MAX_LAYER] = {0}, idx_sz[DS4_MAX_LAYER] = {0};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        comp_sz[il] = (uint64_t)g->ms_n_comp[0][il] * attn_row;
        if (comp_sz[il]) { comp_snap[il] = malloc(comp_sz[il]);
            ds4_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, 0);
            CHECK(v && ds4_gpu_tensor_read(v, 0, comp_snap[il], comp_sz[il]) != 0, "comp D2H il=%u", il);
            ds4_gpu_tensor_free(v); }
        if (ratio == 4) { idx_sz[il] = (uint64_t)g->ms_n_index_comp[0][il] * idx_row;
            if (idx_sz[il]) { idx_snap[il] = malloc(idx_sz[il]);
                ds4_gpu_tensor *v = gpu_graph_bank_index_comp_view(g, il, 0);
                CHECK(v && ds4_gpu_tensor_read(v, 0, idx_snap[il], idx_sz[il]) != 0, "index D2H il=%u", il);
                ds4_gpu_tensor_free(v); } }
    }

    uint64_t f_prefill = 0, tt = 0; (void)ds4_gpu_synchronize(); ds4_gpu_mem_info(&f_prefill, &tt);

    /* 3. Repoint to bank 1 so bank 0 is idle (free_physical refuses cur). */
    CHECK(ds4_session_bank_state_restore(s, 1), "repoint to bank 1 failed");
    uint64_t f_repoint = 0; (void)ds4_gpu_synchronize(); ds4_gpu_mem_info(&f_repoint, &tt);

    /* 4. free_physical(bank 0) — DIRECT cudaFree; physical must return. */
    CHECK(ds4_session_bank_free_physical(s, 0), "free_physical failed");
    (void)ds4_gpu_synchronize();
    uint64_t f_free = 0; ds4_gpu_mem_info(&f_free, &tt);
    const int64_t reclaimed = (int64_t)f_free - (int64_t)f_repoint;
    CHECK(ds4_session_bank_is_evicted(s, 0), "bank 0 not marked evicted after free");
    fprintf(stderr, "evict_restore_gate: free[GiB] prefill=%.3f repoint=%.3f afterfree=%.3f reclaimed=%.3f (touched=%.3f)\n",
            (double)f_prefill/GIB, (double)f_repoint/GIB, (double)f_free/GIB, (double)reclaimed/GIB, (double)touched0/GIB);
    CHECK(reclaimed > 0 && (int64_t)touched0 - reclaimed <= (int64_t)(256ull<<20) &&
          reclaimed - (int64_t)touched0 <= (int64_t)(256ull<<20),
          "reclaimed=%.3f GiB not ~= touched=%.3f GiB", (double)reclaimed/GIB, (double)touched0/GIB);

    /* 5. alloc_physical(bank 0) — fresh slabs + base-table rebuild. */
    CHECK(ds4_session_bank_alloc_physical(s, 0), "alloc_physical failed");
    CHECK(!ds4_session_bank_is_evicted(s, 0), "bank 0 still evicted after alloc");
    { int checked = 0;
        for (uint32_t il = 0; il < DS4_N_LAYER && checked < 4; il++) {
            if (ds4_layer_compress_ratio(il) == 0) continue;
            void *want = ds4_gpu_tensor_device_ptr(g->banks.comp[il][0]);
            void *got = NULL; (void)ds4_gpu_synchronize();
            ds4_gpu_tensor_read(g->banks.comp_bases[il], 0, &got, sizeof(void *));
            CHECK(got == want, "comp_bases[%u][0] rebuild mismatch (got %p want %p)", il, got, want);
            checked++; } }

    /* 6. H2D reload the raw bytes into the fresh bank-0 comp/index. */
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ds4_layer_compress_ratio(il) == 0) continue;
        if (comp_sz[il]) { ds4_gpu_tensor *v = gpu_graph_bank_attn_comp_view(g, il, 0);
            CHECK(v && ds4_gpu_tensor_write(v, 0, comp_snap[il], comp_sz[il]) != 0, "comp H2D il=%u", il);
            ds4_gpu_tensor_free(v); }
        if (idx_sz[il]) { ds4_gpu_tensor *v = gpu_graph_bank_index_comp_view(g, il, 0);
            CHECK(v && ds4_gpu_tensor_write(v, 0, idx_snap[il], idx_sz[il]) != 0, "index H2D il=%u", il);
            ds4_gpu_tensor_free(v); }
    }
    /* Reinstall bank 0's frontier counters + host carry (server: bank_state_restore). */
    CHECK(ds4_session_bank_state_restore(s, 0), "repoint back to bank 0 failed");
    (void)ds4_gpu_synchronize();

    /* 7. checksum again — must be bit-identical to before evict. */
    const uint64_t sum_after = checksum_bank_kv(s, 0);
    CHECK(sum_after == sum_before, "KV NOT bit-identical after evict/restore (%016" PRIx64 " vs %016" PRIx64 ")",
          sum_after, sum_before);
    CHECK(ds4_session_pos(s) == L, "restored pos %d != %d", ds4_session_pos(s), L);
    fprintf(stderr, "evict_restore_gate: restored pos=%d checksum=%016" PRIx64 " (bit-identical: %s)\n",
            ds4_session_pos(s), sum_after, sum_after == sum_before ? "YES" : "NO");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) { free(comp_snap[il]); free(idx_snap[il]); }
    free(toks);
    fprintf(stderr, "EVICT-RESTORE GATE: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
