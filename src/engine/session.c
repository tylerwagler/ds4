#include "ds4_engine_internal.h"


static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}



static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}



static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}



static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}



static DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}



static DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}



static DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}



static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > DS4_SESSION_IO_CHUNK ? DS4_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}



static DS4_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}



static DS4_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}



/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical Metal tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}



/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    const uint64_t comp_row = DS4_GPU_ATTN_COMP_CACHE_FP8
        ? DS4_FP8_KV_ROWBYTES(DS4_N_HEAD_DIM)
        : (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * comp_row;
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}



/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}



static DS4_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const ds4_gpu_tensor *tensor,
                                                                 uint64_t offset_f16, uint64_t count,
                                                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (ds4_gpu_tensor_read(tensor, offset_f16 + done * sizeof(uint16_t),
                                h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read Metal F16 session tensor");
            return 1;
        }
        for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
        if (payload_write_bytes(fp, f, (uint64_t)n * sizeof(float), err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}



static DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, ds4_gpu_tensor *tensor,
                                                                uint64_t offset_f16, uint64_t count,
                                                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                                                char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (payload_read_bytes(fp, f, (uint64_t)n * sizeof(float), remaining, err, errlen) != 0) return 1;
        for (size_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        if (ds4_gpu_tensor_write(tensor, offset_f16 + done * sizeof(uint16_t),
                                 h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore Metal F16 session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}



static bool ds4_session_is_cpu(const ds4_session *s) {
    return s && s->engine && s->engine->backend == DS4_BACKEND_CPU;
}



static uint32_t session_cpu_raw_live_rows(const ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    uint32_t rows = ds4_default_raw_cap((uint32_t)s->ctx_size);
    if (rows > (uint32_t)s->checkpoint.len) rows = (uint32_t)s->checkpoint.len;
    return rows;
}



static uint32_t session_cpu_comp_cap(const ds4_session *s) {
    if (!s) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio == 4) return layer->comp_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio != 0) return layer->comp_cap;
    }
    return (uint32_t)s->ctx_size;
}



static uint64_t session_cpu_payload_live_tensor_bytes(const ds4_session *s) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_cpu_raw_live_rows(s);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;
        bytes += (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}



static void session_cpu_reset_cache(ds4_session *s) {
    kv_cache_free(&s->cpu_cache);
    kv_cache_init(&s->cpu_cache, (uint32_t)s->ctx_size, 0);
}



static bool ds4_layer_payload_range_valid(uint32_t layer_start, uint32_t layer_end) {
    return layer_start <= layer_end && layer_end < (uint32_t)DS4_N_LAYER;
}



uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end) {
    if (!s || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end))
        return 0;
    if (ds4_session_is_cpu(s)) return 0;
    const ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint64_t bytes = (uint64_t)DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    const uint32_t n_layers = layer_end - layer_start + 1u;
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}



int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload save");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
    if (gpu_graph_attn_mx_enabled()) {
        /* The compressed-KV cache is MXFP8 under DS4_ATTN_MX; its on-disk
         * serialization is not implemented yet. Deferred (see Stage 4). */
        payload_set_err(err, errlen, "session save unsupported with DS4_ATTN_MX (MXFP8 KV)");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before layer snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint32_t header[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_LAYER_PAYLOAD_MAGIC,
        DS4_SESSION_LAYER_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        layer_start,
        layer_end,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = layer_start; rc == 0 && il <= layer_end; il++) {
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_write_tensor_span_f16_as_f32(fp,
                                                      g->layer_attn_comp_cache[il],
                                                      0,
                                                      (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM,
                                                      buf,
                                                      DS4_SESSION_IO_CHUNK,
                                                      err,
                                                      errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
}



int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !tokens ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload load");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
    if (gpu_graph_attn_mx_enabled()) {
        payload_set_err(err, errlen, "session load unsupported with DS4_ATTN_MX (MXFP8 KV)");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session layer payload version");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_layer_start = h[11];
    const uint32_t saved_layer_end = h[12];
    const uint32_t saved_raw_live = h[13];
    (void)saved_prefill_cap;
    if (saved_layer_start != layer_start || saved_layer_end != layer_end) {
        payload_set_err(err, errlen, "KV shard layer range does not match requested worker");
        return 1;
    }
    if (saved_ctx > (uint32_t)s->ctx_size ||
        saved_tokens != n_tokens ||
        saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV shard does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM) {
        payload_set_err(err, errlen, "KV shard was written for a different DS4 layout");
        return 1;
    }
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV shard graph chunk layout does not match current runtime");
        return 1;
    }
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap) {
        payload_set_err(err, errlen, "KV shard raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV shard compressed cache is larger than current context");
        return 1;
    }

    const uint32_t n_layers = layer_end - layer_start + 1u;
    uint32_t *n_comp = xcalloc(n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = xcalloc(n_layers, sizeof(n_index_comp[0]));
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_comp[i] > saved_comp_cap || n_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_index_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_index_comp[i] > saved_comp_cap || n_index_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        free(n_comp);
        free(n_index_comp);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV shard restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_read_tensor_span_f32_as_f16(fp,
                                                     g->layer_attn_comp_cache[il],
                                                     0,
                                                     (uint64_t)n_comp[i] * DS4_N_HEAD_DIM,
                                                     buf,
                                                     DS4_SESSION_IO_CHUNK,
                                                     &remaining,
                                                     err,
                                                     errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[i] * DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[i] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc == 0 && remaining != 0) {
        payload_set_err(err, errlen, "KV shard has trailing payload bytes");
        rc = 1;
    }
    if (rc == 0 && ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV shard restore");
        rc = 1;
    }
    if (rc == 0) {
        token_vec_free(&s->checkpoint);
        memset(&s->checkpoint, 0, sizeof(s->checkpoint));
        for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
        for (uint32_t i = 0; i < n_layers; i++) {
            const uint32_t il = layer_start + i;
            g->layer_n_comp[il] = n_comp[i];
            g->layer_n_index_comp[il] = n_index_comp[i];
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        g->mtp_n_raw = 0;
    }
    free(n_comp);
    free(n_index_comp);
    return rc;
}



int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_tensor *gate = e->weights.layer[il].ffn_gate_exps;
        if (!gate) continue;
        return 2;
    }
    return 0;
}



bool ds4_engine_has_output_head(ds4_engine *e) {
    return e && weights_have_output_head(&e->weights);
}



bool ds4_engine_has_mtp(ds4_engine *e) {
    return e && e->backend != DS4_BACKEND_CPU &&
           e->distributed.role == DS4_DISTRIBUTED_NONE &&
           e->mtp_ready;
}



int ds4_engine_mtp_draft_tokens(ds4_engine *e) {
    return ds4_engine_has_mtp(e) ? e->mtp_draft_tokens : 0;
}

bool ds4_engine_has_dspark(ds4_engine *e) {
    return e && e->backend != DS4_BACKEND_CPU &&
           e->distributed.role == DS4_DISTRIBUTED_NONE &&
           e->dspark_ready;
}

int ds4_engine_dspark_draft_tokens(ds4_engine *e) {
    return ds4_engine_has_dspark(e) ? e->dspark_draft_tokens : 0;
}



const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}



static void spec_frontier_free(ds4_spec_frontier *f) {
    if (!f) return;
    memset(f, 0, sizeof(*f));
}



static bool spec_frontier_snapshot(ds4_spec_frontier *f, ds4_session *s) {
    memset(f, 0, sizeof(*f));
    ds4_gpu_graph *g = &s->graph;
    f->mtp_n_raw = g->mtp_n_raw;

    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        f->n_comp[il] = g->layer_n_comp[il];
        f->n_index_comp[il] = g->layer_n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->spec_attn_state_kv[il], 0,
                                   g->layer_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->spec_attn_state_score[il], 0,
                                   g->layer_attn_state_score[il], 0, ab) != 0;
        if (ratio == 4) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ok &&
                 ds4_gpu_tensor_copy(g->spec_index_state_kv[il], 0,
                                       g->layer_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->spec_index_state_score[il], 0,
                                       g->layer_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) return true;

    spec_frontier_free(f);
    return false;
}



static bool spec_frontier_restore(ds4_spec_frontier *f, ds4_session *s) {
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_begin_commands() != 0;
    g->mtp_n_raw = f->mtp_n_raw;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = f->n_comp[il];
        g->layer_n_index_comp[il] = f->n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_attn_state_score[il], 0, ab) != 0;
        if (ok && ratio == 4) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
}



/* Commit the prefix-1 state captured by the N=2 speculative verifier.
 *
 * The verifier has already advanced every layer through both draft tokens.  On
 * a one-token accept the append-only compressed caches can keep the second
 * speculative row as invisible garbage, but the compressor frontiers and row
 * counters must be rewound to the exact state after draft[0].  This is the
 * cheap partial-accept path: copy a few small per-layer frontiers instead of
 * restoring the whole prefix and replaying a one-token target decode. */
static bool spec_frontier_commit_prefix1(ds4_session *s) {
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;

        g->layer_n_comp[il] = g->spec_prefix1_n_comp[il];
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_prefix1_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_prefix1_attn_state_score[il], 0, ab) != 0;
        if (ok && ratio == 4) {
            g->layer_n_index_comp[il] = g->spec_prefix1_n_index_comp[il];
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_prefix1_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_prefix1_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
}



uint64_t ds4_session_payload_bytes(ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    if (s->distributed) return 0;
    if (ds4_session_is_cpu(s)) {
        uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += session_cpu_payload_live_tensor_bytes(s);
        return bytes;
    }
    const ds4_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
}



int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}



void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}



int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = ds4_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}



int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_save_payload(s->distributed, s, fp, err, errlen);
    }
    if (!ds4_session_is_cpu(s) && gpu_graph_attn_mx_enabled()) {
        /* GPU path serializes the MXFP8 comp cache raw; not implemented. Deferred. */
        payload_set_err(err, errlen, "session save unsupported with DS4_ATTN_MX (MXFP8 KV)");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t raw_live = session_cpu_raw_live_rows(s);
        const uint32_t raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t comp_cap = session_cpu_comp_cap(s);
        uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_PAYLOAD_MAGIC,
            DS4_SESSION_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            s->prefill_cap,
            raw_cap,
            raw_cap,
            comp_cap,
            (uint32_t)s->checkpoint.len,
            DS4_N_LAYER,
            DS4_N_HEAD_DIM,
            DS4_N_INDEXER_HEAD_DIM,
            DS4_N_VOCAB,
            raw_live,
        };
        for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
            if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
        }
        for (int i = 0; i < s->checkpoint.len; i++) {
            if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
        }
        if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_index_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (raw_live > layer->n_raw) {
                payload_set_err(err, errlen, "CPU session raw cache has fewer live rows than checkpoint");
                return 1;
            }
            const uint32_t raw_start = layer->n_raw - raw_live;
            if (payload_write_bytes(fp,
                                    layer->raw_kv + (uint64_t)raw_start * DS4_N_HEAD_DIM,
                                    (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            if (payload_write_bytes(fp,
                                    layer->attn_comp_kv,
                                    (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (ratio == 4) {
                if (payload_write_bytes(fp,
                                        layer->index_comp_kv,
                                        (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                        err,
                                        errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
            }
        }
        return 0;
    }
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        DS4_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                           (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * DS4_FP8_KV_ROWBYTES(DS4_N_HEAD_DIM),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        } else if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_write_tensor_span_f16_as_f32(fp,
                                                      g->layer_attn_comp_cache[il],
                                                      0,
                                                      (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM,
                                                      buf,
                                                      DS4_SESSION_IO_CHUNK,
                                                      err,
                                                      errlen);
        } else {
            rc = payload_write_tensor_span(fp,
                                           g->layer_attn_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
}



int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_load_payload(s->distributed, s, fp, payload_bytes, err, errlen);
    }
    if (!ds4_session_is_cpu(s) && gpu_graph_attn_mx_enabled()) {
        payload_set_err(err, errlen, "session load unsupported with DS4_ATTN_MX (MXFP8 KV)");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t saved_ctx = h[2];
        const uint32_t saved_prefill_cap = h[3];
        const uint32_t saved_raw_cap = h[4];
        const uint32_t saved_raw_window = h[5];
        const uint32_t saved_comp_cap = h[6];
        const uint32_t saved_tokens = h[7];
        const uint32_t saved_raw_live = h[12];
        const uint32_t cpu_raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t cpu_comp_cap = session_cpu_comp_cap(s);
        if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
            payload_set_err(err, errlen, "KV checkpoint does not fit current context");
            return 1;
        }
        if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
            h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
        {
            payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
            return 1;
        }
        /* prefill_cap is scratch scheduling capacity, not durable KV layout.
         * Old checkpoints remain valid as long as the raw KV window matches. */
        (void)saved_prefill_cap;
        if (saved_raw_window != cpu_raw_cap) {
            payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
            return 1;
        }
        const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
        if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
            saved_raw_live > saved_raw_cap || saved_raw_live > cpu_raw_cap)
        {
            payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
            return 1;
        }
        if (saved_comp_cap > cpu_comp_cap) {
            payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
            return 1;
        }

        token_vec new_checkpoint = {0};
        for (uint32_t i = 0; i < saved_tokens; i++) {
            uint32_t tok = 0;
            if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            token_vec_push(&new_checkpoint, (int)tok);
        }
        if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining, err, errlen) != 0)
        {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        uint32_t n_comp[DS4_MAX_LAYER];
        uint32_t n_index_comp[DS4_MAX_LAYER];
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_comp[il] > saved_comp_cap || n_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
                return 1;
            }
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
                return 1;
            }
        }

        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        session_cpu_reset_cache(s);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (payload_read_bytes(fp,
                                   layer->raw_kv,
                                   (uint64_t)saved_raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            layer->n_raw = saved_raw_live;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            layer->n_comp = n_comp[il];
            layer->n_index_comp = n_index_comp[il];
            if (payload_read_bytes(fp,
                                   layer->attn_comp_kv,
                                   (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (ratio == 4) {
                if (payload_read_bytes(fp,
                                       layer->index_comp_kv,
                                       (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                       &remaining,
                                       err,
                                       errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0)
                {
                    token_vec_free(&new_checkpoint);
                    return 1;
                }
            }
        }
        if (remaining != 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
            return 1;
        }
        token_vec_free(&s->checkpoint);
        s->checkpoint = new_checkpoint;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        return 0;
    }
    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    g->mtp_n_raw = 0;

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_N_HEAD_DIM * sizeof(float),
                                          (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[il] * DS4_FP8_KV_ROWBYTES(DS4_N_HEAD_DIM),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        } else if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            rc = payload_read_tensor_span_f32_as_f16(fp,
                                                      g->layer_attn_comp_cache[il],
                                                      0,
                                                      (uint64_t)n_comp[il] * DS4_N_HEAD_DIM,
                                                      buf,
                                                      DS4_SESSION_IO_CHUNK,
                                                      &remaining,
                                                      err,
                                                      errlen);
        } else {
            rc = payload_read_tensor_span(fp,
                                          g->layer_attn_comp_cache[il],
                                          0,
                                          (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->cont_anchor_valid = false;
    g->mtp_n_raw = 0;
    return 0;
}



int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}



int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}



void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}



void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}



int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    ds4_model model;
    ds4_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}



static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "ds4: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "ds4: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}



static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}



int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
    if (!e || !dataset_path || !output_path) return 1;
    if (e->backend != DS4_BACKEND_CUDA || !e->gpu_ready) {
        fprintf(stderr, "ds4: imatrix collection currently requires --cuda\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const uint32_t prefill_cap =
        gpu_graph_prefill_cap_for_prompt(ctx_size, e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);

    ds4_gpu_graph g;
    bool ok = gpu_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate imatrix GPU graph runtime\n");
        free(dataset);
        return 1;
    }
    g.quality = e->quality;
    g.ssd_streaming = e->ssd_streaming;
    g.ssd_streaming_cold = e->ssd_streaming_cold;
    g.streaming_preload_experts = e->ssd_streaming_preload_experts;
    g.power_percent = (uint32_t)e->power_percent;

    ds4_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "ds4: failed to allocate imatrix collector\n");
        gpu_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "ds4: collecting routed-MoE imatrix from %s (model=%s, layers=%u, experts=%u, ctx=%d, chunk=%u)\n",
            dataset_path, DS4_MODEL_SHAPE_NAME, DS4_N_LAYER, DS4_N_EXPERT, ctx_size, prefill_cap);

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = strstr(cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = strstr(start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            ds4_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                if (!gpu_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "ds4: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = gpu_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           NULL, NULL,
                                                           &collector,
                                                           NULL, NULL, NULL);
                } else {
                    ok = gpu_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, 0,
                                                         (uint32_t)prompt.len,
                                                         NULL, false,
                                                         &collector,
                                                         NULL, NULL);
                }
                if (!ok) {
                    fprintf(stderr, "ds4: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "ds4: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "ds4: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    gpu_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
}



int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (ds4_backend_uses_graph(e->backend)) {
        if (!e->gpu_ready) {
            fprintf(stderr, "ds4: %s generation requested but the graph backend is unavailable\n",
                    ds4_backend_name(e->backend));
            return 1;
        }
        return generate_gpu_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality,
                                            e->ssd_streaming,
                                            e->ssd_streaming_cold,
                                            e->ssd_streaming_preload_experts,
                                            e->power_percent,
                                            e->prefill_chunk,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
    }

    return generate_raw_swa_cpu(model, vocab, weights, prompt, n_predict,
                                ctx_size,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale,
                                emit, done, emit_ud, progress, progress_ud);
}



int ds4_engine_gpu_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!e->gpu_ready) {
        fprintf(stderr, "ds4: %s graph test requested but backend is unavailable\n",
                ds4_backend_name(e->backend));
        return 1;
    }
    return gpu_graph_decode_test(&e->model, &e->weights, prompt, e->quality);
}



int ds4_engine_gpu_graph_full_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!e->gpu_ready) {
        fprintf(stderr, "ds4: %s full graph test requested but backend is unavailable\n",
                ds4_backend_name(e->backend));
        return 1;
    }
    return gpu_graph_first_token_full_test(&e->model, &e->weights, prompt, e->quality);
}



int ds4_engine_gpu_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size) {
    if (!e->gpu_ready) {
        fprintf(stderr, "ds4: %s prompt graph test requested but backend is unavailable\n",
                ds4_backend_name(e->backend));
        return 1;
    }
    return gpu_graph_prompt_logits_test(&e->model, &e->weights, prompt, ctx_size);
}



int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: head test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;
    const ds4_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = xmalloc((size_t)prompt->len * DS4_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, DS4_N_EMBD, prompt_embd);

    const uint32_t n_hc = DS4_N_HC;
    float *hc0 = xmalloc((size_t)DS4_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * DS4_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, DS4_N_EMBD);

    float *attn_norm0 = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    float *q0 = xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, DS4_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv0, DS4_N_HEAD_DIM);

    float *attn_heads = xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, DS4_N_EMBD);

    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, DS4_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *after_ffn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1],
                  NULL, 0.0f, true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}



int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: first-token test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    float *hc = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(hc[0]));
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    forward_first_token_cpu(hc, model, weights, prompt->v[0]);
    print_vec_stats("first-token final_hc", hc, (uint64_t)DS4_N_HC * DS4_N_EMBD);
    output_logits_one(logits, model, weights, hc);
    print_vec_stats("first-token logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after first-token whole-model CPU pass:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(hc);
    return 0;
}



static bool ds4_engine_configure_streaming_auto_cache(ds4_engine *e) {
    if (!e ||
        !e->ssd_streaming ||
        !ds4_backend_supports_ssd_streaming(e->backend) ||
        e->ssd_streaming_cache_experts != 0 ||
        e->ssd_streaming_cache_bytes != 0) {
        return true;
    }
    if (!ds4_backend_supports_streaming_auto_cache(e->backend)) {
        return true;
    }

    const uint64_t recommended = ds4_gpu_recommended_working_set_size();
    if (recommended == 0) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache: recommended working set unavailable; "
                "set --ssd-streaming-cache-experts N or NGB explicitly\n");
        return false;
    }

    uint64_t non_routed_bytes = 0;
    if (!weights_streaming_non_routed_bytes(&e->weights, &non_routed_bytes)) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache could not measure non-routed model weights\n");
        return false;
    }

    uint64_t per_expert_bytes = 0;
    if (!ds4_streaming_routed_expert_bytes(&e->weights, &per_expert_bytes)) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache could not measure routed expert size\n");
        return false;
    }

    const uint64_t max_model_experts = (uint64_t)DS4_N_LAYER * (uint64_t)DS4_N_EXPERT;
    ds4_ssd_cache_plan plan;
    if (!ds4_ssd_auto_cache_plan(recommended,
                                 non_routed_bytes,
                                 per_expert_bytes,
                                 max_model_experts,
                                 &plan)) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache could not compute a valid cache budget\n");
        return false;
    }

    e->ssd_streaming_cache_experts = plan.cache_experts;
    fprintf(stderr,
            "ds4: SSD streaming auto cache budget\n");
    fprintf(stderr,
            "ds4:   %s recommends %.2f GiB working set\n",
            ds4_backend_name(e->backend),
            (double)recommended / 1073741824.0);
    fprintf(stderr,
            "ds4:   using 80%% total for model + cached experts: %.2f GiB\n",
            (double)plan.model_target_bytes / 1073741824.0);
    fprintf(stderr,
            "ds4:   non-routed weights: %.2f GiB\n",
            (double)non_routed_bytes / 1073741824.0);
    fprintf(stderr,
            "ds4:   routed expert size: %.2f MiB\n",
            (double)per_expert_bytes / 1048576.0);
    fprintf(stderr,
            "ds4:   cached expert count: %u (%.2f GiB)\n",
            e->ssd_streaming_cache_experts,
            (double)plan.effective_cache_bytes / 1073741824.0);
    if (plan.model_target_bytes <= non_routed_bytes) {
        fprintf(stderr,
                "ds4:   note: non-routed weights already fill the 80%% target; keeping a one-expert cache\n");
    }
    return true;
}



int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->dspark_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->ssd_streaming = opt->ssd_streaming;
    e->ssd_streaming_cold = opt->ssd_streaming_cold;
    e->distributed = opt->distributed;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    e->prefill_chunk = opt->prefill_chunk;
    e->ssd_streaming_cache_experts = opt->ssd_streaming_cache_experts;
    e->ssd_streaming_cache_bytes = opt->ssd_streaming_cache_bytes;
    e->ssd_streaming_preload_experts = opt->ssd_streaming_preload_experts;
    if (e->power_percent > 100) e->power_percent = 100;
    e->mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    if (e->mtp_draft_tokens > 16) e->mtp_draft_tokens = 16;
    e->mtp_margin = opt->mtp_margin >= 0.0f ? opt->mtp_margin : 3.0f;
    e->dspark_draft_tokens = opt->dspark_draft_tokens > 0 ? opt->dspark_draft_tokens : 5;
    if (e->dspark_draft_tokens > 16) e->dspark_draft_tokens = 16;
    e->dspark_confidence = opt->dspark_confidence > 0.0f ? opt->dspark_confidence : 0.5f;
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = ds4_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    ds4_acquire_instance_lock();

    if (opt->simulate_used_memory_bytes != 0 &&
        !ds4_ssd_memory_lock_acquire(&e->simulated_memory,
                                     opt->simulate_used_memory_bytes)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }

    bool load_slice = opt->load_slice;
    uint32_t load_layer_start = opt->load_layer_start;
    uint32_t load_layer_end = opt->load_layer_end;
    bool load_output = opt->load_output;
    bool load_output_optional = false;
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE &&
        opt->distributed.layers.set)
    {
        load_slice = true;
        load_layer_start = opt->distributed.layers.start;
        load_layer_end = opt->distributed.layers.has_output ?
                         UINT32_MAX : opt->distributed.layers.end;
        load_output = opt->distributed.layers.has_output;
        load_output_optional = opt->distributed.role == DS4_DISTRIBUTED_COORDINATOR;
    }

    const bool graph_backend = ds4_backend_uses_graph(opt->backend);
    if (graph_backend) ds4_linux_graph_backend_set_oom_score(opt->backend);
    model_open(&e->model, opt->model_path, graph_backend, !opt->inspect_only);
    if (opt->warm_weights) model_warm_weights(&e->model);
    if (!opt->inspect_only) vocab_load(&e->vocab, &e->model);
    config_validate_model(&e->model);
    if (e->ssd_streaming && !ds4_backend_supports_ssd_streaming(e->backend)) {
        fprintf(stderr, "ds4: --ssd-streaming is currently supported only with --cuda\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (opt->expert_overlay && opt->expert_overlay[0]) {
        if (e->ssd_streaming) {
            fprintf(stderr, "ds4: --expert-overlay is not compatible with --ssd-streaming\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        const char *sep = strrchr(opt->expert_overlay, ':');
        if (!sep || sep == opt->expert_overlay || !sep[1]) {
            fprintf(stderr, "ds4: --expert-overlay expects FILE:PREFIX (e.g. donor.gguf:blk.17.)\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        char overlay_path[4096];
        const size_t path_len = (size_t)(sep - opt->expert_overlay);
        if (path_len >= sizeof(overlay_path)) {
            fprintf(stderr, "ds4: --expert-overlay path is too long\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        memcpy(overlay_path, opt->expert_overlay, path_len);
        overlay_path[path_len] = '\0';
        model_open(&e->overlay_model, overlay_path, graph_backend, false);
        e->overlay_ready = true;
        /* PREFIX is a comma-separated list so several layers can be swapped
         * in one run (e.g. compose "anchor + candidate" from a cheap base
         * without materializing the combined model as a file). */
        char prefixes[2048];
        const size_t plist_len = strlen(sep + 1);
        if (plist_len >= sizeof(prefixes)) {
            fprintf(stderr, "ds4: --expert-overlay prefix list is too long\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        memcpy(prefixes, sep + 1, plist_len + 1);
        uint32_t swapped = 0;
        for (char *p = strtok(prefixes, ","); p; p = strtok(NULL, ",")) {
            const uint32_t n = model_apply_expert_overlay(&e->model, &e->overlay_model, p);
            if (n == 0) {
                fprintf(stderr, "ds4: --expert-overlay prefix '%s' matched no routed-expert tensors\n",
                        p);
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            swapped += n;
        }
        fprintf(stderr, "ds4: expert overlay: %u tensors swapped in from %s (prefixes %s)\n",
                swapped, overlay_path, sep + 1);
    }
    weights_bind(&e->weights,
                 &e->model,
                 load_slice,
                 load_layer_start,
                 load_layer_end,
                 load_output,
                 load_output_optional);
    if (e->ssd_streaming && e->ssd_streaming_cache_bytes != 0) {
        const uint64_t requested_cache_bytes = e->ssd_streaming_cache_bytes;
        const uint64_t safe_cache_bytes =
            ds4_streaming_manual_cache_safe_bytes();
        if (safe_cache_bytes != 0 &&
            e->ssd_streaming_cache_bytes > safe_cache_bytes) {
            e->ssd_streaming_cache_bytes = safe_cache_bytes;
            fprintf(stderr,
                    "ds4: %s SSD streaming cache budget %.2f GiB capped to %.2f GiB "
                    "to keep expert buffers lockable\n",
                    ds4_backend_name(e->backend),
                    (double)requested_cache_bytes / 1073741824.0,
                    (double)e->ssd_streaming_cache_bytes / 1073741824.0);
        }
        uint64_t per_expert_bytes = 0;
        const uint32_t budget =
            ds4_streaming_cache_experts_for_byte_budget(
                    &e->weights,
                    e->ssd_streaming_cache_bytes,
                    &per_expert_bytes);
        if (budget == 0 || per_expert_bytes == 0) {
            fprintf(stderr,
                    "ds4: --ssd-streaming-cache-experts byte budget is too small or invalid for this model\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        e->ssd_streaming_cache_experts = budget;
        fprintf(stderr,
                "ds4: %s SSD streaming cache budget %.2f GiB / %.2f MiB per expert = %u experts\n",
                ds4_backend_name(e->backend),
                (double)e->ssd_streaming_cache_bytes / 1073741824.0,
                (double)per_expert_bytes / 1048576.0,
                budget);
    }
    if (opt->inspect_only) {
        *out = e;
        return 0;
    }
    if (e->backend == DS4_BACKEND_CPU && !cpu_load_directional_steering(e)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (opt->mtp_path && opt->mtp_path[0] &&
        opt->distributed.role == DS4_DISTRIBUTED_NONE) {
        if (e->ssd_streaming) {
            fprintf(stderr, "ds4: --ssd-streaming is not compatible with --mtp yet\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        model_open(&e->mtp_model, opt->mtp_path, graph_backend, true);
        mtp_weights_bind(&e->mtp_weights, &e->mtp_model);
        e->mtp_ready = true;
        fprintf(stderr, "ds4: MTP support model loaded: %s (draft=%d)\n",
                opt->mtp_path,
                e->mtp_draft_tokens);
    }

    if (opt->dspark_path && opt->dspark_path[0] &&
        opt->distributed.role == DS4_DISTRIBUTED_NONE) {
        if (e->ssd_streaming) {
            fprintf(stderr, "ds4: --ssd-streaming is not compatible with --dspark\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->mtp_ready) {
            fprintf(stderr, "ds4: --mtp and --dspark are mutually exclusive\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        model_open(&e->dspark_model, opt->dspark_path, graph_backend, true);
        dspark_weights_bind(&e->dspark_weights, &e->dspark_model);
        e->dspark_ready = true;
        fprintf(stderr, "ds4: DSpark support model loaded: %s (draft=%d, confidence=%.2f)\n",
                opt->dspark_path,
                e->dspark_draft_tokens,
                (double)e->dspark_confidence);
    }

    if (graph_backend) {
        e->gpu_ready = ds4_gpu_init() != 0;
        if (!e->gpu_ready) {
            fprintf(stderr, "ds4: %s backend unavailable; aborting startup\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_quality(e->quality);
        ds4_gpu_set_ssd_streaming(e->ssd_streaming);
        if (!ds4_engine_configure_streaming_auto_cache(e)) {
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_streaming_expert_cache_budget(e->ssd_streaming_cache_experts);
        if (e->ssd_streaming) {
            /*
             * Pin the expert cache's slab size class to the model's uniform
             * per-expert bytes, and count mixed-precision (boosted) layers:
             * those are served through mapped model views instead of the
             * cache (see weights_streaming_layer_experts_uniform).
             */
            uint64_t slab_expert_bytes = 0;
            if (ds4_streaming_routed_expert_bytes(&e->weights, &slab_expert_bytes)) {
                ds4_gpu_set_streaming_expert_cache_expert_bytes(slab_expert_bytes);
                uint32_t routed = 0, boosted = 0;
                for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                    const ds4_layer_weights *l = &e->weights.layer[il];
                    if (!l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps) continue;
                    routed++;
                    if (!weights_streaming_layer_experts_uniform(&e->weights, il)) boosted++;
                }
                if (boosted > 0) {
                    fprintf(stderr,
                            "ds4: SSD streaming mixed-precision model: %u/%u routed layers "
                            "off the slab size class will bypass the expert cache and read "
                            "experts via mapped model views\n",
                            boosted, routed);
                }
                if (boosted * 2 > routed) {
                    fprintf(stderr,
                            "ds4: WARNING: the majority of routed layers (%u/%u) are off the "
                            "slab size class (is the FIRST routed layer itself boosted?); "
                            "expert-cache hit rate will be catastrophic\n",
                            boosted, routed);
                }
            }
        }
        (void)ds4_gpu_set_model_fd(e->model.fd);
        int model_map_ok = 0;
        uint64_t *load_offsets = NULL;
        uint64_t *load_sizes = NULL;
        uint32_t load_span_count = 0;
        if (e->ssd_streaming) {
            const bool map_output = load_slice &&
                                    (load_output ||
                                     (load_output_optional &&
                                      weights_have_output_head(&e->weights)));
            ds4_model_map_span_vec spans;
            bool spans_ok = false;
            if (load_slice) {
                spans_ok = weights_model_map_decode_static_slice_spans(
                        &e->weights,
                        load_layer_start,
                        load_layer_end,
                        true,
                        map_output,
                        &spans);
            } else {
                spans_ok = weights_model_map_token_spans(&e->weights, &spans);
            }
            if (!spans_ok) {
                fprintf(stderr, "ds4: invalid SSD streaming initial token embedding map\n");
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
            uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
            uint64_t span_bytes = 0;
            for (uint32_t i = 0; i < spans.len; i++) {
                offsets[i] = spans.v[i].off;
                sizes[i] = spans.v[i].end - spans.v[i].off;
                span_bytes += sizes[i];
            }
            load_offsets = offsets;
            load_sizes = sizes;
            load_span_count = spans.len;
            if (load_slice) {
                char load_end[32];
                if (map_output && load_layer_end == UINT32_MAX) {
                    snprintf(load_end, sizeof(load_end), "output");
                } else if (map_output) {
                    snprintf(load_end, sizeof(load_end), "%u+output", load_layer_end);
                } else {
                    snprintf(load_end, sizeof(load_end), "%u", load_layer_end);
                }
                fprintf(stderr,
                        "ds4: SSD streaming initial %s model map restricted to token + non-routed layers %u:%s (%u spans, %.2f GiB tensor span)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start,
                        load_end,
                        spans.len,
                        (double)span_bytes / 1073741824.0);
            } else {
                fprintf(stderr,
                        "ds4: SSD streaming initial %s model map restricted to token embedding (%u spans, %.2f GiB tensor span)\n",
                        ds4_backend_name(e->backend),
                        spans.len,
                        (double)span_bytes / 1073741824.0);
            }
            model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                        e->model.size,
                                                        load_offsets,
                                                        load_sizes,
                                                        load_span_count,
                                                        spans.max_tensor_bytes);
            free(spans.v);
        } else if (load_slice) {
            const bool map_output = load_output ||
                                    (load_output_optional &&
                                     weights_have_output_head(&e->weights));
            char load_end[32];
            if (map_output && load_layer_end == UINT32_MAX) {
                snprintf(load_end, sizeof(load_end), "output");
            } else if (map_output) {
                snprintf(load_end, sizeof(load_end), "%u+output", load_layer_end);
            } else {
                snprintf(load_end, sizeof(load_end), "%u", load_layer_end);
            }

            ds4_model_map_span_vec spans;
            if (!weights_model_map_spans(&e->weights,
                                         load_layer_start,
                                         load_layer_end,
                                         map_output,
                                         &spans))
            {
                fprintf(stderr, "ds4: invalid model load layer slice %u:%s\n",
                        load_layer_start,
                        load_end);
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
            uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
            uint64_t span_bytes = 0;
            for (uint32_t i = 0; i < spans.len; i++) {
                offsets[i] = spans.v[i].off;
                sizes[i] = spans.v[i].end - spans.v[i].off;
                span_bytes += sizes[i];
            }
            load_offsets = offsets;
            load_sizes = sizes;
            load_span_count = spans.len;
            fprintf(stderr,
                    "ds4: restricting %s model map to layers %u:%s (%u spans, %.2f GiB tensor span)\n",
                    ds4_backend_name(e->backend),
                    load_layer_start,
                    load_end,
                    spans.len,
                    (double)span_bytes / 1073741824.0);
            model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                        e->model.size,
                                                        load_offsets,
                                                        load_sizes,
                                                        load_span_count,
                                                        spans.max_tensor_bytes);
            free(spans.v);
        } else {
            model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                       e->model.size,
                                                       e->model.tensor_data_pos,
                                                       e->model.size - e->model.tensor_data_pos,
                                                       e->model.max_tensor_bytes);
        }
        if (!model_map_ok) {
            fprintf(stderr,
                    "ds4: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->mtp_ready &&
            !ds4_gpu_set_model_map_range(e->mtp_model.map,
                                           e->mtp_model.size,
                                           e->mtp_model.tensor_data_pos,
                                           e->mtp_model.size - e->mtp_model.tensor_data_pos,
                                           e->mtp_model.max_tensor_bytes))
        {
            fprintf(stderr,
                    "ds4: %s failed to map MTP model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->dspark_ready &&
            !ds4_gpu_set_model_map_range(e->dspark_model.map,
                                           e->dspark_model.size,
                                           e->dspark_model.tensor_data_pos,
                                           e->dspark_model.size - e->dspark_model.tensor_data_pos,
                                           e->dspark_model.max_tensor_bytes))
        {
            fprintf(stderr,
                    "ds4: %s failed to map DSpark model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        if (!accelerator_cache_model_tensors(e->backend, &e->model,
                                             load_offsets, load_sizes,
                                             load_span_count)) {
            fprintf(stderr, "ds4: %s failed to prepare optional model cache\n",
                    ds4_backend_name(e->backend));
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        free(load_offsets);
        free(load_sizes);
        /* Also apply explicit optional Q8 preload settings to the MTP support
         * model when loaded. */
        if (e->mtp_ready) {
            (void)ds4_gpu_set_model_fd_for_map(e->mtp_model.fd, e->mtp_model.map);
            if (!accelerator_cache_model_tensors(e->backend, &e->mtp_model,
                                                 NULL, NULL, 0)) {
                fprintf(stderr, "ds4: %s failed to prepare optional MTP model cache\n",
                        ds4_backend_name(e->backend));
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        }
        if (e->dspark_ready) {
            (void)ds4_gpu_set_model_fd_for_map(e->dspark_model.fd, e->dspark_model.map);
            if (!accelerator_cache_model_tensors(e->backend, &e->dspark_model,
                                                 NULL, NULL, 0)) {
                fprintf(stderr, "ds4: %s failed to prepare optional DSpark model cache\n",
                        ds4_backend_name(e->backend));
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        }
        if (e->overlay_ready &&
            !accelerator_prepare_expert_overlay(e->backend, &e->model,
                                                &e->overlay_model)) {
            fprintf(stderr, "ds4: %s failed to prepare expert-overlay spans\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        fprintf(stderr, "ds4: %s backend initialized for graph diagnostics\n",
                ds4_backend_name(e->backend));
    }

    *out = e;
    return 0;
}



void ds4_engine_summary(ds4_engine *e) {
    model_summary(&e->model);
}



int ds4_engine_vocab_size(ds4_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}



int ds4_engine_power(ds4_engine *e) {
    return e ? e->power_percent : 100;
}



int ds4_engine_set_power(ds4_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}



const char *ds4_engine_model_name(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_SHAPE_NAME;
}



int ds4_engine_layer_count(ds4_engine *e) {
    (void)e;
    return (int)DS4_N_LAYER;
}



uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer) {
    (void)e;
    if (layer >= DS4_N_LAYER) return 0;
    return ds4_layer_compress_ratio(layer);
}



uint64_t ds4_engine_hidden_f32_values(ds4_engine *e) {
    (void)e;
    return (uint64_t)DS4_N_HC * DS4_N_EMBD;
}



int ds4_engine_model_id(ds4_engine *e) {
    (void)e;
    return (int)DS4_MODEL_VARIANT;
}



void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    ds4_threads_shutdown();
    if (e->mtp_ready) model_close(&e->mtp_model);
    if (e->dspark_ready) model_close(&e->dspark_model);
    if (e->overlay_ready) model_close(&e->overlay_model);
    model_close(&e->model);
    ds4_gpu_cleanup();
    ds4_ssd_memory_lock_release(&e->simulated_memory);
    ds4_release_instance_lock();
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}



int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (e->backend == DS4_BACKEND_CPU) {
        if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR) {
            fprintf(stderr, "ds4: distributed coordinator sessions require the graph backend\n");
            return 1;
        }
        ds4_session *s = xcalloc(1, sizeof(*s));
        s->engine = e;
        s->ctx_size = ctx_size;
        s->prefill_cap = ds4_prefill_cap_for_prompt(ctx_size,
                                                     e->prefill_chunk);
        kv_cache_init(&s->cpu_cache, (uint32_t)ctx_size, 0);
        cpu_decode_scratch_init(&s->cpu_scratch, (uint32_t)ctx_size);
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        *out = s;
        return 0;
    }
    if (!ds4_backend_uses_graph(e->backend) || !e->gpu_ready) return 1;

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = gpu_graph_prefill_cap_for_prompt(ctx_size,
                                                        e->prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    const ds4_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) {
        fprintf(stderr, "ds4: no transformer layers are loaded\n");
        free(s);
        return 1;
    }
    if (!gpu_graph_alloc_raw_cap(&s->graph, &e->weights, shape_layer,
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap,
                                   e->mtp_ready || e->dspark_ready))
    {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    s->graph.ssd_streaming = e->ssd_streaming;
    s->graph.ssd_streaming_cold = e->ssd_streaming_cold;
    s->graph.streaming_preload_experts = e->ssd_streaming_preload_experts;
    s->graph.power_percent = (uint32_t)e->power_percent;
    if (!gpu_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        gpu_graph_free(&s->graph);
        free(s);
        return 1;
    }
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->mtp_logits[0]));
        s->mtp_draft_token = -1;
    }
    if (e->dspark_ready) {
        if (!gpu_graph_init_dspark_target(&s->graph, e->dspark_weights.target_layer_ids)) {
            fprintf(stderr, "ds4: failed to allocate DSpark graph buffers\n");
            gpu_graph_free(&s->graph);
            free(s->logits);
            free(s->mtp_logits);
            free(s);
            return 1;
        }
    }
    if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR) {
        char err[256];
        if (ds4_dist_session_create(&s->distributed,
                                    e,
                                    &e->distributed,
                                    s,
                                    ctx_size,
                                    err,
                                    sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: failed to create distributed coordinator session: %s\n",
                    err[0] ? err : "unknown error");
            gpu_graph_free(&s->graph);
            free(s->logits);
            free(s->mtp_logits);
            free(s);
            return 1;
        }
    }
    *out = s;
    return 0;
}



void ds4_session_free(ds4_session *s) {
    if (!s) return;
    ds4_dist_session_free(s->distributed);
    if (ds4_session_is_cpu(s)) {
        kv_cache_free(&s->cpu_cache);
        cpu_decode_scratch_free(&s->cpu_scratch);
    }
    else {
        gpu_graph_free(&s->graph);
    }
    token_vec_free(&s->checkpoint);
    free(s->logits);
    free(s->mtp_logits);
    free(s);
}



int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen) {
    if (!s || !s->distributed) {
        if (errlen) snprintf(err, errlen, "session is not a distributed coordinator");
        return -1;
    }
    return ds4_dist_session_route_ready(s->distributed, err, errlen);
}



int ds4_session_power(ds4_session *s) {
    if (!s || !s->engine) return 100;
    return s->engine->power_percent;
}



bool ds4_session_is_distributed(ds4_session *s) {
    return s && s->distributed != NULL;
}



int ds4_session_set_power(ds4_session *s, int power_percent) {
    if (!s || !s->engine || power_percent < 1 || power_percent > 100) return 1;
    s->engine->power_percent = power_percent;
    if (!ds4_session_is_cpu(s)) s->graph.power_percent = (uint32_t)power_percent;
    return 0;
}



void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}



void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}



void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}



static bool ds4_session_cancelled(ds4_session *s) {
    return s && s->cancel && s->cancel(s->cancel_ud);
}



static bool ds4_session_cancelled_cb(void *ud) {
    return ds4_session_cancelled(ud);
}



void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total) {
    if (!s || !s->progress || !event) return;
    s->progress(s->progress_ud, event, current, total);
}



int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen) {
    if (!s) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    ds4_session_invalidate(s);
    if (ds4_session_is_cpu(s)) {
        session_cpu_reset_cache(s);
        return 0;
    }
    if (!gpu_graph_reset_prefill_state(&s->graph)) {
        if (errlen) snprintf(err, errlen, "%s layer-slice state reset failed",
                             ds4_backend_name(s->engine->backend));
        return 1;
    }
    s->graph.mtp_n_raw = 0;
    return 0;
}



int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen) {
    if (!s || !s->engine || !hidden_hc || n_tokens == 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid output-head hidden-state input");
        return 1;
    }

    ds4_engine *e = s->engine;
    if (!weights_have_output_head(&e->weights)) {
        if (errlen) snprintf(err, errlen, "output head is not loaded");
        return 1;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const float *last_hc = hidden_hc + (uint64_t)(n_tokens - 1u) * hc_dim;

    if (ds4_session_is_cpu(s)) {
        output_logits_one(logits, &e->model, &e->weights, last_hc);
        return 0;
    }
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_tensor_write(g->cur_hc,
                                   0,
                                   last_hc,
                                   hc_dim * sizeof(float)) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = gpu_graph_encode_output_head(g,
                                                &e->model,
                                                &e->weights,
                                                e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok) ok = ds4_gpu_tensor_read(g->logits,
                                     0,
                                     logits,
                                     (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after output-head hidden-state failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s output-head hidden-state evaluation failed",
                             ds4_backend_name(e->backend));
        return 1;
    }
    return 0;
}



static int ds4_session_slice_check_timeline(
        ds4_session *s,
        const int   *tokens,
        uint32_t     n_tokens,
        uint32_t     pos0,
        char        *err,
        size_t       errlen) {
    if (!s || !tokens || n_tokens == 0) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice token span");
        return 1;
    }
    const uint32_t ctx_size = (uint32_t)s->ctx_size;
    if (pos0 > (uint32_t)INT_MAX || n_tokens > (uint32_t)INT_MAX ||
        pos0 > ctx_size || n_tokens > ctx_size - pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice token span exceeds context");
        return 1;
    }
    if (!s->checkpoint_valid) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "layer-slice session needs reset before pos %u", pos0);
            return 1;
        }
        return 0;
    }
    if ((uint32_t)s->checkpoint.len != pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice KV position mismatch: have %d want %u",
                             s->checkpoint.len, pos0);
        return 1;
    }
    return 0;
}



static DS4_MAYBE_UNUSED void ds4_session_slice_commit_timeline(ds4_session *s, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
}



int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen) {
    if (!s || !s->engine) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    if (layer_start > layer_end || layer_end >= (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice layer range %u:%u",
                             layer_start, layer_end);
        return 1;
    }
    if (layer_start != 0 && !input_hc) {
        if (errlen) snprintf(err, errlen, "layer-slice layer %u requires input hidden-state",
                             layer_start);
        return 1;
    }
    if (output_logits && layer_end + 1u != (uint32_t)DS4_N_LAYER) {
        if (errlen) snprintf(err, errlen, "layer-slice logits require final transformer layer");
        return 1;
    }
    if (output_logits && !logits) {
        if (errlen) snprintf(err, errlen, "layer-slice logits output is missing");
        return 1;
    }
    if (!weights_layers_bound(&s->engine->weights, layer_start, layer_end)) {
        if (errlen) snprintf(err, errlen, "requested layer slice %u:%u is not loaded",
                             layer_start, layer_end);
        return 1;
    }
    if (!input_hc && !s->engine->weights.token_embd) {
        if (errlen) snprintf(err, errlen, "token embedding is not loaded");
        return 1;
    }
    if (output_logits && !weights_have_output_head(&s->engine->weights)) {
        if (errlen) snprintf(err, errlen, "output head is not loaded");
        return 1;
    }
    /* A distributed prefill pipeline may need only the KV side effect for
     * non-final chunks. In that case both output_hc and logits are NULL. */
    if (ds4_session_slice_check_timeline(s, tokens, n_tokens, pos0, err, errlen) != 0) {
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        if (errlen) snprintf(err, errlen, "layer slices require the graph backend");
        s->checkpoint_valid = false;
        return 1;
    }
    if (n_tokens > s->prefill_cap) {
        if (errlen) snprintf(err, errlen, "layer-slice chunk %u exceeds prefill cap %u",
                             n_tokens, s->prefill_cap);
        return 1;
    }

    ds4_engine *e = s->engine;
    ds4_gpu_graph *g = &s->graph;
    if (!input_hc && !output_hc && output_logits &&
        layer_start == 0 && layer_end + 1u == (uint32_t)DS4_N_LAYER) {
        bool ok = false;
        ds4_tokens span = {0};
        if (pos0 == 0) {
            span.v = (int *)tokens;
            span.len = (int)n_tokens;
            span.cap = (int)n_tokens;
            ok = gpu_graph_prefill_layer_major(g,
                                                 &e->model,
                                                 &e->weights,
                                                 &span,
                                                 0,
                                                 n_tokens,
                                                 logits,
                                                 false,
                                                 NULL,
                                                 NULL,
                                                 NULL);
        } else if (n_tokens == 1) {
            ok = gpu_graph_eval_token_raw_swa(g,
                                                &e->model,
                                                &e->weights,
                                                tokens[0],
                                                pos0,
                                                logits);
        } else {
            if (pos0 > (uint32_t)INT_MAX - n_tokens) {
                if (errlen) snprintf(err, errlen, "layer-slice full span is too large");
                s->checkpoint_valid = false;
                return 1;
            }
            span.len = (int)(pos0 + n_tokens);
            span.cap = span.len;
            span.v = calloc((size_t)span.len, sizeof(span.v[0]));
            if (span.v) {
                for (uint32_t i = 0; i < n_tokens; i++) span.v[pos0 + i] = tokens[i];
                ok = gpu_graph_prefill_layer_major(g,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     pos0,
                                                     n_tokens,
                                                     logits,
                                                     false,
                                                     NULL,
                                                     NULL,
                                                     NULL);
            }
            free(span.v);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice full failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice full evaluation failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }
        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = (uint64_t)n_tokens * hc_dim * sizeof(float);
    if (n_tokens == 1 && pos0 > 0) {
        if (g->raw_cap == 0) {
            if (errlen) snprintf(err, errlen, "%s layer-slice decode has no raw KV cache",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        bool ok = true;
        if (g->ssd_streaming && !input_hc) {
            g->streaming_static_decode_map_current = false;
            ok = gpu_graph_stream_map_token(&e->model, &e->weights);
        }
        if (input_hc) {
            ok = ds4_gpu_tensor_write(g->cur_hc, 0, input_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok && !input_hc) {
            ok = ds4_gpu_embed_token_hc_tensor(g->cur_hc,
                                               e->model.map,
                                               e->model.size,
                                               e->weights.token_embd->abs_offset,
                                               (uint32_t)e->weights.token_embd->dim[1],
                                               (uint32_t)tokens[0],
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;
        }
        const uint32_t raw_row = pos0 % g->raw_cap;
        const uint32_t n_raw = gpu_graph_raw_span_for_batch(g, pos0, 1);
        const uint32_t split_after_layers = gpu_graph_token_split_after_layers();
        uint32_t encoded_layers = 0;
        if (g->ssd_streaming) {
            if (ok) ok = ds4_gpu_end_commands() != 0;
            for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
                g->streaming_static_decode_map_current = false;
                ok = gpu_graph_stream_map_layer_decode(&e->model, &e->weights, il);
                if (ok) ok = ds4_gpu_begin_commands() != 0;
                if (ok) {
                    ok = gpu_graph_encode_decode_layer(g,
                                                         &e->model,
                                                         &e->weights.layer[il],
                                                         il,
                                                         pos0,
                                                         g->layer_raw_cache[il],
                                                         g->raw_cap,
                                                         raw_row,
                                                         n_raw,
                                                         tokens[0]);
                    ds4_gpu_tensor *tmp = g->cur_hc;
                    g->cur_hc = g->after_ffn_hc;
                    g->after_ffn_hc = tmp;
                }
                if (ok) ok = ds4_gpu_end_commands() != 0;
            }
            if (ok && output_logits) {
                g->streaming_static_decode_map_current = false;
                ok = gpu_graph_stream_map_output(&e->model, &e->weights);
                if (ok) ok = ds4_gpu_begin_commands() != 0;
                if (ok) ok = gpu_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
                if (ok) ok = ds4_gpu_end_commands() != 0;
            }
        } else {
            for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
                ok = gpu_graph_encode_decode_layer(g,
                                                     &e->model,
                                                     &e->weights.layer[il],
                                                     il,
                                                     pos0,
                                                     g->layer_raw_cache[il],
                                                     g->raw_cap,
                                                     raw_row,
                                                     n_raw,
                                                     tokens[0]);
                ds4_gpu_tensor *tmp = g->cur_hc;
                g->cur_hc = g->after_ffn_hc;
                g->after_ffn_hc = tmp;
                encoded_layers++;
                if (ok &&
                    split_after_layers != 0 &&
                    encoded_layers == split_after_layers &&
                    il < layer_end)
                {
                    ok = ds4_gpu_flush_commands() != 0;
                }
            }
            if (ok && output_logits) {
                ok = gpu_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
            }
            if (ok) ok = ds4_gpu_end_commands() != 0;
        }
        if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
        if (ok && output_hc) {
            ok = ds4_gpu_tensor_read(g->cur_hc, 0, output_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok && output_logits) {
            ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice decode failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice decode failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    ds4_tokens span = {
        .v = (int *)tokens,
        .len = (int)n_tokens,
        .cap = (int)n_tokens,
    };

    bool ok = true;
    if (g->ssd_streaming && !input_hc) {
        g->streaming_static_decode_map_current = false;
        ok = gpu_graph_stream_map_token(&e->model, &e->weights);
    }
    if (ok) ok = gpu_graph_upload_prompt_tokens(g->prefill_tokens, &span, 0, n_tokens);
    if (ok && input_hc) {
        ok = ds4_gpu_tensor_write(g->batch_cur_hc, 0, input_hc, hc_bytes) != 0;
    } else if (ok) {
        ok = gpu_graph_upload_prompt_embeddings_hc(g->batch_cur_hc,
                                                     g->prefill_tokens,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     0,
                                                     n_tokens);
    }

    ds4_gpu_tensor *last_hc = NULL;
    ds4_gpu_tensor *saved_cur = NULL;
    const bool batch_selected_addr =
        g->ssd_streaming &&
        layer_start == 0 &&
        (gpu_graph_stream_prefill_batch_selected_addr_enabled(g, &e->weights, n_tokens) ||
         gpu_graph_cuda_stream_prefill_batch_selected_addr_enabled(g, &e->weights, n_tokens));
    if (g->ssd_streaming) {
        for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
            g->streaming_static_decode_map_current = false;
            ok = batch_selected_addr ?
                 gpu_graph_stream_map_layer_decode(&e->model, &e->weights, il) :
                 gpu_graph_stream_map_layer(&e->model, &e->weights, il);
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) {
                ok = gpu_graph_encode_layer_batch(g,
                                                    &e->model,
                                                    &e->weights.layer[il],
                                                    il,
                                                    pos0,
                                                    n_tokens);
            }
            if (ok) ok = ds4_gpu_end_commands() != 0;
        }
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
            ok = gpu_graph_encode_layer_batch(g,
                                                &e->model,
                                                &e->weights.layer[il],
                                                il,
                                                pos0,
                                                n_tokens);
        }
    }
    if (ok && output_logits) {
        saved_cur = g->cur_hc;
        last_hc = gpu_graph_tensor_row_view(g->batch_cur_hc, n_tokens - 1u, hc_dim);
        ok = last_hc != NULL;
        if (ok && g->ssd_streaming) {
            g->streaming_static_decode_map_current = false;
            ok = gpu_graph_stream_map_output(&e->model, &e->weights);
        }
        if (ok) {
            g->cur_hc = last_hc;
            if (g->ssd_streaming) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = gpu_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
            if (ok && g->ssd_streaming) ok = ds4_gpu_end_commands() != 0;
            g->cur_hc = saved_cur;
        }
    }
    if (ok && !g->ssd_streaming) ok = ds4_gpu_end_commands() != 0;
    if (saved_cur) g->cur_hc = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);

    if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(g->batch_cur_hc, 0, output_hc, hc_bytes) != 0;
    }
    if (ok && output_logits) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after layer-slice failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s layer-slice failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }

    ds4_session_slice_commit_timeline(s, tokens, n_tokens);
    return 0;
}



static void ds4_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    ds4_sync_progress *p = ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
        p->session->mtp_draft_valid = false;
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}



/* Bring the live backend state to exactly the supplied token prefix.
 *
 * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (ds4_session_cancelled(s)) {
        snprintf(err, errlen, "interrupted");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    s->cont_anchor_valid = false;
    if (s->distributed) {
        const ds4_tokens *checkpoint = s->checkpoint_valid ? &s->checkpoint : NULL;
        return ds4_dist_session_sync(s->distributed,
                                     s,
                                     checkpoint,
                                     prompt,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        if (s->checkpoint_valid &&
            prompt->len >= s->checkpoint.len &&
            ds4_tokens_starts_with(prompt, &s->checkpoint))
        {
            s->mtp_draft_valid = false;
            for (int i = s->checkpoint.len; i < prompt->len; i++) {
                if (ds4_session_cancelled(s)) {
                    snprintf(err, errlen, "interrupted");
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    return DS4_SESSION_SYNC_INTERRUPTED;
                }
                forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                         &e->model,
                                                         &e->weights,
                                                         &s->cpu_cache,
                                                         prompt->v[i],
                                                         (uint32_t)s->checkpoint.len,
                                                         e->directional_steering_dirs,
                                                         e->directional_steering_attn_scale,
                                                         e->directional_steering_ffn_scale,
                                                         &s->cpu_scratch);
                token_vec_push(&s->checkpoint, prompt->v[i]);
                if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i + 1, prompt->len);
            }
            s->checkpoint_valid = true;
            return 0;
        }

        session_cpu_reset_cache(s);
        prefill_layer_major_cpu(s->logits,
                                &e->model,
                                &e->weights,
                                &s->cpu_cache,
                                prompt,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale);
        ds4_tokens_copy(&s->checkpoint, prompt);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", prompt->len, prompt->len);
        return 0;
    }
    ds4_engine *e = s->engine;
    const char *backend_name = ds4_backend_name(e->backend);

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint))
    {
        s->mtp_draft_valid = false;
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = gpu_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            bool cancelled = false;
            ds4_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            bool ok = gpu_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        ds4_session_note_prefill_progress,
                                                        &progress,
                                                        s->display_progress,
                                                        s->display_progress_ud,
                                                        NULL,
                                                        ds4_session_cancelled_cb,
                                                        s,
                                                        &cancelled);
            if (cancelled) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                s->mtp_draft_valid = false;
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (ds4_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                s->mtp_draft_valid = false;
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!gpu_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "%s decode failed while extending checkpoint", backend_name);
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
        }
        return 0;
    }

    bool ok;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    if (!gpu_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "%s prefill state reset failed", backend_name);
        return 1;
    }
    if (s->prefill_cap < (uint32_t)prompt->len) {
        bool cancelled = false;
        ds4_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ok = gpu_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         ds4_session_note_prefill_progress, &progress,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         ds4_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            s->checkpoint_valid = s->checkpoint.len > 0;
            s->mtp_draft_valid = false;
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
    } else {
        bool cancelled = false;
        ok = gpu_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         ds4_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->graph.mtp_n_raw = 0;
    return 0;
}



/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}



/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        snprintf(err, errlen, "prompt exceeds context");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return DS4_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return DS4_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return ds4_session_sync(s, prompt, err, errlen) == 0 ?
            DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
    }

    if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return DS4_SESSION_REWRITE_ERROR;
}



int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}



int ds4_session_argmax(ds4_session *s) {
    return sample_argmax(s->logits, DS4_N_VOCAB);
}



int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    int best = -1;
    float best_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}



int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return 0;
    return sample_top_p_min_p(logits, (uint32_t)n_vocab, temperature, top_k, top_p, min_p, rng);
}



int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    return sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}



int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)DS4_N_VOCAB) k = (int)DS4_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = DS4_NEG_INF;
        out[i].logprob = DS4_NEG_INF;
    }

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : DS4_NEG_INF;
    }
    return k;
}



int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (!s || !out || token < 0 || token >= (int)DS4_N_VOCAB) return 0;

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : DS4_NEG_INF;
    return 1;
}



int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    if (!s || !out || cap < (int)DS4_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)DS4_N_VOCAB * sizeof(out[0]));
    return (int)DS4_N_VOCAB;
}



int ds4_session_set_logits(ds4_session *s, const float *logits, int n) {
    if (!s || !logits || n != (int)DS4_N_VOCAB) return 1;
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}



static int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen) {
    if (!s) return 1;
    if (s->distributed) {
        if (!s->checkpoint_valid) {
            if (errlen) snprintf(err, errlen, "distributed decode requires a valid checkpoint");
            return 1;
        }
        (void)probe_mtp;
        return ds4_dist_session_eval(s->distributed,
                                     s,
                                     &s->checkpoint,
                                     token,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                 &e->model,
                                                 &e->weights,
                                                 &s->cpu_cache,
                                                 token,
                                                 (uint32_t)s->checkpoint.len,
                                                 e->directional_steering_dirs,
                                                 e->directional_steering_attn_scale,
                                                 e->directional_steering_ffn_scale,
                                                 &s->cpu_scratch);
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        (void)probe_mtp;
        return 0;
    }
    ds4_engine *e = s->engine;
    const bool mtp_probe_log = getenv("DS4_MTP_PROBE") != NULL;
    const bool mtp_should_draft =
        probe_mtp && e->mtp_ready && s->mtp_logits &&
        (e->mtp_draft_tokens > 1 || mtp_probe_log);
    if (probe_mtp && s->mtp_draft_valid) {
        if (mtp_probe_log) {
            s->mtp_probe_total++;
            if (s->mtp_draft_token == token) s->mtp_probe_hit++;
            fprintf(stderr,
                    "ds4: mtp probe token=%d draft=%d hit=%llu/%llu\n",
                    token,
                    s->mtp_draft_token,
                    (unsigned long long)s->mtp_probe_hit,
                    (unsigned long long)s->mtp_probe_total);
        }
        s->mtp_draft_valid = false;
    }
    if (!gpu_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    token_vec_push(&s->checkpoint, token);
    s->cont_anchor_valid = false;
    if (mtp_should_draft) {
        int mtp_top = -1;
        if (gpu_graph_eval_mtp_draft(&s->graph,
                                       &e->model,
                                       &e->weights,
                                       &e->mtp_model,
                                       &e->mtp_weights,
                                       token,
                                       (uint32_t)(s->checkpoint.len - 1),
                                       getenv("DS4_MTP_FULL_LOGITS") ? s->mtp_logits : NULL,
                                       &mtp_top)) {
            s->mtp_draft_token = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
            s->mtp_draft_valid = true;
        } else if (getenv("DS4_MTP_PROBE")) {
            fprintf(stderr, "ds4: mtp probe draft failed\n");
        }
    }
    return 0;
}



int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    return ds4_session_eval_internal(s, token, true, err, errlen);
}



/* Speculative decode state machine:
 * 1. commit the normal target token and use its logits to validate draft[0];
 * 2. let MTP recursively draft a tiny suffix from its own raw-cache frontier;
 * 3. verify the suffix with the target graph, committing only the accepted
 *    prefix and rolling back speculative Metal state on miss;
 * 4. fall back to ordinary one-token decode if the fast verifier cannot prove
 *    the target stream. */
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || max_tokens <= 0 || accepted_cap <= 0) return 0;
    if (s->distributed) {
        if (!accepted) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        (void)max_tokens;
        (void)eos_token;
        if (!accepted || accepted_cap <= 0) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    ds4_engine *e = s->engine;
    int n_accept = 0;
    const bool strict_mtp = e->quality || getenv("DS4_MTP_STRICT") != NULL;

    /* Continuous depth-1 speculation (DS4_MTP_CONTINUOUS).  The default path
     * below decodes first_token on its own and then batch-verifies the MTP
     * draft; that standalone decode is a full shared-weight pass the verify
     * could have amortized.  Continuous mode folds the two together: forward
     * [first_token, draft] in a single verify, drafting from the anchor row the
     * previous verify left in batch_cur_hc.  It removes one shared-weight pass
     * per accepted draft (about 0.50 versus 0.70 reads per token).  Same
     * near-greedy class as the batched draft-2 verifier, not bit-exact to a
     * strict decode: a wrong draft is rejected by the verify and never
     * committed, so a stale anchor only lowers acceptance, never corrupts.
     * --quality / DS4_MTP_STRICT ask for a strict decode, so defer to the
     * exact verifier below in that case. */
    if (!strict_mtp && e->mtp_ready && s->mtp_logits &&
        e->mtp_draft_tokens > 1 && s->graph.prefill_cap >= 2 &&
        getenv("DS4_MTP_CONTINUOUS"))
    {
        const uint32_t start = (uint32_t)s->checkpoint.len;

        int draft = -1;
        if (s->cont_anchor_valid) {
            const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
            ds4_gpu_tensor *anchor = gpu_graph_tensor_row_view(s->graph.batch_cur_hc,
                                                                 s->cont_anchor_row, hc_dim);
            const bool drafted = gpu_graph_eval_mtp_draft_from_hc(&s->graph, &e->model, &e->weights,
                                                                    &e->mtp_model, &e->mtp_weights,
                                                                    anchor, s->graph.mtp_state_hc,
                                                                    first_token, start,
                                                                    NULL, &draft);
            if (anchor) ds4_gpu_tensor_free(anchor);
            if (!drafted) draft = -1;
        }

        /*
         * No usable draft (first token of a generation, draft unavailable, EOS,
         * or no room to store or report two tokens): forward first_token alone
         * and seat the anchor on its row, like an ordinary depth-1 cycle.
         */
        if (draft < 0 || first_token == eos_token ||
            max_tokens == 1 || accepted_cap < 2 ||
            start + 2u > (uint32_t)s->ctx_size)
        {
            token_vec_push(&s->checkpoint, first_token);
            bool ok = gpu_graph_verify_suffix_tops(&s->graph, &e->model, &e->weights,
                                                     &s->checkpoint, start, 1,
                                                     false, NULL, NULL) &&
                      gpu_graph_read_spec_logits_row(&s->graph, 0, s->logits);
            if (!ok) {
                s->checkpoint.len = start;
                snprintf(err, errlen, "continuous decode failed");
                s->checkpoint_valid = false;
                s->cont_anchor_valid = false;
                return -1;
            }
            accepted[n_accept++] = first_token;
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            s->cont_anchor_row = 0;
            s->cont_anchor_valid = true;
            return n_accept;
        }

        /*
         * Verify [first_token, draft] together with prefix-1 capture so a
         * rejected draft rewinds cheaply.  first_token is always committed; the
         * draft is committed only when the target agrees at row 0.
         */
        int row_tops[2] = { -1, -1 };
        token_vec_push(&s->checkpoint, first_token);
        token_vec_push(&s->checkpoint, draft);
        bool ok = gpu_graph_verify_suffix_tops(&s->graph, &e->model, &e->weights,
                                                 &s->checkpoint, start, 2,
                                                 true, row_tops, NULL);
        if (ok && row_tops[0] == draft) {
            ok = gpu_graph_read_spec_logits_row(&s->graph, 1, s->logits);
            if (ok) {
                accepted[n_accept++] = first_token;
                if (n_accept < accepted_cap) accepted[n_accept++] = draft;
                s->cont_anchor_row = 1;
            }
        } else if (ok) {
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr, "ds4: mtp cont miss draft=%d top=%d\n", draft, row_tops[0]);
            }
            s->checkpoint.len = start;
            ok = spec_frontier_commit_prefix1(s) &&
                 gpu_graph_read_spec_logits_row(&s->graph, 0, s->logits);
            if (ok) {
                accepted[n_accept++] = first_token;
                token_vec_push(&s->checkpoint, first_token);
                s->cont_anchor_row = 0;
            }
        }
        if (!ok) {
            s->checkpoint.len = start;
            snprintf(err, errlen, "continuous verify failed");
            s->checkpoint_valid = false;
            s->cont_anchor_valid = false;
            return -1;
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        s->cont_anchor_valid = true;
        return n_accept;
    }

    /*
     * MTP in DeepSeek V4 is a speculative drafter, not a replacement sampler.
     * The target model still defines the exact output stream.  A cycle starts
     * by accepting one normal target token, then asks the MTP block to propose
     * a short suffix.  The suffix is useful only if the target model can verify
     * several proposed positions together; running ordinary decode once per
     * draft token is correctness-safe but cannot be faster than baseline.
     */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
    n_accept = 0;
    accepted[n_accept++] = first_token;
    if (first_token == eos_token || max_tokens == 1 || n_accept >= accepted_cap) return n_accept;

    if (!e->mtp_ready || !s->mtp_draft_valid || e->mtp_draft_tokens <= 1) return n_accept;

    int draft_cap = e->mtp_draft_tokens;
    if (draft_cap > max_tokens - n_accept) draft_cap = max_tokens - n_accept;
    if (draft_cap > accepted_cap - n_accept) draft_cap = accepted_cap - n_accept;
    int room = s->ctx_size - s->checkpoint.len;
    if (draft_cap > room - 1) draft_cap = room - 1;
    if (draft_cap <= 0) return n_accept;

    int drafts[16];
    int draft_n = 1;
    drafts[0] = s->mtp_draft_token;
    s->mtp_draft_valid = false;
    float mtp_margin_threshold = e->mtp_margin;
    const char *mtp_margin_env = getenv("DS4_MTP_MIN_MARGIN");
    if (mtp_margin_env && mtp_margin_env[0]) {
        char *end = NULL;
        float v = strtof(mtp_margin_env, &end);
        if (end != mtp_margin_env && v >= 0.0f) mtp_margin_threshold = v;
    }
    const bool mtp_timing = getenv("DS4_MTP_TIMING") != NULL;
    const bool mtp_conf_log = getenv("DS4_MTP_CONF_LOG") != NULL;
    const bool mtp_need_logits = mtp_conf_log ||
        getenv("DS4_MTP_FULL_LOGITS") != NULL ||
        (!strict_mtp && mtp_margin_threshold > 0.0f);
    const double mtp_t0 = mtp_timing ? now_sec() : 0.0;
    double mtp_t_after_draft = mtp_t0;
    float mtp_last_margin = 0.0f;
    int mtp_last_top0 = -1, mtp_last_top1 = -1;

    /*
     * The first proposed token is verified for free: ds4_session_eval() just
     * produced the base logits for the committed prefix.  If MTP disagrees at
     * this point there is no suffix to verify, so the exact behavior is to emit
     * only first_token and skip all speculative work.
     */
    if (sample_argmax(s->logits, DS4_N_VOCAB) != drafts[0]) {
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec miss first draft=%d\n", drafts[0]);
        }
        return n_accept;
    }
    if (drafts[0] == eos_token) draft_cap = 1;
    const uint32_t mtp_base_raw = s->graph.mtp_n_raw;
    /*
     * MTP has its own raw SWA cache. Recursive drafting writes speculative
     * future rows into it; after verification, rows beyond the accepted prefix
     * must become invisible.  We do not copy/rollback the cache body because the
     * next draft attempt will overwrite future slots.  A counter is enough.
     */
#define DS4_MTP_KEEP_ACCEPTED(n_) do { \
        uint32_t keep_ = mtp_base_raw + (uint32_t)(n_); \
        if (keep_ > s->graph.raw_window) keep_ = s->graph.raw_window; \
        s->graph.mtp_n_raw = keep_; \
    } while (0)

    for (; draft_n < draft_cap; draft_n++) {
        ds4_gpu_tensor *prev_hc = (draft_n & 1) ? s->graph.mtp_state_hc : s->graph.mtp_next_hc;
        ds4_gpu_tensor *out_hc = (draft_n & 1) ? s->graph.mtp_next_hc : s->graph.mtp_state_hc;
        int mtp_top = -1;
        if (!gpu_graph_eval_mtp_draft_from_hc(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &e->mtp_model,
                                                &e->mtp_weights,
                                                prev_hc,
                                                out_hc,
                                                drafts[draft_n - 1],
                                                (uint32_t)(s->checkpoint.len + draft_n - 1),
                                                mtp_need_logits ? s->mtp_logits : NULL,
                                                &mtp_top))
        {
            return n_accept;
        }
        drafts[draft_n] = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
        if (drafts[draft_n] == eos_token) {
            draft_n++;
            break;
        }
    }
    if (mtp_conf_log && draft_n > 1) {
        float v0 = 0.0f, v1 = 0.0f;
        logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
        mtp_last_margin = v0 - v1;
    }
    if (mtp_timing) mtp_t_after_draft = now_sec();

    if (!strict_mtp && draft_n == 2 && mtp_margin_threshold > 0.0f) {
        if (!mtp_conf_log) {
            float v0 = 0.0f, v1 = 0.0f;
            logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
            mtp_last_margin = v0 - v1;
        }
        if (mtp_last_margin < mtp_margin_threshold) {
            float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
            const int start = s->checkpoint.len;
            const double verify_t0 = mtp_timing ? now_sec() : 0.0;
            bool ok = gpu_graph_eval_token_raw_swa(&s->graph,
                                                     &e->model,
                                                     &e->weights,
                                                     drafts[0],
                                                     (uint32_t)start,
                                                     row_logits);
            if (!ok) {
                free(row_logits);
                snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
                s->checkpoint_valid = false;
                return -1;
            }
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            free(row_logits);
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            if (mtp_timing) {
                const double done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing margin-skip drafted=2 committed=1 margin=%.3f threshold=%.3f draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        mtp_last_margin,
                        mtp_margin_threshold,
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (done - verify_t0) * 1000.0,
                        (done - mtp_t0) * 1000.0);
            }
            return n_accept;
        }
    }

    /*
     * The useful N=2 verifier is the tiny batch path: it verifies two target
     * positions in one layer-major pass and commits prefix-1 directly on a
     * partial accept.  Like the rest of the non-quality Metal path, it may pick
     * a different greedy token when batched reductions perturb nearly-tied
     * logits.  --quality / DS4_MTP_STRICT selects the exact decode verifier,
     * which preserves the one-token target stream but is not a speed win.
     */
    const bool use_decode2_exact =
        draft_n == 2 && strict_mtp && getenv("DS4_MTP_BATCH_VERIFY") == NULL;
    if (use_decode2_exact) {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        float *row0_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row0_logits[0]));
        const int start = s->checkpoint.len;
        int row0_top = -1;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        bool have_frontier = spec_frontier_snapshot(&frontier, s);
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        bool ok = have_frontier;
        if (ok) {
            ok = gpu_graph_verify_decode2_exact(&s->graph,
                                                  &e->model,
                                                  &e->weights,
                                                  drafts[0],
                                                  drafts[1],
                                                  (uint32_t)start,
                                                  &row0_top,
                                                  row0_logits,
                                                  row_logits);
        }
        const double verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok && row0_top == drafts[1]) {
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            token_vec_push(&s->checkpoint, drafts[0]);
            token_vec_push(&s->checkpoint, drafts[1]);
            accepted[n_accept++] = drafts[0];
            if (n_accept < accepted_cap) accepted[n_accept++] = drafts[1];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(2);
            if (mtp_timing) {
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=2 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (now_sec() - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }

        if (ok) {
            s->checkpoint.len = start;
            ok = spec_frontier_commit_prefix1(s);
        }
        if (ok) memcpy(s->logits, row0_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        if (ok) {
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            if (mtp_timing) {
                const double replay_done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=1 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (replay_done - verify_done) * 1000.0,
                        (replay_done - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }
        if (have_frontier) {
            s->checkpoint.len = start;
            (void)spec_frontier_restore(&frontier, s);
        }
        spec_frontier_free(&frontier);
        free(row0_logits);
        free(row_logits);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp decode2 verifier failed, falling back to sequential\n");
        }
    }

    if (!use_decode2_exact)
    {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        int *row_tops = xmalloc((size_t)draft_n * sizeof(row_tops[0]));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        const int start = s->checkpoint.len;
        /*
         * The production MTP depth is two.  Prefix-1 capture makes partial
         * accepts cheap, but it copies per-layer compressor frontiers even when
         * both draft tokens are accepted.  Full accepts are the path that makes
         * MTP worthwhile, so by default we snapshot before the verifier and
         * replay one token on partial accept.  DS4_MTP_CAPTURE_PREFIX1 restores
         * the older no-replay partial path for measurement.
         */
        const bool capture_prefix1 =
            draft_n == 2 && (!strict_mtp || getenv("DS4_MTP_CAPTURE_PREFIX1") != NULL);
        const bool exact_replay_debug = getenv("DS4_MTP_EXACT_REPLAY") != NULL;
        const bool snapshot_required =
            draft_n > 2 ||
            (draft_n == 2 && (!capture_prefix1 || exact_replay_debug)) ||
            getenv("DS4_MTP_FORCE_SNAPSHOT") != NULL;
        bool have_frontier = false;
        bool ok = true;
        bool verifier_may_have_mutated = false;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        if (snapshot_required) {
            have_frontier = spec_frontier_snapshot(&frontier, s);
            ok = have_frontier;
        }
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);
            verifier_may_have_mutated = true;
            ok = gpu_graph_verify_suffix_tops(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &s->checkpoint,
                                                (uint32_t)start,
                                                (uint32_t)draft_n,
                                                capture_prefix1,
                                                row_tops,
                                                NULL);
        }
        const double micro_verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            int commit_drafts = 1;
            for (int i = 1; i < draft_n; i++) {
                if (row_tops[i - 1] != drafts[i]) break;
                commit_drafts++;
            }
            if (mtp_conf_log) {
                fprintf(stderr,
                        "ds4: mtp conf drafted=%d committed=%d mtp_top=%d runner=%d margin=%.6f target_next=%d draft_next=%d\n",
                        draft_n,
                        commit_drafts,
                        mtp_last_top0,
                        mtp_last_top1,
                        mtp_last_margin,
                        draft_n > 1 ? row_tops[0] : -1,
                        draft_n > 1 ? drafts[1] : -1);
            }
            if (exact_replay_debug && have_frontier) {
                s->checkpoint.len = start;
                ok = spec_frontier_restore(&frontier, s);
                if (ok) {
                    int replayed = 0;
                    for (; replayed < commit_drafts && ok; replayed++) {
                        ok = gpu_graph_eval_token_raw_swa(&s->graph,
                                                            &e->model,
                                                            &e->weights,
                                                            drafts[replayed],
                                                            (uint32_t)(start + replayed),
                                                            row_logits);
                        if (ok) token_vec_push(&s->checkpoint, drafts[replayed]);
                    }
                    if (ok) {
                        memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                        for (int i = 0; i < replayed && n_accept < accepted_cap; i++) {
                            accepted[n_accept++] = drafts[i];
                            if (drafts[i] == eos_token) break;
                        }
                        s->checkpoint_valid = true;
                        s->mtp_draft_valid = false;
                        DS4_MTP_KEEP_ACCEPTED(replayed);
                        spec_frontier_free(&frontier);
                        free(row_logits);
                        free(row_tops);
                        return n_accept;
                    }
                }
            }

            if (commit_drafts == draft_n) {
                ok = gpu_graph_read_spec_logits_row(&s->graph,
                                                      (uint32_t)(draft_n - 1),
                                                      row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(draft_n);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                                draft_n,
                                draft_n,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }

            if (draft_n == 2 && commit_drafts == 1 && capture_prefix1) {
                s->checkpoint.len = start;
                const double prefix_t0 = mtp_timing ? now_sec() : 0.0;
                ok = spec_frontier_commit_prefix1(s);
                const double prefix_done = mtp_timing ? now_sec() : 0.0;
                if (ok) ok = gpu_graph_read_spec_logits_row(&s->graph, 0, row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms noreplay=1\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (prefix_done - prefix_t0) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            } else {
                s->checkpoint.len = start;
                ok = have_frontier && spec_frontier_restore(&frontier, s);
            }
            if (ok && draft_n == 2 && commit_drafts == 1) {
                ok = gpu_graph_eval_token_raw_swa(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    drafts[0],
                                                    (uint32_t)start,
                                                    row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms exact_replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
            if (ok) {
                for (int i = 0; i < commit_drafts; i++) token_vec_push(&s->checkpoint, drafts[i]);
                ok = gpu_graph_verify_suffix_tops(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    &s->checkpoint,
                                                    (uint32_t)start,
                                                    (uint32_t)commit_drafts,
                                                    false,
                                                    row_tops,
                                                    NULL);
                if (ok) ok = gpu_graph_read_spec_logits_row(&s->graph,
                                                              (uint32_t)(commit_drafts - 1),
                                                              row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < commit_drafts && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(commit_drafts);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
        }
        s->checkpoint.len = start;
        if (have_frontier) {
            (void)spec_frontier_restore(&frontier, s);
        } else if (!verifier_may_have_mutated) {
            /* Snapshot setup failed before the verifier touched Metal state.
             * Fall through to the exact sequential verifier below. */
        } else {
            snprintf(err, errlen, "MTP verifier failed");
            s->checkpoint_valid = false;
            DS4_MTP_KEEP_ACCEPTED(0);
            spec_frontier_free(&frontier);
            free(row_logits);
            free(row_tops);
            return -1;
        }
        spec_frontier_free(&frontier);
        free(row_logits);
        free(row_tops);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec micro verifier failed, falling back to sequential\n");
        }
    }

    /*
     * Safety fallback: if the production microbatch verifier fails, verify
     * drafts with the exact normal one-token decode path instead of returning
     * wrong state.  This path is deliberately slow and should not be selected
     * during normal --mtp operation.
     */
    int verified = 0;
    int target_top = sample_argmax(s->logits, DS4_N_VOCAB);
    bool logits_on_host = true;
    const double seq_t0 = mtp_timing ? now_sec() : 0.0;
    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
        if (target_top != drafts[i]) {
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: mtp spec seq miss at=%d draft=%d base=%d drafted=%d accepted=%d\n",
                        i,
                        drafts[i],
                        target_top,
                        draft_n,
                        n_accept);
            }
            break;
        }
        if (!gpu_graph_eval_token_raw_swa_top(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                drafts[i],
                                                (uint32_t)s->checkpoint.len,
                                                &target_top,
                                                NULL))
        {
            snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return -1;
        }
        token_vec_push(&s->checkpoint, drafts[i]);
        logits_on_host = false;
        accepted[n_accept++] = drafts[i];
        verified++;
        if (drafts[i] == eos_token) break;
    }
    if (verified > 0 && !logits_on_host) {
        if (ds4_gpu_tensor_read(s->graph.logits,
                                  0,
                                  s->logits,
                                  (uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])) == 0)
        {
            snprintf(err, errlen, "%s logits readback failed", ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return -1;
        }
        logits_on_host = true;
    }
    (void)logits_on_host;
    DS4_MTP_KEEP_ACCEPTED(verified);
#undef DS4_MTP_KEEP_ACCEPTED
    if (mtp_timing) {
        fprintf(stderr,
                "ds4: mtp timing seq drafted=%d verified=%d draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                draft_n,
                verified,
                (mtp_t_after_draft - mtp_t0) * 1000.0,
                (now_sec() - seq_t0) * 1000.0,
                (now_sec() - mtp_t0) * 1000.0);
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        if (verified == draft_n) {
            fprintf(stderr,
                    "ds4: mtp spec seq accept drafted=%d accepted=%d\n",
                    draft_n,
                    n_accept);
        } else {
            fprintf(stderr,
                    "ds4: mtp spec seq partial drafted=%d verified=%d accepted=%d\n",
                    draft_n,
                    verified,
                    n_accept);
        }
    }
    return n_accept;
}

int ds4_session_eval_speculative_block(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || max_tokens <= 0 || accepted_cap <= 0 || !accepted) return 0;
    if (!ds4_engine_has_dspark(s->engine) || s->distributed || ds4_session_is_cpu(s)) {
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }

    ds4_engine *e = s->engine;
    ds4_gpu_graph *g = &s->graph;
    const uint32_t n_draft = (uint32_t)e->dspark_draft_tokens;
    int n_accept = 0;
    const int dspark_stats = getenv("DS4_DSPARK_STATS") != NULL;
    const double dspark_t0 = dspark_stats ? now_sec() : 0.0;
    double dspark_draft_ms = 0.0;
    int dspark_base0 = -1;   /* draft-forward's row-0 argmax (pre-markov) */

    /* Step 1: run target decode for the first token */
    if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;

    /* Step 2: project main_x from captured target hidden states */
    if (!gpu_graph_dspark_project_main_x(g, &e->dspark_model, &e->dspark_weights))
        return -1;

    /* Step 2b: seed the current frontier's main_kv into the drafter KV BEFORE the
     * draft forward, matching the reference DSparkAttention (store main_kv at
     * start_pos, then attend).  The old code seeded only at the END of the step
     * (Step 7) and only on accepts, so the draft forward attended to an empty /
     * stale context.  One row per step -> dspark_n_raw tracks the frontier. */
    gpu_graph_dspark_seed_draft_kv(g, &e->dspark_model, &e->dspark_weights, 1);

    /* Step 3: build draft input — position 0 = first_token, rest = noise */
    int32_t draft_ids[16];
    draft_ids[0] = (int32_t)first_token;
    for (uint32_t i = 1; i < n_draft; i++)
        draft_ids[i] = DS4_DSPARK_NOISE_TOKEN_ID;

    /* Step 4: run draft forward → N-token base logits in g->spec_logits */
    const double dspark_draft_t0 = dspark_stats ? now_sec() : 0.0;
    if (!gpu_graph_dspark_draft_forward(g, &e->model, &e->weights,
                                         &e->dspark_model, &e->dspark_weights,
                                         g->spec_logits, draft_ids, n_draft))
        return -1;
    if (dspark_stats) {
        (void)ds4_gpu_synchronize();
        dspark_draft_ms = (now_sec() - dspark_draft_t0) * 1000.0;
        float *r0 = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (gpu_graph_read_spec_logits_row(g, 0, r0))
            dspark_base0 = sample_argmax(r0, DS4_N_VOCAB);
        free(r0);
        /* conditioning diagnostics: is target_h captured, main_x sane, KV seeded? */
        float *tmp = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        double mn = 0.0, th[3] = {0, 0, 0};
        if (ds4_gpu_tensor_read(g->dspark_main_x, 0, tmp, (uint64_t)DS4_N_EMBD * 4))
            for (int j = 0; j < (int)DS4_N_EMBD; j++) mn += (double)tmp[j] * tmp[j];
        for (int i = 0; i < 3; i++)
            if (ds4_gpu_tensor_read(g->dspark_target_h[i], 0, tmp, (uint64_t)DS4_N_EMBD * 4))
                for (int j = 0; j < (int)DS4_N_EMBD; j++) th[i] += (double)tmp[j] * tmp[j];
        free(tmp);
        fprintf(stderr, "ds4: dspark cond main_x=%.3f target_h=[%.2f,%.2f,%.2f] n_raw=[%u,%u,%u]\n",
                sqrt(mn), sqrt(th[0]), sqrt(th[1]), sqrt(th[2]),
                g->dspark_n_raw[0], g->dspark_n_raw[1], g->dspark_n_raw[2]);
    }

    /* Step 5: Markov refine — sequential over N positions */
    const ds4_dspark_weights *w = &e->dspark_weights;
    const uint32_t embed_dim = 256;
    const uint32_t vocab_size = w->vocab_size;
    const uint64_t vocab_bytes = (uint64_t)vocab_size * sizeof(float);
    const void *dmap = e->dspark_model.map;
    const uint64_t dsize = e->dspark_model.size;

    ds4_gpu_tensor *dspark_logits = ds4_gpu_tensor_alloc(vocab_bytes);
    if (!dspark_logits) return -1;

    /* refined_ids[0] holds first_token; positions 1..n_draft hold the refined
     * drafts, so the array needs n_draft + 1 slots (17 at the clamp of 16). */
    int32_t refined_ids[17];
    refined_ids[0] = (int32_t)first_token;
    for (uint32_t pos = 0; pos < n_draft; pos++) {
        /* Create view of spec_logits row [pos] as base logits */
        ds4_gpu_tensor *base_row = ds4_gpu_tensor_view(
            g->spec_logits, (uint64_t)pos * vocab_bytes, vocab_bytes);
        if (!base_row) { ds4_gpu_tensor_free(dspark_logits); return -1; }

        int32_t prev = refined_ids[pos];
        bool step_ok = ds4_gpu_dspark_markov_step_model(dspark_logits, &refined_ids[pos + 1],
                                               base_row,
                                               dmap, dsize,
                                               w->markov_w1->abs_offset,
                                               w->markov_w2->abs_offset,
                                               (int32_t)prev, vocab_size, embed_dim);
        ds4_gpu_tensor_free(base_row);
        if (!step_ok) {
            ds4_gpu_tensor_free(dspark_logits);
            return -1;
        }
    }
    ds4_gpu_tensor_free(dspark_logits);

    /* Step 6: verify drafts against target, rejection sampling.
     *
     * Step 1 already committed first_token and left s->logits = P(next |
     * first_token), i.e. the target's prediction of the first draft.  The
     * batch verify below runs the target over the draft_n draft positions and
     * fills row_tops[i-1] with its argmax after committing refined_ids[1..i],
     * so it validates drafts 2..draft_n; the first draft is validated for free
     * against s->logits. */
    const int saved_len = s->checkpoint.len;
    const int draft_n = (int)n_draft;

    ds4_spec_frontier frontier;
    memset(&frontier, 0, sizeof(frontier));
    bool have_frontier = spec_frontier_snapshot(&frontier, s);

    for (int i = 0; i < draft_n; i++)
        token_vec_push(&s->checkpoint, refined_ids[i + 1]);

    int *row_tops = xmalloc((size_t)draft_n * sizeof(int));
    bool verify_ok = gpu_graph_verify_suffix_tops(g, &e->model, &e->weights,
                                                    &s->checkpoint,
                                                    (uint32_t)saved_len,
                                                    (uint32_t)draft_n,
                                                    false,
                                                    row_tops, NULL);
    if (!verify_ok) {
        s->checkpoint.len = saved_len;
        if (have_frontier) (void)spec_frontier_restore(&frontier, s);
        spec_frontier_free(&frontier);
        free(row_tops);
        snprintf(err, errlen, "DSpark verifier failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Accept the longest prefix the target agrees with.  The first draft is
     * gated by the free s->logits prediction; each subsequent draft by the
     * verifier's row_tops.  commit_drafts may be 0 when even the first draft
     * disagrees. */
    int commit_drafts = 0;
    if (sample_argmax(s->logits, DS4_N_VOCAB) == refined_ids[1]) {
        commit_drafts = 1;
        for (int i = 1; i < draft_n; i++) {
            if (row_tops[i - 1] != refined_ids[i + 1]) break;
            commit_drafts++;
        }
    }

    if (dspark_stats)
        fprintf(stderr, "ds4: dspark step draft_n=%d committed=%d tgt_next=%d base0_top=%d refined1=%d "
                        "base0_hit=%d refined1_hit=%d draft_ms=%.1f step_ms=%.1f\n",
                draft_n, commit_drafts,
                sample_argmax(s->logits, DS4_N_VOCAB), dspark_base0, refined_ids[1],
                dspark_base0 == sample_argmax(s->logits, DS4_N_VOCAB),
                refined_ids[1] == sample_argmax(s->logits, DS4_N_VOCAB),
                dspark_draft_ms, (now_sec() - dspark_t0) * 1000.0);

    bool ok_state = true;
    if (commit_drafts == draft_n) {
        /*
         * Full accept: every draft stays committed and the verifier already
         * advanced the target KV over all of them, so no rollback is needed.
         * s->logits is still P(next | first_token) though — refresh it to the
         * target distribution after the last committed draft (spec_logits row
         * draft_n-1), matching the MTP full-accept path.
         */
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        ok_state = gpu_graph_read_spec_logits_row(g, (uint32_t)(draft_n - 1), row_logits);
        if (ok_state)
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        free(row_logits);
    } else {
        /*
         * Partial accept or full reject: the verifier ran the target over all
         * draft_n positions, so its KV/compressor frontier is now ahead of the
         * commit_drafts we are keeping.  Roll back to the pre-draft frontier and
         * replay exactly the committed drafts, which also refreshes s->logits.
         * This is only possible with a valid snapshot — treat its absence as a
         * hard error rather than leaving rejected drafts committed.
         */
        if (!have_frontier) {
            spec_frontier_free(&frontier);
            free(row_tops);
            snprintf(err, errlen, "DSpark frontier snapshot failed");
            s->checkpoint_valid = false;
            return -1;
        }
        s->checkpoint.len = saved_len;
        ok_state = spec_frontier_restore(&frontier, s);
        for (int i = 0; ok_state && i < commit_drafts; i++) {
            if (ds4_session_eval(s, refined_ids[i + 1], err, errlen) != 0)
                ok_state = false;
        }
    }

    spec_frontier_free(&frontier);
    free(row_tops);
    if (!ok_state) {
        snprintf(err, errlen, "DSpark state update failed");
        s->checkpoint_valid = false;
        return -1;
    }

    /* Step 7: nothing to seed here.  Step 2b seeds first_token's main_kv before
     * the forward (one row per step, correct content).  Reusing this step's
     * main_x for the ACCEPTED positions was measured to HURT (28.6% -> 23.9%):
     * the drafter is sensitive to per-position content, so accepted positions
     * need their OWN captured target hidden (see verify-capture, TODO). */

    /* Step 8: return first_token followed by the accepted drafts.  The caller
     * emits only the tokens returned here, so first_token — committed to the
     * target KV in Step 1 — must lead the list or it is dropped from the
     * output while remaining in context. */
    accepted[n_accept++] = first_token;
    for (int i = 0; i < commit_drafts && n_accept < accepted_cap; i++) {
        if (refined_ids[i + 1] == eos_token) break;
        accepted[n_accept++] = refined_ids[i + 1];
    }

    return n_accept;
}



void ds4_session_invalidate(ds4_session *s) {
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    s->cont_anchor_valid = false;
}



void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->mtp_draft_valid = false;
    s->cont_anchor_valid = false;
}



int ds4_session_pos(ds4_session *s) {
    return s->checkpoint.len;
}



int ds4_session_ctx(ds4_session *s) {
    return s->ctx_size;
}



int ds4_session_prefill_cap(ds4_session *s) {
    return s ? (int)s->prefill_cap : 0;
}

