#include "ds4_cuda_internal.h"


/*
 * Each block reduces the argmax over its slice of the vocabulary and writes
 * that partial winner to block_best_id[blockIdx.x] / block_best_val[blockIdx.x].
 * The kernel does NOT reduce across blocks — the host caller reads the
 * grid_dim partials back and picks the global argmax.  (An earlier version
 * wrote only element 0 into a 4-byte buffer, which both overran the allocation
 * and returned block 0's argmax over just the first blockDim entries.)
 */
__global__ static void dspark_markov_step_kernel(
        float *refined_logits,
        int32_t *block_best_id,
        float *block_best_val,
        const float *base_logits,
        const float *markov_w1,
        const float *markov_w2,
        int32_t prev_token,
        uint32_t vocab_size,
        uint32_t embed_dim) {
    const float *embed = markov_w1 + (uint64_t)prev_token * embed_dim;
    float best_val = -INFINITY;
    int32_t best_id = 0;

    for (uint32_t v = threadIdx.x + blockIdx.x * blockDim.x; v < vocab_size;
         v += blockDim.x * gridDim.x) {
        float dot = 0.0f;
        const float *w2_row = markov_w2 + (uint64_t)v * embed_dim;
        for (uint32_t i = 0; i < embed_dim; i++) dot += w2_row[i] * embed[i];
        float val = base_logits[v] + dot;
        refined_logits[v] = val;
        if (val > best_val) { best_val = val; best_id = (int32_t)v; }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (best_vals[tid + stride] > best_vals[tid]) {
                best_vals[tid] = best_vals[tid + stride];
                best_ids[tid] = best_ids[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_best_id[blockIdx.x] = best_ids[0];
        block_best_val[blockIdx.x] = best_vals[0];
    }
}

/*
 * Read the per-block argmax partials back to the host and pick the global
 * winner.  Blocks map to ascending contiguous vocab ranges (grid_dim =
 * ceil(vocab/blockDim), one range per block), and the per-block reduction
 * breaks ties toward the lowest id, so a strict '>' here yields the
 * lowest-id global argmax — matching a sequential argmax over the vocab.
 */
static int dspark_markov_reduce_blocks(const ds4_gpu_tensor *id_dev,
                                        const ds4_gpu_tensor *val_dev,
                                        uint32_t grid_dim,
                                        int32_t *refined_id_dst) {
    int32_t *ids = (int32_t *)malloc((size_t)grid_dim * sizeof(int32_t));
    float *vals = (float *)malloc((size_t)grid_dim * sizeof(float));
    int rc = 0;
    if (ids && vals &&
        ds4_gpu_tensor_read(id_dev, 0, ids, (uint64_t)grid_dim * sizeof(int32_t)) &&
        ds4_gpu_tensor_read(val_dev, 0, vals, (uint64_t)grid_dim * sizeof(float))) {
        int32_t best_id = ids[0];
        float best_val = vals[0];
        for (uint32_t b = 1; b < grid_dim; b++) {
            if (vals[b] > best_val) { best_val = vals[b]; best_id = ids[b]; }
        }
        *refined_id_dst = best_id;
        rc = 1;
    }
    free(ids);
    free(vals);
    return rc;
}


extern "C" int ds4_gpu_dspark_markov_step_model(
        ds4_gpu_tensor *refined_logits,
        int32_t *refined_id_dst,
        const ds4_gpu_tensor *base_logits,
        const void *dspark_model_map,
        uint64_t dspark_model_size,
        uint64_t markov_w1_offset,
        uint64_t markov_w2_offset,
        int32_t prev_token,
        uint32_t vocab_size,
        uint32_t embed_dim) {
    if (!refined_logits || !refined_id_dst || !base_logits || !dspark_model_map)
        return 0;
    if (vocab_size == 0 || embed_dim == 0 || embed_dim > 1024) return 0;
    if (refined_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (base_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if ((uint64_t)prev_token >= vocab_size) return 0;

    const uint64_t w_bytes = (uint64_t)vocab_size * embed_dim * sizeof(float);
    if (markov_w1_offset > dspark_model_size ||
        w_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (markov_w2_offset > dspark_model_size ||
        w_bytes > dspark_model_size - markov_w2_offset) return 0;

    const float *w1 = (const float *)cuda_model_range_ptr(
        dspark_model_map, markov_w1_offset, w_bytes, "dspark_markov_w1");
    const float *w2 = (const float *)cuda_model_range_ptr(
        dspark_model_map, markov_w2_offset, w_bytes, "dspark_markov_w2");
    if (!w1 || !w2) return 0;

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (vocab_size + block_dim - 1) / block_dim;
    if (grid_dim > 65535) return 0;

    /* Persistent reduce buffers: grid_dim is fixed for a given vocab, and this
     * runs once per draft position per spec step -- per-call cudaMalloc/cudaFree
     * pairs were 2 device-serializing allocs each. Single submission thread. */
    static ds4_gpu_tensor *id_dev = NULL;
    static ds4_gpu_tensor *val_dev = NULL;
    static uint32_t reduce_cap = 0;
    if (grid_dim > reduce_cap) {
        ds4_gpu_tensor_free(id_dev);
        ds4_gpu_tensor_free(val_dev);
        id_dev = ds4_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        val_dev = ds4_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        reduce_cap = (id_dev && val_dev) ? grid_dim : 0;
    }
    if (!id_dev || !val_dev) return 0;

    dspark_markov_step_kernel<<<grid_dim, block_dim>>>(
        (float *)refined_logits->ptr,
        (int32_t *)id_dev->ptr,
        (float *)val_dev->ptr,
        (const float *)base_logits->ptr,
        w1, w2, prev_token, vocab_size, embed_dim);

    int rc = 0;
    if (cudaGetLastError() == cudaSuccess)
        rc = dspark_markov_reduce_blocks(id_dev, val_dev, grid_dim, refined_id_dst);
    return rc;
}


/* DSpark confidence head (model.py DSparkConfidenceHead): per block position,
 * scores[p] = sigmoid( proj . cat(hidden[p], markov_embed[token_ids[p]]) ), where
 * hidden is the post-hc_head drafter hidden and markov_embed is a row of markov_w1
 * gathered by token id. Used to size the verify budget (confidence-scheduled
 * verification) -- higher confidence => the draft is more likely accepted. */
__global__ static void dspark_confidence_score_kernel(
        float *scores,
        const float *hidden,        /* [n_positions, hidden_dim] */
        const int32_t *token_ids,   /* [n_positions] */
        const float *markov_w1,     /* [vocab, embed_dim] */
        const float *proj,          /* [hidden_dim + embed_dim] */
        uint32_t n_positions, uint32_t hidden_dim, uint32_t embed_dim, uint32_t vocab_size) {
    uint32_t p = blockIdx.x;
    if (p >= n_positions) return;
    int32_t t = token_ids[p];
    if (t < 0 || (uint32_t)t >= vocab_size) t = 0;
    const float *hp = hidden + (uint64_t)p * hidden_dim;
    const float *emb = markov_w1 + (uint64_t)t * embed_dim;
    float dot = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) dot += hp[i] * proj[i];
    for (uint32_t i = threadIdx.x; i < embed_dim; i += blockDim.x) dot += emb[i] * proj[hidden_dim + i];
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[p] = 1.0f / (1.0f + expf(-partial[0]));
}


extern "C" int ds4_gpu_dspark_confidence_score_model(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *token_ids,
        const void *dspark_model_map,
        uint64_t dspark_model_size,
        uint64_t markov_w1_offset,
        uint64_t proj_offset,
        uint32_t n_positions, uint32_t hidden_dim, uint32_t embed_dim, uint32_t vocab_size) {
    if (!scores || !hidden || !token_ids || !dspark_model_map) return 0;
    if (n_positions == 0 || hidden_dim == 0 || embed_dim == 0 || vocab_size == 0) return 0;
    if (scores->bytes < (uint64_t)n_positions * sizeof(float)) return 0;
    if (hidden->bytes < (uint64_t)n_positions * hidden_dim * sizeof(float)) return 0;
    if (token_ids->bytes < (uint64_t)n_positions * sizeof(int32_t)) return 0;
    const uint64_t w1_bytes = (uint64_t)vocab_size * embed_dim * sizeof(float);
    const uint64_t proj_bytes = (uint64_t)(hidden_dim + embed_dim) * sizeof(float);
    if (markov_w1_offset > dspark_model_size || w1_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (proj_offset > dspark_model_size || proj_bytes > dspark_model_size - proj_offset) return 0;
    const float *w1 = (const float *)cuda_model_range_ptr(dspark_model_map, markov_w1_offset, w1_bytes, "dspark_conf_w1");
    const float *proj = (const float *)cuda_model_range_ptr(dspark_model_map, proj_offset, proj_bytes, "dspark_conf_proj");
    if (!w1 || !proj) return 0;
    dspark_confidence_score_kernel<<<n_positions, 256>>>(
        (float *)scores->ptr, (const float *)hidden->ptr, (const int32_t *)token_ids->ptr,
        w1, proj, n_positions, hidden_dim, embed_dim, vocab_size);
    return cuda_ok(cudaGetLastError(), "dspark confidence score");
}


__global__ static void dspark_hc_mean_reduce_kernel(
        float *out,
        const float *after_ffn_hc,
        uint32_t n_embd,
        uint32_t n_hc) {
    for (uint32_t d = threadIdx.x + blockIdx.x * blockDim.x; d < n_embd;
         d += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (uint32_t hc = 0; hc < n_hc; hc++)
            sum += after_ffn_hc[(uint64_t)hc * n_embd + d];
        out[d] = sum / (float)n_hc;
    }
}

extern "C" int ds4_gpu_dspark_hc_mean_reduce(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *after_ffn_hc,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (!out || !after_ffn_hc || n_embd == 0 || n_hc == 0) return 0;
    if (out->bytes < (uint64_t)n_embd * sizeof(float)) return 0;
    if (after_ffn_hc->bytes < (uint64_t)n_hc * n_embd * sizeof(float)) return 0;

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (n_embd + block_dim - 1) / block_dim;

    dspark_hc_mean_reduce_kernel<<<grid_dim, block_dim>>>(
        (float *)out->ptr,
        (const float *)after_ffn_hc->ptr,
        n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "dspark hc mean reduce");
}


/* Batched variant over a [n_tokens, n_hc, n_embd] HC tensor: out[p] = mean over
 * hc of in[p]. Captures the drafter's anchor-layer hidden for EVERY position of
 * a spec verify batch, so the fused loop can pick the last-accepted position's
 * hidden without replaying. */
__global__ static void dspark_hc_mean_reduce_batch_kernel(
        float *out,
        const float *hc_batch,
        uint32_t n_embd,
        uint32_t n_hc) {
    const uint32_t p = blockIdx.y;
    const float *in = hc_batch + (uint64_t)p * n_hc * n_embd;
    float *op = out + (uint64_t)p * n_embd;
    for (uint32_t d = threadIdx.x + blockIdx.x * blockDim.x; d < n_embd;
         d += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (uint32_t hc = 0; hc < n_hc; hc++)
            sum += in[(uint64_t)hc * n_embd + d];
        op[d] = sum / (float)n_hc;
    }
}


extern "C" int ds4_gpu_dspark_hc_mean_reduce_batch(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *hc_batch,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens) {
    if (!out || !hc_batch || n_embd == 0 || n_hc == 0 || n_tokens == 0) return 0;
    if (out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float)) return 0;
    if (hc_batch->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) return 0;

    const uint32_t block_dim = 256;
    dim3 grid((n_embd + block_dim - 1) / block_dim, n_tokens, 1);
    dspark_hc_mean_reduce_batch_kernel<<<grid, block_dim>>>(
        (float *)out->ptr,
        (const float *)hc_batch->ptr,
        n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "dspark hc mean reduce batch");
}
