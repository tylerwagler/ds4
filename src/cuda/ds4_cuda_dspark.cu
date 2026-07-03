#include "ds4_cuda_internal.h"


__global__ static void dspark_markov_step_kernel(
        float *refined_logits,
        int32_t *refined_id,
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
        refined_id[blockIdx.x] = best_ids[0];
        if (blockIdx.x == 0) *refined_id = best_ids[0];
    }
}


extern "C" int ds4_gpu_dspark_markov_step(
        ds4_gpu_tensor *refined_logits,
        int32_t *refined_id_dst,
        const ds4_gpu_tensor *base_logits,
        const ds4_gpu_tensor *markov_w1,
        const ds4_gpu_tensor *markov_w2,
        int32_t prev_token,
        uint32_t vocab_size,
        uint32_t embed_dim) {
    if (!refined_logits || !refined_id_dst || !base_logits || !markov_w1 || !markov_w2)
        return 0;
    if (vocab_size == 0 || embed_dim == 0 || embed_dim > 1024) return 0;
    if (refined_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (base_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (markov_w1->bytes < (uint64_t)vocab_size * embed_dim * sizeof(float)) return 0;
    if (markov_w2->bytes < (uint64_t)vocab_size * embed_dim * sizeof(float)) return 0;
    if ((uint64_t)prev_token >= vocab_size) return 0;

    ds4_gpu_tensor *id_dev = ds4_gpu_tensor_alloc(sizeof(int32_t));
    if (!id_dev) return 0;

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (vocab_size + block_dim - 1) / block_dim;
    if (grid_dim > 65535) {
        ds4_gpu_tensor_free(id_dev);
        return 0;
    }

    dspark_markov_step_kernel<<<grid_dim, block_dim>>>(
        (float *)refined_logits->ptr,
        (int32_t *)id_dev->ptr,
        (const float *)base_logits->ptr,
        (const float *)markov_w1->ptr,
        (const float *)markov_w2->ptr,
        prev_token, vocab_size, embed_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ds4_gpu_tensor_free(id_dev);
        return 0;
    }

    if (!ds4_gpu_tensor_read(id_dev, 0, refined_id_dst, sizeof(int32_t))) {
        ds4_gpu_tensor_free(id_dev);
        return 0;
    }

    ds4_gpu_tensor_free(id_dev);
    return 1;
}


__global__ static void dspark_confidence_score_kernel(
        float *scores,
        const float *hidden,
        const float *markov_embed,
        const float *proj_weight,
        uint32_t n_positions,
        uint32_t hidden_dim,
        uint32_t embed_dim) {
    uint32_t p = blockIdx.x;
    if (p >= n_positions) return;

    const float *hp = hidden + (uint64_t)p * hidden_dim;
    const float *mp = markov_embed + (uint64_t)p * embed_dim;
    float dot = 0.0f;

    for (uint32_t i = threadIdx.x; i < hidden_dim; i += blockDim.x)
        dot += hp[i] * proj_weight[i];
    for (uint32_t i = threadIdx.x; i < embed_dim; i += blockDim.x)
        dot += mp[i] * proj_weight[hidden_dim + i];

    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        scores[p] = 1.0f / (1.0f + expf(-partial[0]));
}


extern "C" int ds4_gpu_dspark_confidence_score(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *markov_embed,
        const ds4_gpu_tensor *proj_weight,
        uint32_t n_positions,
        uint32_t hidden_dim,
        uint32_t embed_dim) {
    if (!scores || !hidden || !markov_embed || !proj_weight) return 0;
    if (n_positions == 0 || hidden_dim == 0 || embed_dim == 0) return 0;
    if (scores->bytes < (uint64_t)n_positions * sizeof(float)) return 0;
    if (hidden->bytes < (uint64_t)n_positions * hidden_dim * sizeof(float)) return 0;
    if (markov_embed->bytes < (uint64_t)n_positions * embed_dim * sizeof(float)) return 0;
    if (proj_weight->bytes < (uint64_t)(hidden_dim + embed_dim) * sizeof(float)) return 0;

    dspark_confidence_score_kernel<<<n_positions, 256>>>(
        (float *)scores->ptr,
        (const float *)hidden->ptr,
        (const float *)markov_embed->ptr,
        (const float *)proj_weight->ptr,
        n_positions, hidden_dim, embed_dim);

    return cuda_ok(cudaGetLastError(), "dspark confidence score");
}
