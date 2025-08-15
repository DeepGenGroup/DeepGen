#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils.h"
#include <cassert>

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr, const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = __ldg(cos_ptr + x_index);
    sin = __ldg(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = __ldg(cos_ptr + x_index / 2);
    sin = __ldg(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(const int32_t* __restrict__ positions,       // [batch_size, seq_len] or [num_tokens]
                                        scalar_t* __restrict__ query,                // [batch_size, seq_len, num_heads, head_size] or [num_tokens, num_heads, head_size]
                                        scalar_t* __restrict__ key,                  // [batch_size, seq_len, num_kv_heads, head_size] or [num_tokens, num_kv_heads, head_size]
                                        const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
                                        const int rot_dim, const int64_t query_stride, const int64_t key_stride, const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
}


template <typename scalar_t, int HEAD_DIM>
__global__ void rotary_embedding_online_kernel(
  const int32_t* __restrict__ positions,
  scalar_t* __restrict__ query,
  scalar_t* __restrict__ key,
  const float rope_theta,
  const int64_t query_stride,
  const int64_t key_stride,
  const int head_num,
  const int kv_head_num
) { 

  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];

  constexpr int EMBED_DIM = HEAD_DIM / 2;
  // precompute
  __shared__ float freq_cos[EMBED_DIM];
  __shared__ float freq_sin[EMBED_DIM];
  for (int i = threadIdx.x; i < EMBED_DIM; i += blockDim.x) {
    float freq = float(pos) / __powf(rope_theta, float(2 * i) / float(HEAD_DIM));
    __sincosf(freq, &freq_sin[i], &freq_cos[i]);
  }
  __syncthreads();

  const int nq = head_num * EMBED_DIM;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / EMBED_DIM;
    const int64_t token_head = token_idx * query_stride + head_idx * HEAD_DIM;
    int x_index = i % EMBED_DIM;
    int y_index = x_index + EMBED_DIM;
    const scalar_t x = query[token_head + x_index];
    const scalar_t y = query[token_head + y_index];
    query[token_head + x_index] = x * freq_cos[x_index] - y * freq_sin[x_index];
    query[token_head + y_index] = y * freq_cos[x_index] + x * freq_sin[x_index];
  }

  const int nk = kv_head_num * EMBED_DIM;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / EMBED_DIM;
    const int64_t token_head = token_idx * key_stride + head_idx * HEAD_DIM;
    const int rot_offset = i % EMBED_DIM;
    int x_index = i % EMBED_DIM;
    int y_index = x_index + EMBED_DIM;
    const scalar_t x = key[token_head + x_index];
    const scalar_t y = key[token_head + y_index];
    key[token_head + x_index] = x * freq_cos[x_index] - y * freq_sin[x_index];
    key[token_head + y_index] = y * freq_cos[x_index] + x * freq_sin[x_index];
  }
}


template <typename scalar_t, int HEAD_DIM>
__global__ void rotary_embedding_single_online_kernel(
  const int32_t* __restrict__ positions,
  scalar_t* __restrict__ arr,
  const float rope_theta,
  const int64_t stride,
  const int head_num
) { 

  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];

  constexpr int EMBED_DIM = HEAD_DIM / 2;
  // precompute
  __shared__ float freq_cos[EMBED_DIM];
  __shared__ float freq_sin[EMBED_DIM];
  for (int i = threadIdx.x; i < EMBED_DIM; i += blockDim.x) {
    float freq = float(pos) / __powf(rope_theta, float(2 * i) / float(HEAD_DIM));
    __sincosf(freq, &freq_sin[i], &freq_cos[i]);
  }
  __syncthreads();

  const int n = head_num * EMBED_DIM;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int head_idx = i / EMBED_DIM;
    const int64_t token_head = token_idx * stride + head_idx * HEAD_DIM;
    int x_index = i % EMBED_DIM;
    int y_index = x_index + EMBED_DIM;
    const scalar_t x = arr[token_head + x_index];
    const scalar_t y = arr[token_head + y_index];
    arr[token_head + x_index] = x * freq_cos[x_index] - y * freq_sin[x_index];
    arr[token_head + y_index] = y * freq_cos[x_index] + x * freq_sin[x_index];
  }
}


void rotary_embedding(torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
                      torch::Tensor& query,      // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
                      torch::Tensor& key,        // [batch_size, seq_len, num_kv_heads * head_size] or [num_tokens, num_kv_heads * head_size]
                      int head_size,
                      torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
                      bool is_neox) {
  int64_t num_tokens = query.numel() / query.size(-1);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding", [&] {
    if (is_neox) {
      rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(positions.data_ptr<int32_t>(), query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                                                                          rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
    } else {
      rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(positions.data_ptr<int32_t>(), query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(), cos_sin_cache.data_ptr<scalar_t>(),
                                                                           rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
    }
  });
}


void rotary_embedding_online(torch::Tensor& positions,  // [num_tokens]
                             torch::Tensor& query,      // [num_tokens, num_heads * head_size]
                             torch::Tensor& key,        // [num_tokens, num_kv_heads * head_size]
                             int head_dim,
                             float rope_theta
) {
  int64_t token_num = query.numel() / query.size(-1);
  int head_num = query.size(-1) / head_dim;
  int kv_head_num = key.size(-1) / head_dim;
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);

  dim3 grid(token_num);
  dim3 block(std::min(head_num * head_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  assert(head_dim == 128);
  DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding_online", [&] {
    rotary_embedding_online_kernel<scalar_t, 128><<<grid, block, 0, stream>>>(positions.data_ptr<int32_t>(), query.data_ptr<scalar_t>(), key.data_ptr<scalar_t>(), rope_theta, query_stride, key_stride, head_num, kv_head_num);
  });
}

void rotary_embedding_single_online(torch::Tensor& positions,  // [num_tokens]
                                    torch::Tensor& arr,      // [num_tokens, num_heads * head_size]
                                    int head_dim,
                                    float rope_theta
) {
  int64_t token_num = arr.numel() / arr.size(-1);
  int head_num = arr.size(-1) / head_dim;
  int64_t stride = arr.stride(-2);

  dim3 grid(token_num);
  dim3 block(std::min(head_num * head_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  assert(head_dim == 128);
  DISPATCH_FLOATING_TYPES(arr.scalar_type(), "rotary_embedding_single_online", [&] {
    rotary_embedding_single_online_kernel<scalar_t, 128><<<grid, block, 0, stream>>>(positions.data_ptr<int32_t>(), arr.data_ptr<scalar_t>(), rope_theta, stride, head_num);
  });
}

// TODO: remove redundant code