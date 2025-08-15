#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "utils.h"

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename scalar_t>
__global__ void silu_and_mul_kernel(scalar_t* __restrict__ out,          // [..., d]
                                    const scalar_t* __restrict__ input,  // [..., 2, d]
                                    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_kernel", [&] { silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d); });
}
