#include "common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__global__ void forward_kernel(
    const float* queries,
    const float* query_coordinates,
    const float* keys,
    const float* values,
    const float* opacity,
    const float* means,
    const float* stds,
    const float* rho,
    const float* routing_bias,
    const int64_t* candidate_indices,
    const bool* valid,
    float* output,
    int B, int Q, int N, int D, int C, int K) {
  const int block = blockIdx.x;
  if (block >= B * Q) return;
  const int b = block / Q;
  const int q = block - b * Q;
  extern __shared__ float weights[];

  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (int k = 0; k < K; ++k) {
      const float logit = msi_compute_logit(
          queries, query_coordinates, keys, opacity, means, stds, rho,
          routing_bias, candidate_indices, valid, b, q, k, Q, N, D, K);
      weights[k] = logit;
      maximum = fmaxf(maximum, logit);
    }
    float denominator = 0.0f;
    for (int k = 0; k < K; ++k) {
      const float weight = isfinite(weights[k]) ? expf(weights[k] - maximum) : 0.0f;
      weights[k] = weight;
      denominator += weight;
    }
    denominator = fmaxf(denominator, 1e-20f);
    for (int k = 0; k < K; ++k) weights[k] /= denominator;
  }
  __syncthreads();

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      if (!valid[q * K + k]) continue;
      const int n = static_cast<int>(candidate_indices[q * K + k]);
      sum += weights[k] * values[(b * N + n) * C + c];
    }
    output[(b * Q + q) * C + c] = sum;
  }
}

}  // namespace

torch::Tensor msi_renderer_forward_cuda(
    torch::Tensor queries,
    torch::Tensor query_coordinates,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor opacity,
    torch::Tensor means,
    torch::Tensor stds,
    torch::Tensor rho,
    torch::Tensor routing_bias,
    torch::Tensor candidate_indices,
    torch::Tensor valid) {
  const c10::cuda::CUDAGuard device_guard(queries.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int B = queries.size(0);
  const int Q = queries.size(1);
  const int D = queries.size(2);
  const int N = keys.size(1);
  const int C = values.size(2);
  const int K = candidate_indices.size(1);
  auto output = torch::zeros({B, Q, C}, values.options());
  constexpr int threads = 128;
  forward_kernel<<<B * Q, threads, K * sizeof(float), stream>>>(
      queries.data_ptr<float>(), query_coordinates.data_ptr<float>(),
      keys.data_ptr<float>(), values.data_ptr<float>(), opacity.data_ptr<float>(),
      means.data_ptr<float>(), stds.data_ptr<float>(), rho.data_ptr<float>(),
      routing_bias.data_ptr<float>(), candidate_indices.data_ptr<int64_t>(),
      valid.data_ptr<bool>(), output.data_ptr<float>(), B, Q, N, D, C, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
