#include "common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

__global__ void backward_kernel(
    const float* grad_output,
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
    float* grad_queries,
    float* grad_keys,
    float* grad_values,
    float* grad_opacity,
    float* grad_means,
    float* grad_stds,
    float* grad_rho,
    float* grad_routing_bias,
    int B, int Q, int N, int D, int C, int K) {
  const int block = blockIdx.x;
  if (block >= B * Q) return;
  const int b = block / Q;
  const int q = block - b * Q;
  extern __shared__ float shared[];
  float* weights = shared;
  float* dlogits = shared + K;

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

    float weighted_dvalue = 0.0f;
    for (int k = 0; k < K; ++k) {
      if (!valid[q * K + k]) {
        dlogits[k] = 0.0f;
        continue;
      }
      const int n = static_cast<int>(candidate_indices[q * K + k]);
      float dweight = 0.0f;
      for (int c = 0; c < C; ++c) {
        dweight += grad_output[(b * Q + q) * C + c]
            * values[(b * N + n) * C + c];
      }
      dlogits[k] = dweight;
      weighted_dvalue += weights[k] * dweight;
    }
    for (int k = 0; k < K; ++k) {
      dlogits[k] = weights[k] * (dlogits[k] - weighted_dvalue);
      grad_routing_bias[(b * Q + q) * K + k] = dlogits[k];
    }

    const float qx = query_coordinates[q * 2];
    const float qy = query_coordinates[q * 2 + 1];
    for (int k = 0; k < K; ++k) {
      if (!valid[q * K + k]) continue;
      const int n = static_cast<int>(candidate_indices[q * K + k]);
      const int geom2 = (b * N + n) * 2;
      const int geom1 = b * N + n;
      const float sx = fmaxf(stds[geom2], 1e-4f);
      const float sy = fmaxf(stds[geom2 + 1], 1e-4f);
      const float r = rho[geom1];
      const float dx = (qx - means[geom2]) / sx;
      const float dy = (qy - means[geom2 + 1]) / sy;
      const float beta = 1.0f - r * r + 1e-6f;
      const float numerator = dx * dx + dy * dy - 2.0f * r * dx * dy;
      const float dlogit_ddx = -(dx - r * dy) / beta;
      const float dlogit_ddy = -(dy - r * dx) / beta;
      const float dlogit_dr = dx * dy / beta - r * numerator / (beta * beta);
      const float gradient = dlogits[k];
      atomicAdd(&grad_opacity[geom1],
                gradient / fmaxf(opacity[geom1], 1e-8f));
      atomicAdd(&grad_means[geom2], gradient * (-dlogit_ddx / sx));
      atomicAdd(&grad_means[geom2 + 1], gradient * (-dlogit_ddy / sy));
      atomicAdd(&grad_stds[geom2], gradient * dlogit_ddx * (-dx / sx));
      atomicAdd(&grad_stds[geom2 + 1], gradient * dlogit_ddy * (-dy / sy));
      atomicAdd(&grad_rho[geom1], gradient * dlogit_dr);
    }
  }
  __syncthreads();

  const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float query_gradient = 0.0f;
    for (int k = 0; k < K; ++k) {
      if (!valid[q * K + k]) continue;
      const int n = static_cast<int>(candidate_indices[q * K + k]);
      query_gradient += dlogits[k] * keys[(b * N + n) * D + d] * inv_sqrt_d;
      atomicAdd(&grad_keys[(b * N + n) * D + d],
                dlogits[k] * queries[(b * Q + q) * D + d] * inv_sqrt_d);
    }
    grad_queries[(b * Q + q) * D + d] = query_gradient;
  }

  for (int linear = threadIdx.x; linear < K * C; linear += blockDim.x) {
    const int k = linear / C;
    const int c = linear - k * C;
    if (!valid[q * K + k]) continue;
    const int n = static_cast<int>(candidate_indices[q * K + k]);
    atomicAdd(&grad_values[(b * N + n) * C + c],
              weights[k] * grad_output[(b * Q + q) * C + c]);
  }
}

}  // namespace

std::vector<torch::Tensor> msi_renderer_backward_cuda(
    torch::Tensor grad_output,
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
  auto grad_queries = torch::zeros_like(queries);
  auto grad_keys = torch::zeros_like(keys);
  auto grad_values = torch::zeros_like(values);
  auto grad_opacity = torch::zeros_like(opacity);
  auto grad_means = torch::zeros_like(means);
  auto grad_stds = torch::zeros_like(stds);
  auto grad_rho = torch::zeros_like(rho);
  auto grad_routing_bias = torch::zeros_like(routing_bias);
  constexpr int threads = 128;
  backward_kernel<<<B * Q, threads, 2 * K * sizeof(float), stream>>>(
      grad_output.data_ptr<float>(), queries.data_ptr<float>(),
      query_coordinates.data_ptr<float>(), keys.data_ptr<float>(),
      values.data_ptr<float>(), opacity.data_ptr<float>(), means.data_ptr<float>(),
      stds.data_ptr<float>(), rho.data_ptr<float>(), routing_bias.data_ptr<float>(),
      candidate_indices.data_ptr<int64_t>(), valid.data_ptr<bool>(),
      grad_queries.data_ptr<float>(), grad_keys.data_ptr<float>(),
      grad_values.data_ptr<float>(), grad_opacity.data_ptr<float>(),
      grad_means.data_ptr<float>(), grad_stds.data_ptr<float>(),
      grad_rho.data_ptr<float>(), grad_routing_bias.data_ptr<float>(),
      B, Q, N, D, C, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_queries, grad_keys, grad_values, grad_opacity,
          grad_means, grad_stds, grad_rho, grad_routing_bias};
}
