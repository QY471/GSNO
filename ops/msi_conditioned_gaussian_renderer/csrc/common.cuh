#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__device__ inline float msi_compute_logit(
    const float* queries,
    const float* query_coordinates,
    const float* keys,
    const float* opacity,
    const float* means,
    const float* stds,
    const float* rho,
    const float* routing_bias,
    const int64_t* candidate_indices,
    const bool* valid,
    int b, int q, int k,
    int Q, int N, int D, int K) {
  const int qk = q * K + k;
  if (!valid[qk]) return -INFINITY;
  const int n = static_cast<int>(candidate_indices[qk]);
  const int query_base = (b * Q + q) * D;
  const int key_base = (b * N + n) * D;
  float dot = 0.0f;
  for (int d = 0; d < D; ++d) {
    dot += queries[query_base + d] * keys[key_base + d];
  }
  dot *= rsqrtf(static_cast<float>(D));

  const float qx = query_coordinates[q * 2];
  const float qy = query_coordinates[q * 2 + 1];
  const int geom2 = (b * N + n) * 2;
  const int geom1 = b * N + n;
  const float sx = fmaxf(stds[geom2], 1e-4f);
  const float sy = fmaxf(stds[geom2 + 1], 1e-4f);
  const float r = rho[geom1];
  const float dx = (qx - means[geom2]) / sx;
  const float dy = (qy - means[geom2 + 1]) / sy;
  const float beta = 1.0f - r * r + 1e-6f;
  const float numerator = dx * dx + dy * dy - 2.0f * r * dx * dy;
  const float gaussian =
      -0.5f * numerator / beta + logf(fmaxf(opacity[geom1], 1e-8f));
  return gaussian + dot + routing_bias[(b * Q + q) * K + k];
}
