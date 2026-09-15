#include <torch/extension.h>
#include <vector>

torch::Tensor msi_renderer_forward_cuda(
    torch::Tensor queries,
    torch::Tensor query_coordinates,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor opacity,
    torch::Tensor mu,
    torch::Tensor std,
    torch::Tensor rho,
    torch::Tensor routing_bias,
    torch::Tensor candidate_indices,
    torch::Tensor valid);

std::vector<torch::Tensor> msi_renderer_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor queries,
    torch::Tensor query_coordinates,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor opacity,
    torch::Tensor mu,
    torch::Tensor std,
    torch::Tensor rho,
    torch::Tensor routing_bias,
    torch::Tensor candidate_indices,
    torch::Tensor valid);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

void check_common(
    const torch::Tensor& queries,
    const torch::Tensor& query_coordinates,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& opacity,
    const torch::Tensor& mu,
    const torch::Tensor& std,
    const torch::Tensor& rho,
    const torch::Tensor& routing_bias,
    const torch::Tensor& candidate_indices,
    const torch::Tensor& valid) {
  CHECK_CUDA(queries); CHECK_CUDA(query_coordinates); CHECK_CUDA(keys);
  CHECK_CUDA(values); CHECK_CUDA(opacity); CHECK_CUDA(mu); CHECK_CUDA(std);
  CHECK_CUDA(rho); CHECK_CUDA(routing_bias); CHECK_CUDA(candidate_indices);
  CHECK_CUDA(valid);
  CHECK_CONTIGUOUS(queries); CHECK_CONTIGUOUS(query_coordinates);
  CHECK_CONTIGUOUS(keys); CHECK_CONTIGUOUS(values); CHECK_CONTIGUOUS(opacity);
  CHECK_CONTIGUOUS(mu); CHECK_CONTIGUOUS(std); CHECK_CONTIGUOUS(rho);
  CHECK_CONTIGUOUS(routing_bias); CHECK_CONTIGUOUS(candidate_indices);
  CHECK_CONTIGUOUS(valid);
  CHECK_FLOAT(queries); CHECK_FLOAT(query_coordinates); CHECK_FLOAT(keys);
  CHECK_FLOAT(values); CHECK_FLOAT(opacity); CHECK_FLOAT(mu); CHECK_FLOAT(std);
  CHECK_FLOAT(rho); CHECK_FLOAT(routing_bias);
  TORCH_CHECK(candidate_indices.scalar_type() == torch::kInt64,
              "candidate_indices must be int64");
  TORCH_CHECK(valid.scalar_type() == torch::kBool, "valid must be bool");
  TORCH_CHECK(queries.dim() == 3 && keys.dim() == 3 && values.dim() == 3,
              "queries/keys/values must be rank 3");
  const auto batch = queries.size(0);
  const auto query_count = queries.size(1);
  const auto routing_dim = queries.size(2);
  const auto gaussian_count = keys.size(1);
  TORCH_CHECK(keys.size(0) == batch && keys.size(2) == routing_dim,
              "keys must be [B,N,D] with the same B,D as queries");
  TORCH_CHECK(values.size(0) == batch && values.size(1) == gaussian_count,
              "values must be [B,N,C] with the same B,N as keys");
  TORCH_CHECK(query_coordinates.dim() == 2 && query_coordinates.size(1) == 2,
              "query_coordinates must be [Q,2]");
  TORCH_CHECK(query_coordinates.size(0) == query_count,
              "query_coordinates Q must match queries");
  TORCH_CHECK(candidate_indices.dim() == 2 && valid.sizes() == candidate_indices.sizes(),
              "candidate_indices and valid must be [Q,K]");
  TORCH_CHECK(candidate_indices.size(0) == query_count,
              "candidate_indices Q must match queries");
  TORCH_CHECK(opacity.sizes() == torch::IntArrayRef({batch, gaussian_count, 1}),
              "opacity must be [B,N,1]");
  TORCH_CHECK(mu.sizes() == torch::IntArrayRef({batch, gaussian_count, 2}),
              "means must be [B,N,2]");
  TORCH_CHECK(std.sizes() == torch::IntArrayRef({batch, gaussian_count, 2}),
              "stds must be [B,N,2]");
  TORCH_CHECK(rho.sizes() == torch::IntArrayRef({batch, gaussian_count, 1}),
              "rho must be [B,N,1]");
  TORCH_CHECK(routing_bias.sizes() ==
                  torch::IntArrayRef({batch, query_count, candidate_indices.size(1)}),
              "routing_bias must be [B,Q,K]");
}

torch::Tensor forward_checked(
    torch::Tensor queries,
    torch::Tensor query_coordinates,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor opacity,
    torch::Tensor mu,
    torch::Tensor std,
    torch::Tensor rho,
    torch::Tensor routing_bias,
    torch::Tensor candidate_indices,
    torch::Tensor valid) {
  check_common(queries, query_coordinates, keys, values, opacity, mu, std, rho,
               routing_bias, candidate_indices, valid);
  return msi_renderer_forward_cuda(queries, query_coordinates, keys, values,
                                   opacity, mu, std, rho, routing_bias,
                                   candidate_indices, valid);
}

std::vector<torch::Tensor> backward_checked(
    torch::Tensor grad_output,
    torch::Tensor queries,
    torch::Tensor query_coordinates,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor opacity,
    torch::Tensor mu,
    torch::Tensor std,
    torch::Tensor rho,
    torch::Tensor routing_bias,
    torch::Tensor candidate_indices,
    torch::Tensor valid) {
  check_common(queries, query_coordinates, keys, values, opacity, mu, std, rho,
               routing_bias, candidate_indices, valid);
  CHECK_CUDA(grad_output); CHECK_CONTIGUOUS(grad_output); CHECK_FLOAT(grad_output);
  return msi_renderer_backward_cuda(grad_output, queries, query_coordinates,
                                    keys, values, opacity, mu, std, rho,
                                    routing_bias, candidate_indices, valid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_checked, "MSI-conditioned renderer forward (CUDA)");
  m.def("backward", &backward_checked, "MSI-conditioned renderer backward (CUDA)");
}
