"""Autograd wrapper for the independent query-centric CUDA extension."""

from __future__ import annotations

import importlib
from typing import Optional

import torch

from .reference import make_candidate_indices, make_query_coordinates


_EXTENSION: Optional[object] = None


def load_extension(required: bool = True):
    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = importlib.import_module("msi_conditioned_gaussian_renderer_cuda")
        except ImportError:
            if required:
                raise RuntimeError(
                    "Independent E2 CUDA extension is not built. Run "
                    "`python ops/msi_conditioned_gaussian_renderer/setup.py build_ext --inplace`."
                )
            return None
    return _EXTENSION


class _MSIConditionedGaussianRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries,
        query_coordinates,
        keys,
        values,
        opacity,
        mu,
        std,
        rho,
        routing_bias,
        candidate_indices,
        valid,
    ):
        extension = load_extension(required=True)
        tensors = (
            queries.float().contiguous(),
            query_coordinates.float().contiguous(),
            keys.float().contiguous(),
            values.float().contiguous(),
            opacity.float().contiguous(),
            mu.float().contiguous(),
            std.float().contiguous(),
            rho.float().contiguous(),
            routing_bias.float().contiguous(),
            candidate_indices.long().contiguous(),
            valid.bool().contiguous(),
        )
        output = extension.forward(*tensors)
        ctx.save_for_backward(*tensors)
        ctx.input_dtype = values.dtype
        return output.to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        extension = load_extension(required=True)
        tensors = ctx.saved_tensors
        gradients = extension.backward(grad_output.float().contiguous(), *tensors)
        return (
            gradients[0],  # queries
            None,          # fixed query coordinates
            gradients[1],  # keys
            gradients[2],  # values
            gradients[3],  # opacity
            gradients[4],  # mu
            gradients[5],  # std
            gradients[6],  # rho
            gradients[7],  # routing bias (backpropagates through bias networks)
            None,
            None,
        )


def cuda_candidate_render(
    queries,
    query_coordinates,
    keys,
    values,
    opacity,
    mu,
    std,
    rho,
    routing_bias,
    candidate_indices,
    valid,
):
    return _MSIConditionedGaussianRender.apply(
        queries,
        query_coordinates,
        keys,
        values,
        opacity,
        mu,
        std,
        rho,
        routing_bias,
        candidate_indices,
        valid,
    )


def cuda_msi_conditioned_render(
    queries,
    keys,
    values,
    means,
    stds,
    rho,
    opacity,
    lr_size,
    out_size,
    candidate_radius,
    routing_bias=None,
):
    """Render a complete HR latent map with the independent E2 CUDA backend.

    This public grid-level API accepts the contract shapes and returns
    ``[B,C,H,W]``. ``routing_bias`` is optional because query-key routing and
    Gaussian geometry are always computed by the CUDA kernel; the V2 model
    supplies its learned relative-position and geometry bias as an extra term.
    """
    h, w = (int(lr_size[0]), int(lr_size[1]))
    H, W = (int(out_size[0]), int(out_size[1]))
    if queries.ndim != 3 or queries.shape[1] != H * W:
        raise ValueError(f"queries must be [B,{H * W},D]")
    if keys.ndim != 3 or keys.shape[1] != h * w:
        raise ValueError(f"keys must be [B,{h * w},D]")
    if values.ndim != 3 or values.shape[:2] != keys.shape[:2]:
        raise ValueError("values must be [B,N,C] with the same B,N as keys")
    if means.shape != (queries.shape[0], h * w, 2):
        raise ValueError(f"means must be [B,{h * w},2]")
    if stds.shape != means.shape:
        raise ValueError("stds must have the same [B,N,2] shape as means")
    rho = rho.unsqueeze(-1) if rho.ndim == 2 else rho
    opacity = opacity.unsqueeze(-1) if opacity.ndim == 2 else opacity
    coordinates = make_query_coordinates(
        (H, W), (h, w), queries.device, torch.float32
    )
    candidate_indices, valid = make_candidate_indices(
        coordinates, (h, w), int(candidate_radius)
    )
    expected_bias_shape = (
        queries.shape[0], H * W, candidate_indices.shape[1]
    )
    if routing_bias is None:
        routing_bias = queries.new_zeros(expected_bias_shape)
    elif routing_bias.shape != expected_bias_shape:
        raise ValueError(f"routing_bias must be {expected_bias_shape}")
    rendered = cuda_candidate_render(
        queries,
        coordinates,
        keys,
        values,
        opacity,
        means,
        stds,
        rho,
        routing_bias,
        candidate_indices,
        valid,
    )
    return rendered.transpose(1, 2).reshape(queries.shape[0], values.shape[2], H, W)
