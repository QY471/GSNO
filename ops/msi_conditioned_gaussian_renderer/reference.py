"""PyTorch references for MSI-conditioned query-centric Gaussian rendering."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def make_query_coordinates(
    out_size: Tuple[int, int], lr_size: Tuple[int, int], device, dtype
) -> torch.Tensor:
    """Return HR pixel centers represented in LR-cell index coordinates."""
    H, W = out_size
    h, w = lr_size
    x = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (w / W) - 0.5
    y = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (h / H) - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def make_candidate_indices(
    query_coordinates: torch.Tensor,
    lr_size: Tuple[int, int],
    radius: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed LR-anchor window around the nearest LR anchor for every query."""
    h, w = lr_size
    offsets = torch.arange(-radius, radius + 1, device=query_coordinates.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    cx = query_coordinates[:, 0].round().long().unsqueeze(1) + ox.reshape(1, -1)
    cy = query_coordinates[:, 1].round().long().unsqueeze(1) + oy.reshape(1, -1)
    valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    indices = cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1)
    return indices.contiguous(), valid.contiguous()


def gather_candidates(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    b = tensor.shape[0]
    batch = torch.arange(b, device=tensor.device).view(b, 1, 1)
    return tensor[batch, indices.unsqueeze(0)]


def gaussian_logit(
    query_coordinates: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    rho: torch.Tensor,
    opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gaussian log density up to constants that cancel in candidate softmax."""
    relative = query_coordinates.view(1, query_coordinates.shape[0], 1, 2) - mu
    dx = relative[..., 0] / std[..., 0].clamp_min(1e-4)
    dy = relative[..., 1] / std[..., 1].clamp_min(1e-4)
    r = rho.squeeze(-1)
    beta = 1.0 - r.square() + 1e-6
    mahalanobis = (dx.square() + dy.square() - 2.0 * r * dx * dy) / beta
    return -0.5 * mahalanobis + opacity.squeeze(-1).clamp_min(1e-8).log(), relative


def all_gaussian_full_reference(
    queries: torch.Tensor,
    query_coordinates: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    opacity: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    rho: torch.Tensor,
    routing_bias: torch.Tensor,
    return_weights: bool = False,
):
    """Small-size ground truth that routes over all N Gaussians for every query.

    Shapes: queries [B,Q,D], coordinates [Q,2], keys [B,N,D], values
    [B,N,C], geometry [B,N,*], routing_bias [B,Q,N].
    """
    b, q, d = queries.shape
    n = keys.shape[1]
    if routing_bias.shape != (b, q, n):
        raise ValueError(f"routing_bias must be {(b, q, n)}, got {tuple(routing_bias.shape)}")
    mu_all = mu.unsqueeze(1).expand(-1, q, -1, -1)
    std_all = std.unsqueeze(1).expand(-1, q, -1, -1)
    rho_all = rho.unsqueeze(1).expand(-1, q, -1, -1)
    opacity_all = opacity.unsqueeze(1).expand(-1, q, -1, -1)
    gaussian, _ = gaussian_logit(query_coordinates, mu_all, std_all, rho_all, opacity_all)
    query_key = torch.einsum("bqd,bnd->bqn", queries, keys) / math.sqrt(d)
    weights = torch.softmax((gaussian + query_key + routing_bias).float(), dim=-1).to(values.dtype)
    output = torch.einsum("bqn,bnc->bqc", weights, values)
    return (output, weights) if return_weights else output

def candidate_window_reference(
    queries: torch.Tensor,
    query_coordinates: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    opacity: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    rho: torch.Tensor,
    routing_bias: torch.Tensor,
    candidate_indices: torch.Tensor,
    valid: torch.Tensor,
    return_weights: bool = False,
):
    """Reference for one query chunk and a fixed candidate-index window."""
    b, q, d = queries.shape
    k = candidate_indices.shape[1]
    if candidate_indices.shape[0] != q or valid.shape != candidate_indices.shape:
        raise ValueError("candidate index/valid shapes do not match query count")
    if routing_bias.shape != (b, q, k):
        raise ValueError(f"routing_bias must be {(b, q, k)}, got {tuple(routing_bias.shape)}")
    key_candidates = gather_candidates(keys, candidate_indices)
    value_candidates = gather_candidates(values, candidate_indices)
    mu_candidates = gather_candidates(mu, candidate_indices)
    std_candidates = gather_candidates(std, candidate_indices)
    rho_candidates = gather_candidates(rho, candidate_indices)
    opacity_candidates = gather_candidates(opacity, candidate_indices)
    gaussian, _ = gaussian_logit(
        query_coordinates,
        mu_candidates,
        std_candidates,
        rho_candidates,
        opacity_candidates,
    )
    query_key = (queries.unsqueeze(2) * key_candidates).sum(-1) / math.sqrt(d)
    logits = (gaussian + query_key + routing_bias).masked_fill(
        ~valid.unsqueeze(0), torch.finfo(gaussian.dtype).min
    )
    weights = torch.softmax(logits.float(), dim=-1).to(values.dtype)
    output = (weights.unsqueeze(-1) * value_candidates).sum(dim=2)
    return (output, weights) if return_weights else output
