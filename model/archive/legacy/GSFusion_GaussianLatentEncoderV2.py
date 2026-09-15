"""Gaussian Latent Encoder V2.2.

This module implements the four frozen experiment configurations from the
V2.2 specification. E1 uses the legacy density-normalized Gaussian renderer.
E2 provides both a chunked PyTorch reference and an independent query-centric
CUDA operator. The bundled legacy rasterizer is never modified or used as E2.

ADCI is reimplemented exactly from the established project implementation in
``model/GSFusion_GSNO.py``. ADCI is not claimed as a contribution of this work.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.msi_conditioned_gaussian_renderer import (
    candidate_window_reference,
    cuda_candidate_render,
    gather_candidates,
    gaussian_logit as reference_gaussian_logit,
    load_extension as load_msi_renderer_extension,
    make_candidate_indices,
    make_query_coordinates,
)


def initialize_weights_xavier(module: nn.Module) -> None:
    """Project-wide generic initialization applied before V2 special resets."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _resolve_cuda_rasterizer():
    """Return the legacy CUDA rasterizer class, or ``None`` when unavailable."""
    try:
        from diff_srgaussian_rasterization import GaussianRasterizer

        return GaussianRasterizer
    except Exception:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        submodule = os.path.join(root, "submodules", "diff-srgaussian-rasterization")
        build = os.path.join(submodule, "build")
        if os.path.isdir(build):
            entries = sorted(
                os.path.join(build, item)
                for item in os.listdir(build)
                if item.startswith("lib.")
            )
            for entry in reversed(entries):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
        if submodule not in sys.path:
            sys.path.append(submodule)
        try:
            from diff_srgaussian_rasterization import GaussianRasterizer

            return GaussianRasterizer
        except Exception:
            return None


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class PointwiseGatedBlock(nn.Module):
    """Channel-only residual block; it performs no spatial mixing."""

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        hidden = dim * expansion
        self.norm = ChannelLayerNorm(dim)
        self.in_proj = nn.Conv2d(dim, hidden * 2, 1)
        self.out_proj = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        return x + self.out_proj(F.gelu(value) * torch.sigmoid(gate))


class LayerNorm(nn.Module):
    """LayerNorm used by the established ADCI implementation."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        out = (x - mean) / (std + self.eps)
        return self.weight * out + self.bias


class ADCI(nn.Module):
    """Established ADCI from ``model/GSFusion_GSNO.py``; not a contribution."""

    def __init__(self, in_channels: int, mlp_hidden_dim: int):
        super().__init__()
        self.qkv_conv = nn.Conv2d(
            in_channels, in_channels * 3, kernel_size=1, bias=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, in_channels),
        )
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        k_unfold = F.unfold(k, kernel_size=3, padding=1).view(b, c, 9, h, w)
        v_unfold = F.unfold(v, kernel_size=3, padding=1).view(b, c, 9, h, w)

        q_expanded = q.unsqueeze(2)
        q_minus_k = q_expanded - k_unfold
        q_minus_k = q_minus_k.permute(0, 3, 4, 2, 1).contiguous()
        mlp_output = self.mlp(q_minus_k)
        attention_scores = F.softmax(mlp_output, dim=-2)

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()
        weighted_v = torch.sum(neighbors_v * attention_scores, dim=3)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()
        return weighted_v + self.gate(x)


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, encoder_type: str):
        super().__init__()
        if encoder_type not in {"pointwise", "adci"}:
            raise ValueError(f"unknown encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.input_proj = nn.Conv2d(in_channels, dim, 1)
        if encoder_type == "pointwise":
            self.blocks = nn.Sequential(PointwiseGatedBlock(dim), PointwiseGatedBlock(dim))
        else:
            self.blocks = nn.Sequential(ADCI(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input_proj(x))


def _relative_offsets(dilations: Sequence[int], device, dtype) -> torch.Tensor:
    offsets = []
    for scale_index, dilation in enumerate(dilations):
        for oy in (-dilation, 0, dilation):
            for ox in (-dilation, 0, dilation):
                if scale_index > 0 and ox == 0 and oy == 0:
                    continue
                offsets.append((float(ox), float(oy)))
    return torch.tensor(offsets, device=device, dtype=dtype)


class GaussianLatentInteractionLayer(nn.Module):
    """Multi-head local token interaction in LR-cell coordinates."""

    def __init__(self, dim: int, num_heads: int, dilations: Sequence[int]):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dilations = tuple(int(item) for item in dilations)
        self.norm = ChannelLayerNorm(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.relpos_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, num_heads))
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.last_stats: Optional[Dict[str, float]] = None

    def _unfold_dilated(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        fields = []
        non_center = (0, 1, 2, 3, 5, 6, 7, 8)
        for scale_index, dilation in enumerate(self.dilations):
            field = F.unfold(x, 3, dilation=dilation, padding=dilation).view(
                b, c, 9, h * w
            )
            if scale_index > 0:
                field = field[:, :, non_center, :]
            fields.append(field)
        return torch.cat(fields, dim=2)

    def _valid_neighbor_mask(self, h: int, w: int, device) -> torch.Tensor:
        support = torch.ones(1, 1, h, w, device=device)
        fields = []
        non_center = (0, 1, 2, 3, 5, 6, 7, 8)
        for scale_index, dilation in enumerate(self.dilations):
            field = F.unfold(
                support, 3, dilation=dilation, padding=dilation
            ).view(1, 1, 9, h * w)
            if scale_index > 0:
                field = field[:, :, non_center, :]
            fields.append(field)
        valid = torch.cat(fields, dim=2).squeeze(1).transpose(1, 2).bool()
        return valid.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        neighbors = 9 + 8 * (len(self.dilations) - 1)
        q = q.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = self._unfold_dilated(k).view(
            b, self.num_heads, self.head_dim, neighbors, h * w
        ).permute(0, 1, 4, 3, 2)
        v = self._unfold_dilated(v).view(
            b, self.num_heads, self.head_dim, neighbors, h * w
        ).permute(0, 1, 4, 3, 2)

        score = (q.unsqueeze(3) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        offsets = _relative_offsets(self.dilations, x.device, x.dtype)
        rel_bias = self.relpos_mlp(offsets).transpose(0, 1).view(1, self.num_heads, 1, neighbors)
        valid = self._valid_neighbor_mask(h, w, x.device)
        logits = (score + rel_bias).masked_fill(~valid, float("-inf"))
        attention = F.softmax(logits, dim=-1)
        message = (attention.unsqueeze(-1) * v).sum(dim=3)
        message = message.permute(0, 1, 3, 2).reshape(b, c, h, w)
        projected = self.out_proj(message)
        out = x + self.gamma * projected

        with torch.no_grad():
            entropy = -(attention.float() * attention.float().clamp_min(1e-8).log()).sum(-1).mean()
            expanded_valid = valid.expand(b, self.num_heads, -1, -1)
            invalid_attention = attention.masked_select(~expanded_valid)
            valid_counts = valid.squeeze(0).squeeze(0).sum(-1)
            self.last_stats = {
                "interaction_gamma": float(self.gamma.detach()),
                "interaction_neighbor_count": neighbors,
                "interaction_valid_neighbors_min": int(valid_counts.min()),
                "interaction_valid_neighbors_max": int(valid_counts.max()),
                "interaction_invalid_attention_max": float(
                    invalid_attention.abs().max() if invalid_attention.numel() else 0.0
                ),
                "interaction_attention_sum_error_max": float(
                    (attention.float().sum(-1) - 1.0).abs().max()
                ),
                "interaction_message_input_ratio": float(
                    projected.detach().abs().mean() / (x.detach().abs().mean() + 1e-8)
                ),
                "interaction_attention_entropy": float(entropy),
            }
        return out


class GaussianLatentInteraction(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int = 2,
        num_heads: int = 4,
        dilations: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [GaussianLatentInteractionLayer(dim, num_heads, dilations) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def sample_msi_at_lr_centers(msi_feature: torch.Tensor, lr_size: Tuple[int, int]) -> torch.Tensor:
    """Take exactly one HR-MSI feature sample at each LR-cell center."""
    b, _, H, W = msi_feature.shape
    h, w = lr_size
    y = (torch.arange(h, device=msi_feature.device, dtype=msi_feature.dtype) + 0.5) * (H / h) - 0.5
    x = (torch.arange(w, device=msi_feature.device, dtype=msi_feature.dtype) + 0.5) * (W / w) - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([2.0 * (xx + 0.5) / W - 1.0, 2.0 * (yy + 0.5) / H - 1.0], dim=-1)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(
        msi_feature, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


class GaussianGeometryHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU(), nn.Conv2d(dim, 6, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, token_map: torch.Tensor, msi_center: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, _, h, w = token_map.shape
        raw = self.head(torch.cat([token_map, msi_center], dim=1))
        raw = raw.permute(0, 2, 3, 1).reshape(b, h * w, 6)
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        offset = 0.5 * torch.tanh(raw[..., 1:3])
        std = 0.125 + 0.875 * torch.sigmoid(raw[..., 3:5])
        rho = 0.999 * torch.tanh(raw[..., 5:6])
        yy, xx = torch.meshgrid(
            torch.arange(h, device=raw.device, dtype=raw.dtype),
            torch.arange(w, device=raw.device, dtype=raw.dtype),
            indexing="ij",
        )
        base = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        mu = base.unsqueeze(0) + offset
        return {"opacity": opacity, "offset": offset, "std": std, "rho": rho, "mu": mu}


def _query_coordinates(
    out_size: Tuple[int, int], lr_size: Tuple[int, int], device, dtype
) -> torch.Tensor:
    H, W = out_size
    h, w = lr_size
    x = (torch.arange(W, device=device, dtype=dtype) + 0.5) * (w / W) - 0.5
    y = (torch.arange(H, device=device, dtype=dtype) + 0.5) * (h / H) - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _candidate_indices(
    queries: torch.Tensor, lr_size: Tuple[int, int], radius: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = lr_size
    offsets = torch.arange(-radius, radius + 1, device=queries.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    cx = queries[:, 0].round().long().unsqueeze(1) + ox.reshape(1, -1)
    cy = queries[:, 1].round().long().unsqueeze(1) + oy.reshape(1, -1)
    valid = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
    indices = cy.clamp(0, h - 1) * w + cx.clamp(0, w - 1)
    return indices, valid


def _gather_tokens(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    b = tensor.shape[0]
    batch = torch.arange(b, device=tensor.device).view(b, 1, 1)
    return tensor[batch, indices.unsqueeze(0)]


def _gaussian_logits(
    queries: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    rho: torch.Tensor,
    opacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = queries.view(1, queries.shape[0], 1, 2) - mu
    dx = delta[..., 0] / std[..., 0].clamp_min(1e-4)
    dy = delta[..., 1] / std[..., 1].clamp_min(1e-4)
    inv = 1.0 / (1.0 - rho.squeeze(-1).square() + 1e-6)
    mahalanobis = inv * (dx.square() + dy.square() - 2.0 * rho.squeeze(-1) * dx * dy)
    logits = -0.5 * mahalanobis + opacity.squeeze(-1).clamp_min(1e-8).log()
    return logits, delta


class DensityNormalizedRenderer(nn.Module):
    """E1: legacy Gaussian density normalization without per-HR MSI routing."""

    def __init__(self, dim: int, candidate_radius: int = 5, chunk: int = 2048):
        super().__init__()
        self.dim = dim
        self.radius = int(candidate_radius)
        self.chunk = int(chunk)
        rasterizer = _resolve_cuda_rasterizer()
        self.cuda_rasterizer = rasterizer(dim + 1) if rasterizer is not None else None
        self.last_backend = "uninitialized"

    def _torch_render(
        self,
        values: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        lr_size: Tuple[int, int],
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        b, _, c = values.shape
        queries = _query_coordinates(out_size, lr_size, values.device, torch.float32)
        output = torch.empty(b, queries.shape[0], c, device=values.device, dtype=values.dtype)
        for start in range(0, queries.shape[0], self.chunk):
            query = queries[start : start + self.chunk]
            indices, valid = _candidate_indices(query, lr_size, self.radius)
            mu = _gather_tokens(geometry["mu"], indices)
            std = _gather_tokens(geometry["std"], indices)
            rho = _gather_tokens(geometry["rho"], indices)
            opacity = _gather_tokens(geometry["opacity"], indices)
            candidate_values = _gather_tokens(values, indices)
            logits, _ = _gaussian_logits(query, mu, std, rho, opacity)
            weights = logits.exp() * valid.to(logits.dtype).unsqueeze(0)
            output[:, start : start + query.shape[0]] = (
                weights.unsqueeze(-1) * candidate_values
            ).sum(dim=2) / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        H, W = out_size
        return output.permute(0, 2, 1).reshape(b, c, H, W)

    def forward(
        self,
        values: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        lr_size: Tuple[int, int],
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        if self.cuda_rasterizer is None or not values.is_cuda:
            self.last_backend = "torch_density_normalized_fallback"
            return self._torch_render(values, geometry, lr_size, out_size)

        h, w = lr_size
        H, W = out_size
        sx, sy = W / w, H / h
        scale = values.new_tensor([sx, sy])
        mu_px = (geometry["mu"] + 0.5) * scale - 0.5
        std_px = geometry["std"] * scale
        density = torch.ones_like(values[..., :1])
        packed = torch.cat([values, density], dim=-1)
        raster_ratio = min(1.0, max(float(self.radius) / h, float(self.radius) / w))
        rendered = self.cuda_rasterizer(
            geometry["opacity"].float(),
            mu_px.float(),
            std_px.float(),
            geometry["rho"].float(),
            packed.float(),
            H,
            W,
            1,
            raster_ratio,
            debug=False,
        ).permute(0, 3, 1, 2)
        numerator, denominator = rendered[:, : self.dim], rendered[:, self.dim :]
        self.last_backend = "legacy_cuda_density_normalized"
        return (numerator / denominator.clamp_min(1e-8)).to(values.dtype)


class MSIConditionedGaussianRenderer(nn.Module):
    """E2 routing with separable query-key, relative, and geometry terms."""

    def __init__(
        self,
        dim: int,
        routing_dim: int = 16,
        candidate_radius: int = 5,
        chunk: int = 1024,
        backend: str = "cuda",
        enable_diagnostics: bool = False,
    ):
        super().__init__()
        if backend not in {"torch_chunked", "cuda"}:
            raise ValueError(f"unknown E2 backend: {backend}")
        self.radius = int(candidate_radius)
        self.chunk = int(chunk)
        self.backend = backend
        self.enable_diagnostics = bool(enable_diagnostics)
        self.msi_query_proj = nn.Conv2d(dim, routing_dim, 1)
        self.token_key_proj = nn.Linear(dim, routing_dim)
        self.relative_position_bias = nn.Sequential(
            nn.Linear(2, routing_dim), nn.GELU(), nn.Linear(routing_dim, 1)
        )
        self.geometry_bias = nn.Sequential(
            nn.Linear(6, routing_dim), nn.GELU(), nn.Linear(routing_dim, 1)
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def forward(
        self,
        values: torch.Tensor,
        tokens: torch.Tensor,
        msi_hr: torch.Tensor,
        geometry: Dict[str, torch.Tensor],
        lr_size: Tuple[int, int],
        out_size: Tuple[int, int],
    ) -> torch.Tensor:
        b, _, c = values.shape
        H, W = out_size
        query_coordinates = make_query_coordinates(
            out_size, lr_size, values.device, torch.float32
        )
        msi_queries = self.msi_query_proj(msi_hr).flatten(2).transpose(1, 2)
        token_keys = self.token_key_proj(tokens)
        geom_vector = torch.cat(
            [geometry["opacity"], geometry["offset"], geometry["std"], geometry["rho"]], dim=-1
        )
        anchor_geometry_bias = self.geometry_bias(geom_vector)
        outputs = []
        gaussian_abs = values.new_zeros(())
        routing_abs = values.new_zeros(())
        entropy_sum = values.new_zeros(())
        max_weight_sum = values.new_zeros(())
        chunks = 0

        if self.backend == "cuda":
            load_msi_renderer_extension(required=True)

        for start in range(0, query_coordinates.shape[0], self.chunk):
            coordinates = query_coordinates[start : start + self.chunk]
            query_features = msi_queries[:, start : start + coordinates.shape[0]]
            indices, valid = make_candidate_indices(coordinates, lr_size, self.radius)
            candidate_mu = gather_candidates(geometry["mu"], indices)
            relative = coordinates.view(1, coordinates.shape[0], 1, 2) - candidate_mu
            routing_bias = self.relative_position_bias(relative).squeeze(-1)
            routing_bias = routing_bias + gather_candidates(anchor_geometry_bias, indices).squeeze(-1)

            if self.backend == "cuda":
                chunk_output = cuda_candidate_render(
                    query_features,
                    coordinates,
                    token_keys,
                    values,
                    geometry["opacity"],
                    geometry["mu"],
                    geometry["std"],
                    geometry["rho"],
                    routing_bias,
                    indices,
                    valid,
                )
            else:
                chunk_output = candidate_window_reference(
                    query_features,
                    coordinates,
                    token_keys,
                    values,
                    geometry["opacity"],
                    geometry["mu"],
                    geometry["std"],
                    geometry["rho"],
                    routing_bias,
                    indices,
                    valid,
                )
            outputs.append(chunk_output)

            if self.enable_diagnostics:
                with torch.no_grad():
                    candidate_std = gather_candidates(geometry["std"], indices)
                    candidate_rho = gather_candidates(geometry["rho"], indices)
                    candidate_opacity = gather_candidates(geometry["opacity"], indices)
                    gaussian_logits, _ = reference_gaussian_logit(
                        coordinates,
                        candidate_mu,
                        candidate_std,
                        candidate_rho,
                        candidate_opacity,
                    )
                    candidate_keys = gather_candidates(token_keys, indices)
                    query_key = (query_features.unsqueeze(2) * candidate_keys).sum(-1)
                    query_key = query_key / math.sqrt(query_features.shape[-1])
                    routing_logits = query_key + routing_bias
                    logits = (gaussian_logits + routing_logits).masked_fill(
                        ~valid.unsqueeze(0), torch.finfo(gaussian_logits.dtype).min
                    )
                    weights = F.softmax(logits.float(), dim=2)
                    gaussian_abs += gaussian_logits.detach().float().abs().mean()
                    routing_abs += routing_logits.detach().float().abs().mean()
                    entropy_sum += -(
                        weights.detach().float()
                        * weights.detach().float().clamp_min(1e-8).log()
                    ).sum(2).mean()
                    max_weight_sum += weights.detach().float().amax(2).mean()
                    chunks += 1

        if self.enable_diagnostics:
            self.last_stats = {
                "e2_backend": self.backend,
                "diagnostics_enabled": True,
                "gaussian_logit_abs_mean": float(gaussian_abs / max(chunks, 1)),
                "routing_logit_abs_mean": float(routing_abs / max(chunks, 1)),
                "routing_gaussian_logit_ratio": float(routing_abs / (gaussian_abs + 1e-8)),
                "candidate_entropy": float(entropy_sum / max(chunks, 1)),
                "candidate_max_weight": float(max_weight_sum / max(chunks, 1)),
            }
        else:
            self.last_stats = {
                "e2_backend": self.backend,
                "diagnostics_enabled": False,
            }
        output = torch.cat(outputs, dim=1)
        return output.permute(0, 2, 1).reshape(b, c, H, W)


class GSFusionGaussianLatentEncoderV2(nn.Module):
    CONFIGS = {
        "gaussian_v2lite_pointwise": ("pointwise", "pointwise", "lite"),
        "gaussian_v2lite_adci": ("adci", "adci", "lite"),
        "gaussian_v2_pointwise": ("pointwise", "pointwise", "msi_conditioned"),
        "gaussian_v2_adci": ("adci", "adci", "msi_conditioned"),
    }

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        hsi_encoder_type: str = "pointwise",
        msi_encoder_type: str = "pointwise",
        renderer_type: str = "lite",
        interaction_layers: int = 2,
        num_heads: int = 4,
        neighbor_dilations: Sequence[int] = (1, 2, 4),
        routing_dim: int = 16,
        candidate_radius: int = 5,
        query_chunk_size: int = 1024,
        e2_backend: str = "cuda",
        enable_diagnostics: bool = False,
        **_: object,
    ):
        super().__init__()
        if renderer_type not in {"lite", "msi_conditioned"}:
            raise ValueError(f"unknown renderer_type: {renderer_type}")
        self.dim = dim
        self.num_bands = num_bands
        self.hsi_encoder_type = hsi_encoder_type
        self.msi_encoder_type = msi_encoder_type
        self.renderer_type = renderer_type
        self.e2_backend = (
            "not_applicable_e1" if renderer_type == "lite" else e2_backend
        )

        self.hsi_encoder = ModalityEncoder(num_bands, dim, hsi_encoder_type)
        self.msi_encoder = ModalityEncoder(num_msi, dim, msi_encoder_type)
        self.interaction = GaussianLatentInteraction(
            dim, interaction_layers, num_heads, neighbor_dilations
        )
        self.geometry_head = GaussianGeometryHead(dim)
        self.value_proj = nn.Linear(dim, dim)
        self.e1_renderer = (
            DensityNormalizedRenderer(
                dim, candidate_radius=candidate_radius, chunk=query_chunk_size
            )
            if renderer_type == "lite"
            else None
        )
        self.e2_renderer = (
            MSIConditionedGaussianRenderer(
                dim,
                routing_dim=routing_dim,
                candidate_radius=candidate_radius,
                chunk=query_chunk_size,
                backend=e2_backend,
                enable_diagnostics=enable_diagnostics,
            )
            if renderer_type == "msi_conditioned"
            else None
        )
        self.decoder = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, num_bands, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.last_stats: Optional[Dict[str, object]] = None

    @classmethod
    def from_config_name(cls, name: str, **kwargs) -> "GSFusionGaussianLatentEncoderV2":
        if name not in cls.CONFIGS:
            raise KeyError(f"unknown V2 config: {name}")
        hsi, msi, renderer = cls.CONFIGS[name]
        return cls(
            hsi_encoder_type=hsi,
            msi_encoder_type=msi,
            renderer_type=renderer,
            **kwargs,
        )

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.geometry_head.head[-1].weight)
        nn.init.zeros_(self.geometry_head.head[-1].bias)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None) -> torch.Tensor:
        del sf
        b, _, h, w = lr_hsi.shape
        H, W = hr_msi.shape[-2:]
        if h < 1 or w < 1 or H < h or W < w:
            raise ValueError("HR output must be at least as large as the LR input")

        base = F.interpolate(lr_hsi, size=(H, W), mode="bicubic", align_corners=False)
        hsi_tokens = self.interaction(self.hsi_encoder(lr_hsi))
        msi_feature = self.msi_encoder(hr_msi)
        msi_center = sample_msi_at_lr_centers(msi_feature, (h, w))
        geometry = self.geometry_head(hsi_tokens, msi_center)
        tokens = hsi_tokens.flatten(2).transpose(1, 2)
        values = self.value_proj(tokens)

        if self.renderer_type == "lite":
            assert self.e1_renderer is not None
            rendered = self.e1_renderer(values, geometry, (h, w), (H, W))
            renderer_stats: Dict[str, object] = {"e1_backend": self.e1_renderer.last_backend}
        else:
            assert self.e2_renderer is not None
            rendered = self.e2_renderer(
                values, tokens, msi_feature, geometry, (h, w), (H, W)
            )
            renderer_stats = dict(self.e2_renderer.last_stats or {})

        residual = self.decoder(rendered)
        with torch.no_grad():
            stats: Dict[str, object] = {
                "renderer_type": self.renderer_type,
                "e2_backend": self.e2_backend,
                "token_count": h * w,
                "lr_shape": (b, self.dim, h, w),
                "hr_shape": (b, self.dim, H, W),
                "opacity_mean": float(geometry["opacity"].detach().mean()),
                "std_x_cell": float(geometry["std"][..., 0].detach().mean()),
                "std_y_cell": float(geometry["std"][..., 1].detach().mean()),
                "offset_abs_cell": float(geometry["offset"].detach().abs().mean()),
                "rho_abs_mean": float(geometry["rho"].detach().abs().mean()),
                "residual_abs_mean": float(residual.detach().abs().mean()),
            }
            for index, layer in enumerate(self.interaction.layers):
                for key, value in (layer.last_stats or {}).items():
                    stats[f"interaction{index}/{key}"] = value
            stats.update(renderer_stats)
            self.last_stats = stats
        return base + residual

    def collect_gs_stats(self):
        return dict(self.last_stats or {})


GSFusion = GSFusionGaussianLatentEncoderV2


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cosine = (pred * target).sum(dim=1) / (
        pred.norm(dim=1) * target.norm(dim=1) + eps
    )
    return (1.0 - cosine).mean()


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    sam_warmup_epochs: int = 5,
    sam_weight: float = 0.1,
    **_: object,
) -> torch.Tensor:
    loss = F.l1_loss(pred, target)
    if epoch >= sam_warmup_epochs and sam_weight > 0:
        loss = loss + sam_weight * sam_loss(pred, target)
    return loss
