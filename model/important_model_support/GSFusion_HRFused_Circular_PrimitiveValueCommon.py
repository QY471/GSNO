"""Shared components for the E5/E6/E7 primitive-value experiments.

This file is intentionally independent of the existing E3 implementation.  It
keeps E3's reconstruction backbone and renderer semantics while allowing the
Gaussian geometry and residual-value heads to consume different feature maps.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


def _resolve_adaptive_gaussian_rasterizer():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    extension_root = os.environ.get(
        "GSFUSION_ADAPTIVE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "adaptive3_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


class HRAdaptiveGaussianResidualDualSource(nn.Module):
    """E3-equivalent circular Gaussian renderer with separate head inputs."""

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        max_offset_px: float = 1.00,
        sigma_radius: float = 3.0,
    ) -> None:
        super().__init__()
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(dim + 1)
        self.dim = int(dim)
        self.std_min_px = float(std_min_px)
        self.std_max_px = float(std_max_px)
        self.max_offset_px = float(max_offset_px)
        self.sigma_radius = float(sigma_radius)

        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, kernel_size=1),
        )
        self.residual_value_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        nn.init.zeros_(self.residual_value_head[-1].weight)
        nn.init.zeros_(self.residual_value_head[-1].bias)

    @staticmethod
    def _pixel_centers(height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).view(1, height * width, 2)

    def forward(
        self,
        transport_x: torch.Tensor,
        value_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if value_x is None:
            value_x = transport_x
        if transport_x.shape != value_x.shape:
            raise ValueError(
                "transport_x and value_x must have identical BCHW shapes, got "
                f"{tuple(transport_x.shape)} and {tuple(value_x.shape)}"
            )

        batch, channels, height, width = transport_x.shape
        raw = self.geometry_head(transport_x)
        raw = raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )

        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_scalar_px = self.std_min_px + (
            self.std_max_px - self.std_min_px
        ) * torch.sigmoid(raw[..., 1:2])
        std_px = std_scalar_px.expand(-1, -1, 2)
        offset_px = raw.new_zeros(batch, height * width, 2)
        rho = raw.new_zeros(batch, height * width, 1)

        means_px = self._pixel_centers(
            height, width, transport_x.device, transport_x.dtype
        ).expand(batch, height * width, 2)
        means_px = torch.stack(
            [
                means_px[..., 0].clamp(0.0, float(width - 1)),
                means_px[..., 1].clamp(0.0, float(height - 1)),
            ],
            dim=-1,
        )

        delta_value = self.residual_value_head(value_x)
        delta_value = delta_value.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        ones = delta_value.new_ones(batch, height * width, 1)
        values_with_density = torch.cat([delta_value, ones], dim=-1)

        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_px / max(width, 1),
                self.sigma_radius * self.std_max_px / max(height, 1),
            ),
        )
        rasterized = self.rasterizer(
            opacity.float(),
            means_px.float(),
            std_px.float(),
            rho.float(),
            values_with_density.float(),
            height,
            width,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        )
        rasterized = rasterized.permute(0, 3, 1, 2).contiguous()
        numerator = rasterized[:, :channels]
        density = rasterized[:, channels:channels + 1]
        gaussian_delta = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            transport_abs = transport_x.detach().abs().mean()
            value_abs = delta_value.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_opacity_std": float(opacity.detach().std()),
                "hrgs_std_x_mean_px": float(std_px[..., 0].detach().mean()),
                "hrgs_std_y_mean_px": float(std_px[..., 1].detach().mean()),
                "hrgs_std_min_px": float(std_px.detach().min()),
                "hrgs_std_max_px": float(std_px.detach().max()),
                "hrgs_rho_abs_mean": float(rho.detach().abs().mean()),
                "hrgs_offset_abs_mean_px": float(offset_px.detach().abs().mean()),
                "hrgs_offset_max_abs_px": float(offset_px.detach().abs().max()),
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(value_abs),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(transport_abs),
                "hrgs_delta_input_ratio": float(
                    delta_abs / (transport_abs + 1e-8)
                ),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
            }
        return gaussian_delta


class MSIGuidedHSILocalRouting(nn.Module):
    """Use MSI keys to select a 3x3 neighborhood of HSI values."""

    def __init__(self, dim: int, routing_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = int(dim)
        self.routing_dim = int(routing_dim or max(1, dim // 4))
        self.q_proj = nn.Conv2d(dim, self.routing_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, self.routing_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_output_init(self) -> None:
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, f_h: torch.Tensor, f_m: torch.Tensor) -> torch.Tensor:
        if f_h.shape != f_m.shape:
            raise ValueError(
                f"F_H and F_M must match, got {tuple(f_h.shape)} and "
                f"{tuple(f_m.shape)}"
            )
        batch, _channels, height, width = f_h.shape
        q = self.q_proj(f_h)
        k = self.k_proj(f_m)
        v = self.v_proj(f_h)

        k_neighbors = F.unfold(
            F.pad(k, (1, 1, 1, 1), mode="replicate"), kernel_size=3
        ).view(batch, self.routing_dim, 9, height, width)
        v_neighbors = F.unfold(
            F.pad(v, (1, 1, 1, 1), mode="replicate"), kernel_size=3
        ).view(batch, self.dim, 9, height, width)

        logits = (
            q.unsqueeze(2) * k_neighbors
        ).sum(dim=1) / math.sqrt(self.routing_dim)
        weights = torch.softmax(logits, dim=1)
        context = (v_neighbors * weights.unsqueeze(1)).sum(dim=2)
        routing_delta = self.out_proj(context)

        with torch.no_grad():
            entropy = -(
                weights.detach() * weights.detach().clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            self.last_stats = {
                "primitive_routing_entropy": float(entropy),
                "primitive_routing_max_weight_mean": float(
                    weights.detach().max(dim=1).values.mean()
                ),
            }
        return routing_delta


class HRFusedPrimitiveValueBase(nn.Module):
    """E3-common layers, registered in exactly the E3 parameter order."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_msi = int(num_msi)

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.gaussian_refine = HRAdaptiveGaussianResidualDualSource(dim)
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        extra_rng_state = torch.get_rng_state()
        self.primitive_input = nn.Conv2d(2 * dim, dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        torch.set_rng_state(extra_rng_state)
        self._last_primitive_stats: Dict[str, float] = {}

    def reset_common_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_refine.reset_residual_init()
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)

    def _encode_common(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_h = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_m = f_msi
        joint = torch.cat([f_h, f_m], dim=1)
        f_fused = self.conv0(joint)
        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)
        return base, f_h, f_m, joint, f_fused, e_g

    def _record_primitive_stats(
        self,
        transport_x: torch.Tensor,
        value_x: torch.Tensor,
        pointwise_delta: Optional[torch.Tensor] = None,
        routing_delta: Optional[torch.Tensor] = None,
        routing: Optional[MSIGuidedHSILocalRouting] = None,
    ) -> None:
        with torch.no_grad():
            stats = {
                "primitive_transport_source_abs_mean": float(
                    transport_x.detach().abs().mean()
                ),
                "primitive_value_source_abs_mean": float(
                    value_x.detach().abs().mean()
                ),
            }
            if pointwise_delta is not None:
                stats["primitive_pointwise_delta_abs_mean"] = float(
                    pointwise_delta.detach().abs().mean()
                )
            if routing_delta is not None:
                stats["primitive_routing_delta_abs_mean"] = float(
                    routing_delta.detach().abs().mean()
                )
            if routing is not None and routing.last_stats:
                stats.update(routing.last_stats)
            self._last_primitive_stats = stats

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = dict(self.gaussian_refine.last_stats or {})
        stats.update(self._last_primitive_stats)
        return [] if not stats else [{"layer": "hr_gaussian", **stats}]


__all__ = [
    "HRAdaptiveGaussianResidualDualSource",
    "HRFusedPrimitiveValueBase",
    "MSIGuidedHSILocalRouting",
    "compute_loss",
    "sam_loss",
]
