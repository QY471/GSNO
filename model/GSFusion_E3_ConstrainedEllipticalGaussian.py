"""GSNO with fixed-center, area-preserving elliptical Gaussian kernels.

The geometry head predicts bounded anisotropy and orientation. Rendering uses
density normalization and adaptive 3-sigma support. A zero geometry head
initializes the kernels as circles.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as CircularPrimitiveEmbeddingGSFusion,
    HRAdaptiveGaussianResidual as CircularGaussianResidual,
    compute_loss,
    sam_loss,
)


class ConstrainedEllipticalGaussianResidual(CircularGaussianResidual):
    """Circular E3 plus bounded, area-preserving anisotropy and rotation."""

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        sigma_radius: float = 3.0,
        max_axis_ratio: float = 2.0,
    ) -> None:
        if max_axis_ratio < 1.0:
            raise ValueError("max_axis_ratio must be at least 1.0")
        super().__init__(
            dim=dim,
            std_min_px=std_min_px,
            std_max_px=std_max_px,
            sigma_radius=sigma_radius,
        )
        self.max_axis_ratio = float(max_axis_ratio)
        # Because sigma_1/sigma_2 = exp(2*a), this bound gives the requested
        # maximum principal-axis ratio.  A signed a swaps the two axes.
        self.max_log_stretch = 0.5 * math.log(self.max_axis_ratio)

        # Predict log-stretch and orientation at each location.
        self.anisotropy_head = nn.Conv2d(dim, 2, kernel_size=1)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        super().reset_residual_init()
        # Exact circular initialization.  The zero weights still receive
        # gradients because the head reads nonzero primitive features.
        nn.init.zeros_(self.anisotropy_head.weight)
        nn.init.zeros_(self.anisotropy_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        circular_raw = self.geometry_head(x)
        circular_raw = circular_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        opacity = 0.05 + 0.95 * torch.sigmoid(circular_raw[..., 0:1])

        shape_raw = self.anisotropy_head(x)
        shape_raw = shape_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        log_stretch = self.max_log_stretch * torch.tanh(shape_raw[..., 0:1])
        theta = 0.5 * math.pi * torch.tanh(shape_raw[..., 1:2])

        # Preserve the determinant for a given base sigma:
        # sigma_1 * sigma_2 = base_sigma**2.  Tighten the admissible base
        # interval as anisotropy grows so both principal axes remain inside
        # the original [std_min_px, std_max_px] physical HR-pixel range.
        stretch_abs = log_stretch.abs()
        base_min = self.std_min_px * torch.exp(stretch_abs)
        base_max = self.std_max_px * torch.exp(-stretch_abs)
        base_sigma = base_min + (base_max - base_min) * torch.sigmoid(
            circular_raw[..., 1:2]
        )
        sigma_1 = base_sigma * torch.exp(log_stretch)
        sigma_2 = base_sigma * torch.exp(-log_stretch)

        # Convert principal-axis scale + explicit angle to the rasterizer's
        # equivalent (marginal std_x, marginal std_y, correlation rho).
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        var_1 = sigma_1.square()
        var_2 = sigma_2.square()
        var_x = var_1 * cos_theta.square() + var_2 * sin_theta.square()
        var_y = var_1 * sin_theta.square() + var_2 * cos_theta.square()
        cov_xy = (var_1 - var_2) * sin_theta * cos_theta
        std_x = torch.sqrt(var_x)
        std_y = torch.sqrt(var_y)
        rho = cov_xy / (std_x * std_y).clamp_min(1e-12)
        rho = rho.clamp(-0.95, 0.95)
        std_px = torch.cat([std_x, std_y], dim=-1)

        offset_px = x.new_zeros(batch, height * width, 2)
        base_px = self._pixel_centers(
            height, width, x.device, x.dtype
        ).expand(batch, height * width, 2)
        means_px = base_px

        delta_value = self.residual_value_head(x)
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
        out = x + gaussian_delta

        with torch.no_grad():
            axis_major = torch.maximum(sigma_1, sigma_2)
            axis_minor = torch.minimum(sigma_1, sigma_2)
            axis_ratio = axis_major / axis_minor.clamp_min(1e-12)
            x_abs = x.detach().abs().mean()
            value_abs = delta_value.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_opacity_std": float(opacity.detach().std()),
                "hrgs_base_sigma_mean_px": float(base_sigma.detach().mean()),
                "hrgs_axis_major_mean_px": float(axis_major.detach().mean()),
                "hrgs_axis_minor_mean_px": float(axis_minor.detach().mean()),
                "hrgs_axis_ratio_mean": float(axis_ratio.detach().mean()),
                "hrgs_axis_ratio_max": float(axis_ratio.detach().max()),
                "hrgs_log_stretch_abs_mean": float(log_stretch.detach().abs().mean()),
                "hrgs_theta_abs_mean_deg": float(
                    theta.detach().abs().mean() * (180.0 / math.pi)
                ),
                "hrgs_std_x_mean_px": float(std_x.detach().mean()),
                "hrgs_std_y_mean_px": float(std_y.detach().mean()),
                "hrgs_rho_abs_mean": float(rho.detach().abs().mean()),
                "hrgs_offset_abs_mean_px": float(offset_px.detach().abs().mean()),
                "hrgs_offset_max_abs_px": float(offset_px.detach().abs().max()),
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(value_abs),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(x_abs),
                "hrgs_delta_input_ratio": float(delta_abs / (x_abs + 1e-8)),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
                "hrgs_max_axis_ratio_limit": float(self.max_axis_ratio),
            }
        return out


class GSFusion(CircularPrimitiveEmbeddingGSFusion):
    """DIM-configurable E3 with constrained elliptical Gaussian geometry."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )

        # Replacing the registered module retains the common E3 module order.
        # The caller's RNG state is restored because the paired checkpoint
        # supplies every common tensor; only anisotropy_head is model-specific.
        rng_state = torch.get_rng_state()
        self.gaussian_refine = ConstrainedEllipticalGaussianResidual(
            dim=dim,
            max_axis_ratio=max_axis_ratio,
        )
        torch.set_rng_state(rng_state)

        self.arch_summary = (
            "E3 constrained elliptical Gaussian: DIM-configurable ADCI and "
            "primitive embedding unchanged; fixed HR-pixel center; bounded "
            "area-preserving principal axes with learned orientation; "
            "axis ratio <= 2; density-normalized adaptive-3sigma scatter"
        )
        self.reset_custom_init()


__all__ = [
    "GSFusion",
    "ConstrainedEllipticalGaussianResidual",
    "compute_loss",
    "sam_loss",
]
