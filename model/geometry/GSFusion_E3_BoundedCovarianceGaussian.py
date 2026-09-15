"""E3 with a bounded, area-preserving covariance-residual Gaussian.

The Gaussian center remains fixed on the HR pixel and the determinant remains
equal to the circular base scale.  Two zero-initialized trace-free covariance
components can learn any mild anisotropy direction directly from the circular
starting point.  This avoids the unstable freedom of offset/std_x/std_y/rho
while avoiding an explicit angle that is unidentifiable at a perfect circle.
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


class BoundedCovarianceGaussianResidual(CircularGaussianResidual):
    """Circular E3 plus a bounded trace-free log-covariance residual."""

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
        self.max_log_stretch = 0.5 * math.log(self.max_axis_ratio)
        self.covariance_head = nn.Conv2d(dim, 2, kernel_size=1)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        super().reset_residual_init()
        nn.init.zeros_(self.covariance_head.weight)
        nn.init.zeros_(self.covariance_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        circular_raw = self.geometry_head(x)
        circular_raw = circular_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        opacity = 0.05 + 0.95 * torch.sigmoid(circular_raw[..., 0:1])

        covariance_raw = self.covariance_head(x)
        covariance_raw = covariance_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        raw_a = covariance_raw[..., 0:1]
        raw_b = covariance_raw[..., 1:2]
        raw_norm = torch.sqrt(raw_a.square() + raw_b.square() + 1e-12)
        bounded_norm = self.max_log_stretch * torch.tanh(raw_norm)
        component_scale = bounded_norm / raw_norm
        component_a = raw_a * component_scale
        component_b = raw_b * component_scale
        component_norm_sq = component_a.square() + component_b.square()
        # Subtract the numerical floor so the zero head remains an exactly
        # circular geometry rather than a tiny artificial ellipse.
        stretch = torch.sqrt(component_norm_sq + 1e-12) - 1e-6

        # Keep both principal-axis standard deviations inside E3's physical
        # HR-pixel range while preserving their product for a given base scale.
        base_min = self.std_min_px * torch.exp(stretch)
        base_max = self.std_max_px * torch.exp(-stretch)
        base_sigma = base_min + (base_max - base_min) * torch.sigmoid(
            circular_raw[..., 1:2]
        )

        # For T=[[a,b],[b,-a]], exp(2T) has determinant one.  The closed form
        # below maps base_sigma^2*exp(2T) to (std_x,std_y,rho) without an
        # explicit angle.  sinh(2r)/r tends smoothly to 2 as r approaches 0.
        two_stretch = 2.0 * stretch
        cosh_term = torch.cosh(two_stretch)
        sinh_over_r = torch.where(
            stretch < 1e-4,
            2.0 + (4.0 / 3.0) * stretch.square(),
            torch.sinh(two_stretch) / stretch.clamp_min(1e-6),
        )
        base_variance = base_sigma.square()
        var_x = base_variance * (cosh_term + sinh_over_r * component_a)
        var_y = base_variance * (cosh_term - sinh_over_r * component_a)
        cov_xy = base_variance * sinh_over_r * component_b
        std_x = torch.sqrt(var_x.clamp_min(1e-12))
        std_y = torch.sqrt(var_y.clamp_min(1e-12))
        rho = cov_xy / (std_x * std_y).clamp_min(1e-12)
        rho = rho.clamp(-0.95, 0.95)
        std_px = torch.cat([std_x, std_y], dim=-1)

        base_px = self._pixel_centers(
            height, width, x.device, x.dtype
        ).expand(batch, height * width, 2)
        means_px = base_px

        values = self.residual_value_head(x)
        values = values.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        values_with_density = torch.cat(
            [values, values.new_ones(batch, height * width, 1)], dim=-1
        )
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
        ).permute(0, 3, 1, 2).contiguous()
        numerator = rasterized[:, :channels]
        density = rasterized[:, channels:channels + 1]
        gaussian_delta = numerator / density.clamp_min(1e-6)
        out = x + gaussian_delta

        with torch.no_grad():
            axis_ratio = torch.exp(2.0 * stretch)
            theta = 0.5 * torch.atan2(component_b, component_a)
            input_abs = x.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_base_sigma_mean_px": float(base_sigma.detach().mean()),
                "hrgs_axis_ratio_mean": float(axis_ratio.detach().mean()),
                "hrgs_axis_ratio_max": float(axis_ratio.detach().max()),
                "hrgs_log_stretch_mean": float(stretch.detach().mean()),
                "hrgs_cov_a_abs_mean": float(component_a.detach().abs().mean()),
                "hrgs_cov_b_abs_mean": float(component_b.detach().abs().mean()),
                "hrgs_theta_abs_mean_deg": float(
                    theta.detach().abs().mean() * (180.0 / math.pi)
                ),
                "hrgs_std_x_mean_px": float(std_x.detach().mean()),
                "hrgs_std_y_mean_px": float(std_y.detach().mean()),
                "hrgs_rho_abs_mean": float(rho.detach().abs().mean()),
                "hrgs_offset_abs_mean_px": 0.0,
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(values.detach().abs().mean()),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(input_abs),
                "hrgs_delta_input_ratio": float(delta_abs / (input_abs + 1e-8)),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
                "hrgs_max_axis_ratio_limit": float(self.max_axis_ratio),
                "hrgs_bounded_covariance": 1.0,
            }
        return out


class GSFusion(CircularPrimitiveEmbeddingGSFusion):
    """DIM-configurable E3 with bounded covariance-residual geometry."""

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
        rng_state = torch.get_rng_state()
        self.gaussian_refine = BoundedCovarianceGaussianResidual(
            dim=dim,
            max_axis_ratio=max_axis_ratio,
        )
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "E3 bounded covariance residual: fixed HR center; zero-initialized "
            "trace-free log-covariance; arbitrary mild orientation; bounded "
            "axis ratio; area-preserving adaptive-3sigma normalized scatter"
        )
        self.reset_custom_init()


__all__ = [
    "GSFusion",
    "BoundedCovarianceGaussianResidual",
    "compute_loss",
    "sam_loss",
]
