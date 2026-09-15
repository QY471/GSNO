"""Bounded area-preserving covariance for LR-cell Gaussian transport."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.important_model_support.GSFusion_LRPrimitiveTransportCommon import (
    LRCellGaussianTransportDualSource,
)


class LRCellBoundedCovarianceGaussianTransport(LRCellGaussianTransportDualSource):
    """Add zero-initialized mild anisotropy without offsets or area changes."""

    def __init__(
        self,
        dim: int,
        std_min_cell: float = 0.125,
        std_max_cell: float = 1.0,
        std_init_cell: float = 0.25,
        sigma_radius: float = 3.0,
        density_eps: float = 1e-6,
        std_multiplier: float = 1.0,
        max_axis_ratio: float = 2.0,
    ) -> None:
        if max_axis_ratio < 1.0:
            raise ValueError("max_axis_ratio must be at least 1.0")
        super().__init__(
            dim=dim,
            std_min_cell=std_min_cell,
            std_max_cell=std_max_cell,
            std_init_cell=std_init_cell,
            sigma_radius=sigma_radius,
            density_eps=density_eps,
            std_multiplier=std_multiplier,
        )
        self.max_axis_ratio = float(max_axis_ratio)
        self.max_log_stretch = 0.5 * math.log(self.max_axis_ratio)
        self.covariance_head = nn.Conv2d(dim, 2, kernel_size=1)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_custom_init(self) -> None:
        super().reset_custom_init()
        nn.init.zeros_(self.covariance_head.weight)
        nn.init.zeros_(self.covariance_head.bias)

    def forward(
        self,
        transport_x: torch.Tensor,
        value_x: torch.Tensor,
        out_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        if transport_x.shape != value_x.shape:
            raise ValueError(
                "transport_x and value_x must have identical BCHW shapes, got "
                f"{tuple(transport_x.shape)} and {tuple(value_x.shape)}"
            )
        batch, channels, h, w = transport_x.shape
        height, width = (int(out_size[0]), int(out_size[1]))
        if height < h or width < w:
            raise ValueError("target HR grid must not be smaller than the LR grid")

        raw = self.geometry_head(transport_x).permute(0, 2, 3, 1).contiguous()
        raw = raw.view(batch, h * w, 2)
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])

        covariance_raw = self.covariance_head(transport_x)
        covariance_raw = covariance_raw.permute(0, 2, 3, 1).contiguous()
        covariance_raw = covariance_raw.view(batch, h * w, 2)
        raw_a = covariance_raw[..., 0:1]
        raw_b = covariance_raw[..., 1:2]
        raw_norm = torch.sqrt(raw_a.square() + raw_b.square() + 1e-12)
        bounded_norm = self.max_log_stretch * torch.tanh(raw_norm)
        component_scale = bounded_norm / raw_norm
        component_a = raw_a * component_scale
        component_b = raw_b * component_scale
        stretch = torch.sqrt(
            component_a.square() + component_b.square() + 1e-12
        ) - 1e-6

        base_min = self.std_min_cell * torch.exp(stretch)
        base_max = self.std_max_cell * torch.exp(-stretch)
        base_std_cell = base_min + (base_max - base_min) * torch.sigmoid(
            raw[..., 1:2]
        )

        two_stretch = 2.0 * stretch
        cosh_term = torch.cosh(two_stretch)
        sinh_over_r = torch.where(
            stretch < 1e-4,
            2.0 + (4.0 / 3.0) * stretch.square(),
            torch.sinh(two_stretch) / stretch.clamp_min(1e-6),
        )
        base_variance = base_std_cell.square()
        var_x_cell = base_variance * (
            cosh_term + sinh_over_r * component_a
        )
        var_y_cell = base_variance * (
            cosh_term - sinh_over_r * component_a
        )
        cov_xy_cell = base_variance * sinh_over_r * component_b
        std_x_cell = torch.sqrt(var_x_cell.clamp_min(1e-12))
        std_y_cell = torch.sqrt(var_y_cell.clamp_min(1e-12))
        rho = cov_xy_cell / (std_x_cell * std_y_cell).clamp_min(1e-12)
        rho = rho.clamp(-0.95, 0.95)

        scale_x = width / w
        scale_y = height / h
        std_x_hr = std_x_cell * scale_x * self.std_multiplier
        std_y_hr = std_y_cell * scale_y * self.std_multiplier
        std_hr = torch.cat((std_x_hr, std_y_hr), dim=-1)
        means_hr = self.lr_centers_in_hr(
            h, w, height, width, transport_x.device, transport_x.dtype
        ).expand(batch, -1, -1)
        offset_hr = means_hr.new_zeros(batch, h * w, 2)

        value = self.residual_value_head(value_x)
        value = value.permute(0, 2, 3, 1).contiguous().view(
            batch, h * w, channels
        )
        values_with_density = torch.cat(
            (value, value.new_ones(batch, h * w, 1)), dim=-1
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius
                * self.std_max_cell
                * self.std_multiplier
                / max(w, 1),
                self.sigma_radius
                * self.std_max_cell
                * self.std_multiplier
                / max(h, 1),
            ),
        )
        rasterized = self.rasterizer(
            opacity.float(),
            means_hr.float(),
            std_hr.float(),
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
        density = rasterized[:, channels : channels + 1]
        gaussian_delta_hr = numerator / density.clamp_min(self.density_eps)

        with torch.no_grad():
            axis_ratio = torch.exp(2.0 * stretch)
            theta = 0.5 * torch.atan2(component_b, component_a)
            low_density = density < 1e-4
            self.last_stats = {
                "primitive_count": float(h * w),
                "std_multiplier": self.std_multiplier,
                "base_std_cell_mean": float(base_std_cell.detach().mean()),
                "std_cell_mean": float(
                    torch.sqrt(std_x_cell * std_y_cell).detach().mean()
                ),
                "std_x_hr_mean": float(std_x_hr.detach().mean()),
                "std_y_hr_mean": float(std_y_hr.detach().mean()),
                "axis_ratio_mean": float(axis_ratio.detach().mean()),
                "axis_ratio_max": float(axis_ratio.detach().max()),
                "rho_abs_mean": float(rho.detach().abs().mean()),
                "theta_abs_mean_deg": float(
                    theta.detach().abs().mean() * (180.0 / math.pi)
                ),
                "opacity_mean": float(opacity.detach().mean()),
                "density_min": float(density.detach().min()),
                "density_mean": float(density.detach().mean()),
                "density_max": float(density.detach().max()),
                "density_lt_1e_4_ratio": float(low_density.float().mean()),
                "value_abs_mean": float(value.detach().abs().mean()),
                "gaussian_delta_abs_mean": float(
                    gaussian_delta_hr.detach().abs().mean()
                ),
                "adaptive_window": 1.0,
                "sigma_radius": self.sigma_radius,
                "raster_ratio": float(raster_ratio),
                "bounded_covariance": 1.0,
                "max_axis_ratio_limit": self.max_axis_ratio,
            }
            self.last_aux = {
                "opacity": opacity.detach(),
                "std_cell": base_std_cell.detach(),
                "std_hr": std_hr.detach(),
                "means_hr": means_hr.detach(),
                "offset_hr": offset_hr.detach(),
                "rho": rho.detach(),
                "density": density.detach(),
                "axis_ratio": axis_ratio.detach(),
                "theta": theta.detach(),
            }

        if not return_aux:
            return gaussian_delta_hr
        return gaussian_delta_hr, dict(self.last_aux)


__all__ = ["LRCellBoundedCovarianceGaussianTransport"]
