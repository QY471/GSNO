"""Shared LR-cell primitive components for explicit LR-to-HR Gaussian transport."""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.important_model_support.GSFusion_HRFused_Circular_PrimitiveValueCommon import (
    MSIGuidedHSILocalRouting,
)


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


class ScaleSharedMSIFootprintSampler(nn.Module):
    """Sample the same 4x4 normalized positions inside every LR cell."""

    def __init__(self) -> None:
        super().__init__()
        offsets = (-0.375, -0.125, 0.125, 0.375)
        footprint = torch.tensor(
            [(oy, ox) for oy in offsets for ox in offsets], dtype=torch.float32
        )
        self.register_buffer("foot_offsets", footprint, persistent=False)

    def sample_grid(self, feature: torch.Tensor, lr_size: Tuple[int, int]):
        batch, channels, _height, _width = feature.shape
        h, w = (int(lr_size[0]), int(lr_size[1]))
        dtype, device = feature.dtype, feature.device
        cy = torch.arange(h, device=device, dtype=dtype) + 0.5
        cx = torch.arange(w, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        offsets = self.foot_offsets.to(device=device, dtype=dtype)
        py = gy.unsqueeze(0) + offsets[:, 0].view(16, 1, 1)
        px = gx.unsqueeze(0) + offsets[:, 1].view(16, 1, 1)
        grid = torch.stack((2.0 * px / w - 1.0, 2.0 * py / h - 1.0), dim=-1)
        grid = grid.view(1, 16 * h, w, 2).expand(batch, -1, -1, -1)
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.view(batch, channels, 16, h, w).reshape(
            batch, channels * 16, h, w
        )

    @staticmethod
    def sample_pixel_unshuffle(feature: torch.Tensor):
        return F.pixel_unshuffle(feature, downscale_factor=4)

    @staticmethod
    def sample_area(feature: torch.Tensor, lr_size: Tuple[int, int]):
        h, w = (int(lr_size[0]), int(lr_size[1]))
        canonical = F.interpolate(feature, size=(4 * h, 4 * w), mode="area")
        return F.pixel_unshuffle(canonical, downscale_factor=4)

    def forward(
        self,
        feature: torch.Tensor,
        lr_size: Tuple[int, int],
        mode: str = "point",
    ):
        h, w = (int(lr_size[0]), int(lr_size[1]))
        if mode == "area":
            return self.sample_area(feature, (h, w))
        if mode != "point":
            raise ValueError(f"unsupported MSI footprint mode: {mode}")
        height, width = feature.shape[-2:]
        if height == 4 * h and width == 4 * w:
            return self.sample_pixel_unshuffle(feature)
        return self.sample_grid(feature, (h, w))


class LRCellGaussianTransportDualSource(nn.Module):
    """One circular Gaussian per LR cell, directly rasterized to the HR grid."""

    def __init__(
        self,
        dim: int,
        std_min_cell: float = 0.125,
        std_max_cell: float = 1.0,
        std_init_cell: float = 0.25,
        sigma_radius: float = 3.0,
        density_eps: float = 1e-6,
        std_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if not (std_min_cell < std_init_cell < std_max_cell):
            raise ValueError("std_init_cell must lie strictly inside the std range")
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(dim + 1)
        self.dim = int(dim)
        self.std_min_cell = float(std_min_cell)
        self.std_max_cell = float(std_max_cell)
        self.std_init_cell = float(std_init_cell)
        self.sigma_radius = float(sigma_radius)
        self.density_eps = float(density_eps)
        if std_multiplier <= 0:
            raise ValueError("std_multiplier must be positive")
        self.std_multiplier = float(std_multiplier)

        # Keep exactly the same parameter shapes/order as E6's HR Gaussian
        # block so all earlier common layers and fc1/fc2 receive identical
        # seeded initialization under Train_Cave.py.
        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, 2, 1)
        )
        self.residual_value_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.last_stats: Optional[Dict[str, float]] = None
        self.last_aux: Optional[Dict[str, torch.Tensor]] = None

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        fraction = (self.std_init_cell - self.std_min_cell) / (
            self.std_max_cell - self.std_min_cell
        )
        std_logit = math.log(fraction / (1.0 - fraction))
        with torch.no_grad():
            self.geometry_head[-1].bias[1] = std_logit
        nn.init.zeros_(self.residual_value_head[-1].weight)
        nn.init.zeros_(self.residual_value_head[-1].bias)

    @staticmethod
    def lr_centers_in_hr(
        h: int, w: int, height: int, width: int, device, dtype
    ) -> torch.Tensor:
        y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * (
            height / h
        ) - 0.5
        x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * (
            width / w
        ) - 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).view(1, h * w, 2)

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
        std_cell = self.std_min_cell + (
            self.std_max_cell - self.std_min_cell
        ) * torch.sigmoid(raw[..., 1:2])
        scale_x, scale_y = width / w, height / h
        std_x_hr = std_cell * scale_x * self.std_multiplier
        std_y_hr = std_cell * scale_y * self.std_multiplier
        std_hr = torch.cat((std_x_hr, std_y_hr), dim=-1)
        means_hr = self.lr_centers_in_hr(
            h, w, height, width, transport_x.device, transport_x.dtype
        ).expand(batch, -1, -1)
        offset_hr = means_hr.new_zeros(batch, h * w, 2)
        rho = means_hr.new_zeros(batch, h * w, 1)

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
            low_density = density < 1e-4
            self.last_stats = {
                "primitive_count": float(h * w),
                "std_multiplier": self.std_multiplier,
                "std_cell_mean": float(std_cell.detach().mean()),
                "std_cell_std": float(std_cell.detach().std()),
                "std_cell_min": float(std_cell.detach().min()),
                "std_cell_max": float(std_cell.detach().max()),
                "std_x_hr_mean": float(std_x_hr.detach().mean()),
                "std_y_hr_mean": float(std_y_hr.detach().mean()),
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
            }
            self.last_aux = {
                "opacity": opacity.detach(),
                "std_cell": std_cell.detach(),
                "std_hr": std_hr.detach(),
                "means_hr": means_hr.detach(),
                "offset_hr": offset_hr.detach(),
                "rho": rho.detach(),
                "density": density.detach(),
            }

        if not return_aux:
            return gaussian_delta_hr
        aux = {
            "opacity": opacity,
            "std_cell": std_cell,
            "std_hr": std_hr,
            "means_hr": means_hr,
            "offset_hr": offset_hr,
            "rho": rho,
            "density": density,
        }
        return gaussian_delta_hr, aux


__all__ = [
    "LRCellGaussianTransportDualSource",
    "MSIGuidedHSILocalRouting",
    "ScaleSharedMSIFootprintSampler",
]
