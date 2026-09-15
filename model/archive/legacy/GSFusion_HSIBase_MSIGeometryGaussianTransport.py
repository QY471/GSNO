"""Exact C: HSI-base transport with D as value and MSI controlling geometry.

The HSI and MSI streams are encoded independently and aligned on the HR grid.
Their concatenation predicts a cross-modal detail latent D used as Gaussian
value, while the MSI encoder feature predicts circular std/opacity. D has no direct
path to the output: it must pass through density-normalized adaptive-3sigma
CUDA scatter. The transported
detail is added to the HSI latent, decoded to a 31-band residual, and finally
added to raw bicubic HSI. Offset and rho are fixed to zero.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


def _resolve_adaptive_gaussian_rasterizer():
    """Load the isolated adaptive-window extension bundled for this model."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    extension_root = os.environ.get(
        "GSFUSION_ADAPTIVE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "adaptive3_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


class HSIBaseGaussianCrossModalTransport(nn.Module):
    """Transport a cross-modal detail latent with circular HR Gaussians."""

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
            # opacity logit + one scalar std logit shared by x/y
            nn.Conv2d(dim, 2, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    @staticmethod
    def _pixel_centers(height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).view(1, height * width, 2)

    def forward(
        self, detail: torch.Tensor, geometry_condition: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = detail.shape
        raw = self.geometry_head(geometry_condition)
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

        base_px = self._pixel_centers(
            height, width, detail.device, detail.dtype
        ).expand(batch, height * width, 2)
        means_px = base_px + offset_px
        means_px = torch.stack(
            [
                means_px[..., 0].clamp(0.0, float(width - 1)),
                means_px[..., 1].clamp(0.0, float(height - 1)),
            ],
            dim=-1,
        )

        detail_value = detail.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        ones = detail_value.new_ones(batch, height * width, 1)
        values_with_density = torch.cat([detail_value, ones], dim=-1)

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
        gaussian_detail = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            value_abs = detail_value.detach().abs().mean()
            transported_abs = gaussian_detail.detach().abs().mean()
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_opacity_std": float(opacity.detach().std()),
                "hrgs_std_x_mean_px": float(std_px[..., 0].detach().mean()),
                "hrgs_std_y_mean_px": float(std_px[..., 1].detach().mean()),
                "hrgs_rho_abs_mean": float(rho.detach().abs().mean()),
                "hrgs_offset_abs_mean_px": float(offset_px.detach().abs().mean()),
                "hrgs_offset_max_abs_px": float(offset_px.detach().abs().max()),
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(value_abs),
                "hrgs_delta_abs_mean": float(transported_abs),
                "hrgs_input_abs_mean": float(value_abs),
                "hrgs_delta_input_ratio": float(
                    transported_abs / (value_abs + 1e-8)
                ),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
            }
        return gaussian_detail


class GSFusion(nn.Module):
    """HSI latent base plus mandatory Gaussian-transported cross-modal detail."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)

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
        self.gaussian_refine = HSIBaseGaussianCrossModalTransport(dim)
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        self.arch_summary = (
            "Exact C: HSI-base latent + joint HSI/MSI detail D as Gaussian "
            "value; MSI encoder feature predicts circular std/opacity; "
            "mandatory normalized adaptive-3sigma transport + decoder"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        # Start with no injected cross-modal detail: F_out == F_H.
        nn.init.zeros_(self.conv0[-1].weight)
        nn.init.zeros_(self.conv0[-1].bias)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = self.gaussian_refine.last_stats
        return [] if stats is None else [{"layer": "hr_gaussian", **stats}]

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
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

        f_hsi_hr = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        joint = torch.cat([f_hsi_hr, f_msi], dim=1)
        detail = self.conv0(joint)
        transported_detail = self.gaussian_refine(detail, f_msi)
        hsi_with_detail = f_hsi_hr + transported_detail
        residual = self.fc2(F.gelu(self.fc1(hsi_with_detail)))
        return base + residual


__all__ = [
    "GSFusion",
    "HSIBaseGaussianCrossModalTransport",
    "compute_loss",
    "sam_loss",
]
