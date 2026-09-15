"""E2: HR-fused circular Gaussian with a direct HSI spectral value source.

F_H is the encoded LR-HSI bicubic-aligned to the HR grid; F_M is the encoded
HR-MSI; and F is the original fusion of concat(F_H,F_M). Geometry reads only F
and predicts circular scalar std plus opacity. Gaussian value reads exactly
concat(F,F_H). The normalized scatter delta is added to F before the original
decoder. Offset and rho are fixed to zero.
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


class HRAdaptiveGaussianResidual(nn.Module):
    """One local, density-normalized Gaussian residual on the HR feature grid."""

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
            # opacity logit + one shared (x/y) isotropic std logit
            nn.Conv2d(dim, 2, kernel_size=1),
        )
        self.residual_value_head = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        """Make the Gaussian residual exactly zero at initialization."""
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
        self, fused_f: torch.Tensor, value_source: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = fused_f.shape
        raw = self.geometry_head(fused_f)
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
            height, width, fused_f.device, fused_f.dtype
        ).expand(batch, height * width, 2)
        means_px = base_px + offset_px
        means_px = torch.stack(
            [
                means_px[..., 0].clamp(0.0, float(width - 1)),
                means_px[..., 1].clamp(0.0, float(height - 1)),
            ],
            dim=-1,
        )

        delta_value = self.residual_value_head(value_source)
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
        out = fused_f + gaussian_delta

        with torch.no_grad():
            x_abs = fused_f.detach().abs().mean()
            value_abs = delta_value.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
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
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(x_abs),
                "hrgs_delta_input_ratio": float(delta_abs / (x_abs + 1e-8)),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
            }
        return out


class GSFusion(nn.Module):
    """Strong two-stream ADCI backbone plus one HR Gaussian residual."""

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
        self.gaussian_refine = HRAdaptiveGaussianResidual(dim)
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        self.arch_summary = (
            "E2: F_H=upsampled HSI latent; F_M=MSI latent; "
            "F=fusion(concat(F_H,F_M)); value=ValueHead(concat(F,F_H)); "
            "std/opacity=GeometryHead(F); circular normalized scatter; "
            "F+delta -> original decoder"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_refine.reset_residual_init()

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

        F_H = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        F_M = f_msi
        F_fused = self.conv0(torch.cat([F_H, F_M], dim=1))
        value_source = torch.cat([F_fused, F_H], dim=1)
        refined = self.gaussian_refine(F_fused, value_source)
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = [
    "GSFusion",
    "HRAdaptiveGaussianResidual",
    "compute_loss",
    "sam_loss",
]
