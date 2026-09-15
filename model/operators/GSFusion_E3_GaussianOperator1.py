"""E3 backbone with one mandatory HR Gaussian operator block.

Unlike E3's terminal Gaussian residual branch, there is no parallel fused
latent that can bypass the Gaussian renderer and reach the decoder.  ADCI is
retained for HSI/MSI feature extraction.  A 1x1 projection only aligns the
concatenated features to ``dim`` channels, after which every latent value is
rendered by the Gaussian operator block before the lightweight 1x1 decoder.
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
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    extension_root = os.environ.get(
        "GSFUSION_ADAPTIVE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "adaptive3_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


class GaussianOperatorBlock(nn.Module):
    """Pointwise spectral mixing followed by mandatory Gaussian rendering."""

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        sigma_radius: float = 3.0,
    ) -> None:
        super().__init__()
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(dim + 1)
        self.dim = int(dim)
        self.std_min_px = float(std_min_px)
        self.std_max_px = float(std_max_px)
        self.sigma_radius = float(sigma_radius)

        # This transform mixes channels independently at every HR position.
        # It does not read spatial neighbours; the renderer below is the only
        # spatial integration inside this block.
        self.spectral_transform = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )
        # HR-MSI conditions opacity and the circular physical-pixel scale.
        self.geometry_head = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_operator_init(self) -> None:
        # Start the pointwise update as identity.  Gaussian rendering remains
        # mandatory: there is deliberately no x + gaussian_delta bypass.
        nn.init.zeros_(self.spectral_transform[-1].weight)
        nn.init.zeros_(self.spectral_transform[-1].bias)

    @staticmethod
    def _pixel_centers(height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).view(1, height * width, 2)

    def forward(self, x: torch.Tensor, msi_feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if msi_feature.shape != x.shape:
            raise ValueError(
                f"MSI condition shape {tuple(msi_feature.shape)} must equal "
                f"latent shape {tuple(x.shape)}"
            )

        values = x + self.spectral_transform(x)
        geometry = self.geometry_head(torch.cat([values, msi_feature], dim=1))
        geometry = geometry.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        opacity = 0.05 + 0.95 * torch.sigmoid(geometry[..., 0:1])
        std_scalar_px = self.std_min_px + (
            self.std_max_px - self.std_min_px
        ) * torch.sigmoid(geometry[..., 1:2])
        std_px = std_scalar_px.expand(-1, -1, 2)
        rho = geometry.new_zeros(batch, height * width, 1)
        means_px = self._pixel_centers(
            height, width, x.device, x.dtype
        ).expand(batch, height * width, 2)

        flat_values = values.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )
        values_with_density = torch.cat(
            [flat_values, flat_values.new_ones(batch, height * width, 1)],
            dim=-1,
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
        # Crucial design choice: return the rendered latent itself, not
        # ``x + a small rendered delta``.
        out = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            input_abs = x.detach().abs().mean()
            pointwise_abs = (values - x).detach().abs().mean()
            render_change_abs = (out - values).detach().abs().mean()
            self.last_stats = {
                "gop_opacity_mean": float(opacity.detach().mean()),
                "gop_opacity_std": float(opacity.detach().std()),
                "gop_std_mean_px": float(std_scalar_px.detach().mean()),
                "gop_std_min_px": float(std_scalar_px.detach().min()),
                "gop_std_max_px": float(std_scalar_px.detach().max()),
                "gop_density_min": float(density.detach().min()),
                "gop_density_mean": float(density.detach().mean()),
                "gop_input_abs_mean": float(input_abs),
                "gop_pointwise_update_abs_mean": float(pointwise_abs),
                "gop_render_change_abs_mean": float(render_change_abs),
                "gop_pointwise_input_ratio": float(
                    pointwise_abs / (input_abs + 1e-8)
                ),
                "gop_render_input_ratio": float(
                    render_change_abs / (input_abs + 1e-8)
                ),
                "gop_forced_render": 1.0,
                "gop_adaptive_window": 1.0,
                "gop_sigma_radius": float(self.sigma_radius),
            }
        return out


class GSFusion(nn.Module):
    """ADCI feature extraction followed by one mandatory Gaussian operator."""

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
        # Channel alignment only.  Its output has no route to the decoder
        # except through both Gaussian operator blocks.
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.operator_blocks = nn.ModuleList([GaussianOperatorBlock(dim)])
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)
        self.arch_summary = (
            "ADCI retained; bicubic LR-HSI latent + HR-MSI latent are aligned "
            "by 1x1 conv; one pointwise-spectral/MSI-conditioned Gaussian "
            "operator block must render the complete HR latent; no fused-"
            "feature-to-decoder bypass; lightweight 1x1 decoder"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        for block in self.operator_blocks:
            block.reset_operator_init()

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        result = []
        for index, block in enumerate(self.operator_blocks, start=1):
            if block.last_stats is not None:
                result.append({"layer": f"gaussian_operator_{index}", **block.last_stats})
        return result

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
        latent = self.conv0(torch.cat([f_hsi_hr, f_msi], dim=1))
        for block in self.operator_blocks:
            latent = block(latent, f_msi)
        residual = self.fc2(F.gelu(self.fc1(latent)))
        return base + residual


__all__ = [
    "GSFusion",
    "GaussianOperatorBlock",
    "compute_loss",
    "sam_loss",
]
