"""GSPan-inspired continuous Gaussian residual field for HSI-MSI fusion.

The model estimates Gaussian primitives from LR-HSI and HR-MSI on a reference
grid, then renders the learned spectral residual field on an independently
specified query grid.  All primitive geometry is expressed relative to the
reference domain rather than a fixed output tensor size.  The learned detail
path has no conventional fused-feature reconstruction bypass: every learned
spectral correction reaches the output through Gaussian rendering.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ADCI_Exact import ADCIExact
from model.GSFusion_GSNO import compute_loss, sam_loss


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


class NormalizedContinuousGaussianResidual(nn.Module):
    """Render a learned spectral residual field on an arbitrary query grid."""

    def __init__(
        self,
        dim: int,
        num_bands: int,
        std_min_reference_px: float = 0.30,
        std_max_reference_px: float = 1.50,
        std_init_reference_px: float = 0.55,
        max_axis_ratio: float = 2.0,
        max_offset_reference_px: float = 0.50,
        sigma_radius: float = 3.0,
    ) -> None:
        super().__init__()
        if not (
            0.0
            < std_min_reference_px
            < std_init_reference_px
            < std_max_reference_px
        ):
            raise ValueError("invalid reference-pixel Gaussian scale range")
        if max_axis_ratio < 1.0:
            raise ValueError("max_axis_ratio must be at least 1")
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(num_bands)
        self.num_bands = int(num_bands)
        self.std_min_reference_px = float(std_min_reference_px)
        self.std_max_reference_px = float(std_max_reference_px)
        self.std_init_reference_px = float(std_init_reference_px)
        self.max_log_stretch = 0.5 * math.log(float(max_axis_ratio))
        self.max_axis_ratio = float(max_axis_ratio)
        self.max_offset_reference_px = float(max_offset_reference_px)
        self.sigma_radius = float(sigma_radius)

        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 6, 1),
        )
        self.spectral_residual_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, num_bands, 1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        fraction = (
            self.std_init_reference_px - self.std_min_reference_px
        ) / (self.std_max_reference_px - self.std_min_reference_px)
        std_logit = math.log(fraction / (1.0 - fraction))
        with torch.no_grad():
            self.geometry_head[-1].bias[2] = std_logit
        nn.init.normal_(self.spectral_residual_head[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.spectral_residual_head[-1].bias)

    @staticmethod
    def _normalized_centers(height: int, width: int, device, dtype):
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).view(1, height * width, 2)

    def forward(
        self,
        embedding: torch.Tensor,
        output_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        batch, _channels, reference_h, reference_w = embedding.shape
        output_h, output_w = map(int, output_size)
        if min(reference_h, reference_w, output_h, output_w) <= 0:
            raise ValueError(f"invalid output_size: {output_size}")

        raw = self.geometry_head(embedding)
        raw = raw.permute(0, 2, 3, 1).contiguous().view(
            batch, reference_h * reference_w, 6
        )
        offset_x_reference = self.max_offset_reference_px * torch.tanh(raw[..., 0:1])
        offset_y_reference = self.max_offset_reference_px * torch.tanh(raw[..., 1:2])
        base_sigma_reference = self.std_min_reference_px + (
            self.std_max_reference_px - self.std_min_reference_px
        ) * torch.sigmoid(raw[..., 2:3])
        log_stretch = self.max_log_stretch * torch.tanh(raw[..., 3:4])
        theta = math.pi * torch.tanh(raw[..., 4:5])
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 5:6])

        centers_normalized = self._normalized_centers(
            reference_h, reference_w, embedding.device, embedding.dtype
        ).expand(batch, -1, -1)
        offsets_normalized = torch.cat(
            (
                offset_x_reference / reference_w,
                offset_y_reference / reference_h,
            ),
            dim=-1,
        )
        means_normalized = (centers_normalized + offsets_normalized).clamp(0.0, 1.0)
        means_output = torch.stack(
            (
                means_normalized[..., 0] * output_w - 0.5,
                means_normalized[..., 1] * output_h - 0.5,
            ),
            dim=-1,
        )

        sigma_1_reference = base_sigma_reference * torch.exp(log_stretch)
        sigma_2_reference = base_sigma_reference * torch.exp(-log_stretch)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        var_1 = sigma_1_reference.square()
        var_2 = sigma_2_reference.square()
        var_x_reference = var_1 * cos_theta.square() + var_2 * sin_theta.square()
        var_y_reference = var_1 * sin_theta.square() + var_2 * cos_theta.square()
        cov_xy_reference = (var_1 - var_2) * sin_theta * cos_theta
        std_x_reference = torch.sqrt(var_x_reference)
        std_y_reference = torch.sqrt(var_y_reference)
        rho = cov_xy_reference / (
            std_x_reference * std_y_reference
        ).clamp_min(1e-12)
        rho = rho.clamp(-0.95, 0.95)
        std_output = torch.cat(
            (
                std_x_reference * (output_w / reference_w),
                std_y_reference * (output_h / reference_h),
            ),
            dim=-1,
        )

        spectral_residual = self.spectral_residual_head(embedding)
        spectral_residual = spectral_residual.permute(0, 2, 3, 1).contiguous().view(
            batch, reference_h * reference_w, self.num_bands
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_reference_px / reference_w,
                self.sigma_radius * self.std_max_reference_px / reference_h,
            ),
        )
        rendered_residual = self.rasterizer(
            opacity.float(),
            means_output.float(),
            std_output.float(),
            rho.float(),
            spectral_residual.float(),
            output_h,
            output_w,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        ).permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            axis_major = torch.maximum(sigma_1_reference, sigma_2_reference)
            axis_minor = torch.minimum(sigma_1_reference, sigma_2_reference)
            self.last_stats = {
                "primitive_count": float(reference_h * reference_w),
                "reference_height": float(reference_h),
                "reference_width": float(reference_w),
                "output_height": float(output_h),
                "output_width": float(output_w),
                "base_sigma_reference_px_mean": float(
                    base_sigma_reference.detach().mean()
                ),
                "axis_ratio_mean": float(
                    (axis_major / axis_minor.clamp_min(1e-12)).detach().mean()
                ),
                "offset_reference_px_abs_mean": float(
                    torch.cat(
                        (offset_x_reference, offset_y_reference), dim=-1
                    ).detach().abs().mean()
                ),
                "opacity_mean": float(opacity.detach().mean()),
                "spectral_residual_abs_mean": float(
                    spectral_residual.detach().abs().mean()
                ),
                "rendered_residual_abs_mean": float(
                    rendered_residual.detach().abs().mean()
                ),
                "normalized_coordinates": 1.0,
                "density_normalized": 0.0,
                "adaptive_window": 1.0,
                "sigma_radius": self.sigma_radius,
                "max_axis_ratio": self.max_axis_ratio,
            }

        if not return_aux:
            return rendered_residual
        return rendered_residual, {
            "means_normalized": means_normalized,
            "means_output": means_output,
            "std_output": std_output,
            "rho": rho,
            "opacity": opacity,
            "spectral_residual": spectral_residual,
        }


class GSFusion(nn.Module):
    """Continuous HSI residual operator inspired by GSPan."""

    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_msi = int(num_msi)
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        self.primitive_embedding = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.gaussian_residual = NormalizedContinuousGaussianResidual(
            dim=dim,
            num_bands=num_bands,
            max_axis_ratio=max_axis_ratio,
        )
        self.arch_summary = (
            "GSPan-inspired HSI-MSI continuous residual field; DIM80 dual ADCI "
            "Exact encoders; normalized learnable Gaussian centers and bounded "
            "elliptical covariance; adaptive-3sigma rendering at arbitrary "
            "output_size; bicubic HSI base; no fused reconstruction bypass"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        self.gaussian_residual.reset_custom_init()

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        del sf
        reference_size = tuple(map(int, hr_msi.shape[-2:]))
        query_size = reference_size if output_size is None else tuple(map(int, output_size))
        base = F.interpolate(
            lr_hsi, size=query_size, mode="bicubic", align_corners=False
        )
        hsi_feature = self.shallow_encoder1(lr_hsi)
        msi_feature = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            hsi_feature = layer(hsi_feature)
        for layer in self.adci_msi_layers:
            msi_feature = layer(msi_feature)
        hsi_reference = F.interpolate(
            hsi_feature, size=reference_size, mode="bicubic", align_corners=False
        )
        embedding = self.primitive_embedding(
            torch.cat((hsi_reference, msi_feature), dim=1)
        )
        residual, gaussian_aux = self.gaussian_residual(
            embedding, query_size, return_aux=True
        )
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {
            "base": base,
            "hsi_reference": hsi_reference,
            "msi_feature": msi_feature,
            "primitive_embedding": embedding,
            "gaussian_residual": residual,
            **gaussian_aux,
        }

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        if not self.gaussian_residual.last_stats:
            return []
        return [{"layer": "gspan_hsi_continuous", **self.gaussian_residual.last_stats}]


__all__ = [
    "GSFusion",
    "NormalizedContinuousGaussianResidual",
    "compute_loss",
    "sam_loss",
]
