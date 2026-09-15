"""E3 with a 64-channel stable trunk and an 80-channel Gaussian descriptor.

Only the dedicated primitive branch is widened.  The two ADCI encoders,
fusion latent, decoder, and rendered residual remain 64-channel.  This
separates Gaussian descriptor capacity from whole-backbone capacity.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ADCI_Exact import ADCIExact
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    HRAdaptiveGaussianResidual,
    _resolve_adaptive_gaussian_rasterizer,
    compute_loss,
    sam_loss,
)


class PrimitiveDescriptorEllipticalGaussian(nn.Module):
    """Predict bounded geometry from a wide descriptor and render a thin delta."""

    def __init__(
        self,
        descriptor_dim: int,
        output_dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        sigma_radius: float = 3.0,
        max_axis_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if max_axis_ratio < 1.0:
            raise ValueError("max_axis_ratio must be at least 1.0")
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(output_dim + 1)
        self.descriptor_dim = int(descriptor_dim)
        self.output_dim = int(output_dim)
        self.std_min_px = float(std_min_px)
        self.std_max_px = float(std_max_px)
        self.sigma_radius = float(sigma_radius)
        self.max_axis_ratio = float(max_axis_ratio)
        self.max_log_stretch = 0.5 * math.log(self.max_axis_ratio)

        self.geometry_head = nn.Sequential(
            nn.Conv2d(descriptor_dim, descriptor_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(descriptor_dim, 2, kernel_size=1),
        )
        self.anisotropy_head = nn.Conv2d(
            descriptor_dim, 2, kernel_size=1
        )
        self.residual_value_head = nn.Sequential(
            nn.Conv2d(descriptor_dim, descriptor_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(descriptor_dim, output_dim, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        nn.init.zeros_(self.residual_value_head[-1].weight)
        nn.init.zeros_(self.residual_value_head[-1].bias)
        # Start exactly circular; anisotropy must earn its contribution.
        nn.init.zeros_(self.anisotropy_head.weight)
        nn.init.zeros_(self.anisotropy_head.bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = descriptor.shape
        if channels != self.descriptor_dim:
            raise ValueError(
                f"descriptor channels={channels}, expected {self.descriptor_dim}"
            )

        circular_raw = self.geometry_head(descriptor)
        circular_raw = circular_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        opacity = 0.05 + 0.95 * torch.sigmoid(circular_raw[..., 0:1])

        shape_raw = self.anisotropy_head(descriptor)
        shape_raw = shape_raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )
        log_stretch = self.max_log_stretch * torch.tanh(shape_raw[..., 0:1])
        theta = 0.5 * math.pi * torch.tanh(shape_raw[..., 1:2])

        stretch_abs = log_stretch.abs()
        base_min = self.std_min_px * torch.exp(stretch_abs)
        base_max = self.std_max_px * torch.exp(-stretch_abs)
        base_sigma = base_min + (base_max - base_min) * torch.sigmoid(
            circular_raw[..., 1:2]
        )
        sigma_1 = base_sigma * torch.exp(log_stretch)
        sigma_2 = base_sigma * torch.exp(-log_stretch)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        var_1 = sigma_1.square()
        var_2 = sigma_2.square()
        var_x = var_1 * cos_theta.square() + var_2 * sin_theta.square()
        var_y = var_1 * sin_theta.square() + var_2 * cos_theta.square()
        cov_xy = (var_1 - var_2) * sin_theta * cos_theta
        std_x = torch.sqrt(var_x)
        std_y = torch.sqrt(var_y)
        rho = (cov_xy / (std_x * std_y).clamp_min(1e-12)).clamp(-0.95, 0.95)
        std_px = torch.cat([std_x, std_y], dim=-1)

        means_px = HRAdaptiveGaussianResidual._pixel_centers(
            height, width, descriptor.device, descriptor.dtype
        ).expand(batch, height * width, 2)

        delta_value = self.residual_value_head(descriptor)
        delta_value = delta_value.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, self.output_dim
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
        numerator = rasterized[:, : self.output_dim]
        density = rasterized[:, self.output_dim : self.output_dim + 1]
        gaussian_delta = numerator / density.clamp_min(1e-6)

        with torch.no_grad():
            axis_major = torch.maximum(sigma_1, sigma_2)
            axis_minor = torch.minimum(sigma_1, sigma_2)
            axis_ratio = axis_major / axis_minor.clamp_min(1e-12)
            descriptor_abs = descriptor.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self.last_stats = {
                "hrgs_descriptor_dim": float(self.descriptor_dim),
                "hrgs_output_dim": float(self.output_dim),
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_opacity_std": float(opacity.detach().std()),
                "hrgs_base_sigma_mean_px": float(base_sigma.detach().mean()),
                "hrgs_axis_major_mean_px": float(axis_major.detach().mean()),
                "hrgs_axis_minor_mean_px": float(axis_minor.detach().mean()),
                "hrgs_axis_ratio_mean": float(axis_ratio.detach().mean()),
                "hrgs_axis_ratio_max": float(axis_ratio.detach().max()),
                "hrgs_theta_abs_mean_deg": float(
                    theta.detach().abs().mean() * (180.0 / math.pi)
                ),
                "hrgs_std_x_mean_px": float(std_x.detach().mean()),
                "hrgs_std_y_mean_px": float(std_y.detach().mean()),
                "hrgs_rho_abs_mean": float(rho.detach().abs().mean()),
                "hrgs_offset_abs_mean_px": 0.0,
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(delta_value.detach().abs().mean()),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_descriptor_abs_mean": float(descriptor_abs),
                "hrgs_delta_descriptor_ratio": float(
                    delta_abs / (descriptor_abs + 1e-8)
                ),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
                "hrgs_max_axis_ratio_limit": float(self.max_axis_ratio),
            }
        return gaussian_delta


class GSFusion(E3GSFusion):
    """E3 trunk64 plus a primitive80 constrained-ellipse Gaussian branch."""

    def __init__(
        self,
        dim: int = 64,
        primitive_dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        **kwargs: object,
    ) -> None:
        if int(dim) != 64:
            raise ValueError("This controlled candidate requires a DIM64 trunk")
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        # Keep ADCI's equations, parameter names, and tensor shapes unchanged;
        # only use the already-audited exact CUDA/Triton value aggregation.
        rng_state = torch.get_rng_state()
        self.adci_hsi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        torch.set_rng_state(rng_state)
        self.primitive_dim = int(primitive_dim)
        self.primitive_input = nn.Conv2d(2 * dim, primitive_dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(primitive_dim, primitive_dim, 1),
            nn.GELU(),
            nn.Conv2d(primitive_dim, primitive_dim, 1),
        )
        self.gaussian_refine = PrimitiveDescriptorEllipticalGaussian(
            descriptor_dim=primitive_dim,
            output_dim=dim,
            max_axis_ratio=max_axis_ratio,
        )
        self.arch_summary = (
            "E3 Primitive80 ConstrainedEllipse: both exact-CUDA ADCI encoders, "
            "fusion and decoder remain DIM64; only the dedicated primitive "
            "descriptor is DIM80; fixed centers; area-preserving axis ratio "
            "<=2; the density-normalized adaptive-3sigma rasterizer emits a "
            "DIM64 delta"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
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
        f_hsi_hr = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        joint = torch.cat([f_hsi_hr, f_msi], dim=1)
        fused = self.conv0(joint)
        primitive_base = self.primitive_input(joint)
        primitive = primitive_base + self.primitive_residual(primitive_base)
        gaussian_delta = self.gaussian_refine(primitive)
        if gaussian_delta.shape != fused.shape:
            raise RuntimeError(
                f"Gaussian delta shape {gaussian_delta.shape} != fused {fused.shape}"
            )
        refined = fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = [
    "GSFusion",
    "PrimitiveDescriptorEllipticalGaussian",
    "compute_loss",
    "sam_loss",
]
