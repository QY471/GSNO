"""Gaussian Spatial-Spectral Neural Operator.

The local-interaction implementation is adapted from AFNO; see
third_party/README.md for provenance.
"""

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_adaptive_gaussian_rasterizer():
    """Load the isolated adaptive-window extension bundled for this model."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extension_root = os.environ.get(
        "GSFUSION_ADAPTIVE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "adaptive3_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        out = (x - mean) / (std + self.eps)
        return self.weight * out + self.bias


class LKI(nn.Module):
    """Local Kernel Interaction between center and neighboring features."""

    def __init__(self, in_channels, mlp_hidden_dim):
        super().__init__()
        self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, in_channels),
        )
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)

        k_unfold = F.unfold(k, kernel_size=3, padding=1).view(b, c, 9, h, w)
        v_unfold = F.unfold(v, kernel_size=3, padding=1).view(b, c, 9, h, w)

        q_expanded = q.unsqueeze(2)
        q_minus_k = q_expanded - k_unfold

        q_minus_k = q_minus_k.permute(0, 3, 4, 2, 1).contiguous()
        mlp_output = self.mlp(q_minus_k)
        attention_scores = F.softmax(mlp_output, dim=-2)

        neighbors_v = v_unfold.permute(0, 3, 4, 2, 1).contiguous()
        weighted_v = torch.sum(neighbors_v * attention_scores, dim=3)
        weighted_v = weighted_v.permute(0, 3, 1, 2).contiguous()

        return weighted_v + self.gate(x)


_LEGACY_MODULE_NAMES = {
    "adci_hsi_layers": "lki_hsi_layers",
    "adci_msi_layers": "lki_msi_layers",
    "gaussian_refine": "gsio",
}


def _rename_legacy_state_key(key: str) -> str:
    return ".".join(_LEGACY_MODULE_NAMES.get(part, part) for part in key.split("."))


def remap_legacy_state_dict(state_dict):
    """Accept checkpoints saved before the public LKI/GSIO module names."""
    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = _rename_legacy_state_key(key)
        if new_key in remapped:
            raise ValueError(f"Duplicate model parameter after renaming: {new_key}")
        remapped[new_key] = value
    if hasattr(state_dict, "_metadata"):
        remapped._metadata = OrderedDict(
            (_rename_legacy_state_key(key), value)
            for key, value in state_dict._metadata.items()
        )
    return remapped


class GaussianResidual(nn.Module):
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
            nn.Conv2d(dim, dim, kernel_size=1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        raw = self.geometry_head(x)
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
            height, width, x.device, x.dtype
        ).expand(batch, height * width, 2)
        means_px = base_px + offset_px
        means_px = torch.stack(
            [
                means_px[..., 0].clamp(0.0, float(width - 1)),
                means_px[..., 1].clamp(0.0, float(height - 1)),
            ],
            dim=-1,
        )

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
            x_abs = x.detach().abs().mean()
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


class FusionBackbone(nn.Module):
    """Two-stream feature fusion and Gaussian residual reconstruction."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        lki_layers: int = 3,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.lki_hsi_layers = nn.ModuleList(
            [LKI(dim, dim) for _ in range(lki_layers)]
        )
        self.lki_msi_layers = nn.ModuleList(
            [LKI(dim, dim) for _ in range(lki_layers)]
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.gsio = GaussianResidual(dim)
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)
        # Preserve the RNG sequence used by the training initialization.
        extra_rng_state = torch.get_rng_state()
        self.primitive_input = nn.Conv2d(2 * dim, dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        torch.set_rng_state(extra_rng_state)

        self.arch_summary = (
            "GSNO: F_H=upsampled HSI latent; F_M=MSI latent; "
            "F=original fusion(concat(F_H,F_M)); E0=Conv1x1(concat); "
            "E_g=E0+Conv1x1(GELU(Conv1x1(E0))); primitive heads read E_g; "
            "circular normalized delta is added to F; original decoder"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gsio.reset_residual_init()
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = self.gsio.last_stats
        return [] if stats is None else [{"layer": "hr_gaussian", **stats}]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(
            remap_legacy_state_dict(state_dict), strict=strict, assign=assign
        )

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.lki_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.lki_msi_layers:
            f_msi = layer(f_msi)

        F_H = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        F_M = f_msi
        joint = torch.cat([F_H, F_M], dim=1)
        F_fused = self.conv0(joint)
        E0 = self.primitive_input(joint)
        E_g = E0 + self.primitive_residual(E0)
        primitive_with_delta = self.gsio(E_g)
        gaussian_delta = primitive_with_delta - E_g
        refined = F_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


class GSIO(GaussianResidual):
    """Gaussian Spatial Integral Operator with learned kernel orientation."""

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
        # sigma_1 / sigma_2 = exp(2 * log_stretch).
        self.max_log_stretch = 0.5 * math.log(self.max_axis_ratio)

        # Predict log-stretch and orientation at each location.
        self.anisotropy_head = nn.Conv2d(dim, 2, kernel_size=1)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        super().reset_residual_init()
        # Initialize isotropic kernels.
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

        # Keep both axes within the pixel-scale bounds while preserving
        # sigma_1 * sigma_2 = base_sigma**2.
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


class GSNO(FusionBackbone):
    """Spatial-spectral fusion with elliptical Gaussian integration."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        lki_layers: Optional[int] = None,
        adci_layers: Optional[int] = None,
        max_axis_ratio: float = 2.0,
        **kwargs: object,
    ) -> None:
        if lki_layers is None:
            lki_layers = 3 if adci_layers is None else adci_layers
        elif adci_layers is not None and lki_layers != adci_layers:
            raise ValueError("lki_layers and legacy adci_layers disagree")
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            lki_layers=lki_layers,
            **kwargs,
        )

        # Preserve the RNG sequence for subsequent layer initialization.
        rng_state = torch.get_rng_state()
        self.gsio = GSIO(
            dim=dim,
            max_axis_ratio=max_axis_ratio,
        )
        torch.set_rng_state(rng_state)

        self.arch_summary = (
            "GSNO constrained elliptical Gaussian: DIM-configurable LKI and "
            "primitive embedding unchanged; fixed HR-pixel center; bounded "
            "area-preserving principal axes with learned orientation; "
            "axis ratio <= 2; density-normalized adaptive-3sigma scatter"
        )
        self.reset_custom_init()


def sam_loss(pred, gt, eps=1e-8):
    cos = (pred * gt).sum(dim=1) / (pred.norm(dim=1) * gt.norm(dim=1) + eps)
    return (1.0 - cos).mean()


def compute_loss(pred, gt, epoch, sam_warmup_epochs=5, sam_weight=0.1):
    l1 = F.l1_loss(pred, gt)
    if epoch < sam_warmup_epochs:
        return l1
    w = min(sam_weight, sam_weight * (epoch - sam_warmup_epochs + 1) / 5.0)
    return l1 + w * sam_loss(pred, gt)


GSFusion = GSNO

__all__ = [
    "GSNO", "GSFusion", "LKI", "GSIO", "compute_loss", "sam_loss",
    "remap_legacy_state_dict",
]
