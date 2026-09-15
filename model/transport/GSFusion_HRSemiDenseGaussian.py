"""E3 with a fixed-HR semi-dense circular Gaussian field."""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


def _resolve_adaptive_gaussian_rasterizer():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extension_root = os.environ.get(
        "GSFUSION_ADAPTIVE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "adaptive3_rasterizer"),
    )
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)
    from diff_srgaussian_rasterization import GaussianRasterizer

    return GaussianRasterizer


def _gradient_l2(tensors) -> float:
    squared_sum = 0.0
    for tensor in tensors:
        if tensor is not None:
            squared_sum += float(tensor.detach().double().square().sum().cpu())
    return math.sqrt(squared_sum)


class HRSemiDenseGaussianTransport(nn.Module):
    """Rasterize one Gaussian per fixed-stride HR anchor cell."""

    def __init__(
        self,
        dim: int,
        anchor_stride_hr: int,
        std_min_cell: float = 0.125,
        std_max_cell: float = 1.0,
        std_init_cell: float = 0.5,
        sigma_radius: float = 3.0,
        density_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if anchor_stride_hr not in (2, 4):
            raise ValueError("anchor_stride_hr must be 2 or 4")
        if not (std_min_cell < std_init_cell < std_max_cell):
            raise ValueError("std_init_cell must lie strictly inside the std range")
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
        self.rasterizer = GaussianRasterizer(dim + 1)
        self.dim = int(dim)
        self.anchor_stride_hr = int(anchor_stride_hr)
        self.std_min_cell = float(std_min_cell)
        self.std_max_cell = float(std_max_cell)
        self.std_init_cell = float(std_init_cell)
        self.sigma_radius = float(sigma_radius)
        self.density_eps = float(density_eps)
        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, 2, 1)
        )
        self.residual_value_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_custom_init(self) -> None:
        fraction = (self.std_init_cell - self.std_min_cell) / (
            self.std_max_cell - self.std_min_cell
        )
        std_logit = math.log(fraction / (1.0 - fraction))
        with torch.no_grad():
            self.geometry_head[-1].weight[1].zero_()
            self.geometry_head[-1].bias[1] = std_logit
        nn.init.zeros_(self.residual_value_head[-1].weight)
        nn.init.zeros_(self.residual_value_head[-1].bias)

    def _anchor_centers(self, height: int, width: int, device, dtype):
        stride = self.anchor_stride_hr
        y = (torch.arange(height // stride, device=device, dtype=dtype) + 0.5) * stride - 0.5
        x = (torch.arange(width // stride, device=device, dtype=dtype) + 0.5) * stride - 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).view(1, -1, 2)

    def forward(
        self,
        primitive: torch.Tensor,
        out_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        batch, channels, anchor_height, anchor_width = primitive.shape
        height, width = int(out_size[0]), int(out_size[1])
        stride = self.anchor_stride_hr
        if height % stride or width % stride:
            raise ValueError(
                f"HR size {(height, width)} must be divisible by stride {stride}"
            )
        expected = (height // stride, width // stride)
        if (anchor_height, anchor_width) != expected:
            raise ValueError(
                f"primitive grid {(anchor_height, anchor_width)} != expected {expected}"
            )

        raw = self.geometry_head(primitive).permute(0, 2, 3, 1).contiguous()
        raw = raw.view(batch, anchor_height * anchor_width, 2)
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_cell = self.std_min_cell + (
            self.std_max_cell - self.std_min_cell
        ) * torch.sigmoid(raw[..., 1:2])
        std_px_scalar = std_cell * stride
        std_px = std_px_scalar.expand(-1, -1, 2)
        means_px = self._anchor_centers(
            height, width, primitive.device, primitive.dtype
        ).expand(batch, -1, -1)
        rho = means_px.new_zeros(batch, anchor_height * anchor_width, 1)

        value = self.residual_value_head(primitive)
        value = value.permute(0, 2, 3, 1).contiguous().view(
            batch, anchor_height * anchor_width, channels
        )
        values_with_density = torch.cat(
            (value, value.new_ones(batch, anchor_height * anchor_width, 1)), dim=-1
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_cell / max(anchor_width, 1),
                self.sigma_radius * self.std_max_cell / max(anchor_height, 1),
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
        density = rasterized[:, channels : channels + 1]
        gaussian_delta = numerator / density.clamp_min(self.density_eps)

        with torch.no_grad():
            density_detached = density.detach()
            stats = {
                "anchor_stride_hr": float(stride),
                "primitive_grid_height": float(anchor_height),
                "primitive_grid_width": float(anchor_width),
                "primitive_count": float(anchor_height * anchor_width),
                "std_cell_min": float(std_cell.detach().min()),
                "std_cell_mean": float(std_cell.detach().mean()),
                "std_cell_max": float(std_cell.detach().max()),
                "std_cell_std": float(std_cell.detach().std()),
                "std_px_min": float(std_px_scalar.detach().min()),
                "std_px_mean": float(std_px_scalar.detach().mean()),
                "std_px_max": float(std_px_scalar.detach().max()),
                "std_px_std": float(std_px_scalar.detach().std()),
                "opacity_min": float(opacity.detach().min()),
                "opacity_mean": float(opacity.detach().mean()),
                "opacity_max": float(opacity.detach().max()),
                "opacity_std": float(opacity.detach().std()),
                "value_abs_mean": float(value.detach().abs().mean()),
                "value_norm_mean": float(
                    torch.linalg.vector_norm(value.detach(), dim=-1).mean()
                ),
                "gaussian_delta_abs_mean": float(gaussian_delta.detach().abs().mean()),
                "density_min": float(density_detached.min()),
                "density_mean": float(density_detached.mean()),
                "density_max": float(density_detached.max()),
                "zero_density_ratio": float(
                    (density_detached <= self.density_eps).float().mean()
                ),
                "adaptive_window": 1.0,
                "sigma_radius": self.sigma_radius,
                "raster_ratio": float(raster_ratio),
            }
            if not self.training:
                flattened_density = density_detached.float().reshape(-1)
                stats["density_p001"] = float(
                    torch.quantile(flattened_density, 0.001)
                )
                stats["density_p01"] = float(
                    torch.quantile(flattened_density, 0.01)
                )
            self.last_stats = stats

        if not return_aux:
            return gaussian_delta
        return gaussian_delta, {
            "opacity": opacity,
            "std_cell": std_cell,
            "std_px": std_px,
            "means_px": means_px,
            "density": density,
            "value": value,
        }


class GSFusion(nn.Module):
    """E3 backbone with a fixed-HR PixelUnshuffle primitive grid."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        anchor_stride_hr: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        if anchor_stride_hr not in (2, 4):
            raise ValueError("anchor_stride_hr must be 2 or 4")
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_msi = int(num_msi)
        self.anchor_stride_hr = int(anchor_stride_hr)

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.gaussian_transport = HRSemiDenseGaussianTransport(
            dim, anchor_stride_hr=anchor_stride_hr
        )
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        extra_rng_state = torch.get_rng_state()
        footprint_channels = dim * anchor_stride_hr * anchor_stride_hr
        self.hsi_foot_embed = nn.Sequential(
            nn.Conv2d(footprint_channels, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.msi_foot_embed = nn.Sequential(
            nn.Conv2d(footprint_channels, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.primitive_fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        torch.set_rng_state(extra_rng_state)
        self._last_stats: Dict[str, float] = {}
        self._last_gradient_stats: Dict[str, float] = {}
        self.arch_summary = (
            f"E3 fixed-HR semi-dense circular Gaussian; stride={anchor_stride_hr}; "
            "PixelUnshuffle footprint encoding; fixed cell-center anchors; "
            "cell-unit std; density-normalized adaptive 3-sigma CUDA rasterization"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_transport.reset_custom_init()

    def collect_gradient_stats(self) -> Dict[str, float]:
        geometry_last = self.gaussian_transport.geometry_head[-1]
        weight_grad = geometry_last.weight.grad
        bias_grad = geometry_last.bias.grad
        opacity_grads = [] if weight_grad is None else [weight_grad[0]]
        std_grads = [] if weight_grad is None else [weight_grad[1]]
        if bias_grad is not None:
            opacity_grads.append(bias_grad[0])
            std_grads.append(bias_grad[1])
        value_grads = [
            parameter.grad
            for parameter in self.gaussian_transport.residual_value_head.parameters()
        ]
        primitive_grads = [
            parameter.grad
            for module in (
                self.hsi_foot_embed,
                self.msi_foot_embed,
                self.primitive_fuse,
            )
            for parameter in module.parameters()
        ]
        grad_std = _gradient_l2(std_grads)
        grad_opacity = _gradient_l2(opacity_grads)
        grad_value = _gradient_l2(value_grads)
        self._last_gradient_stats = {
            "grad_std_head": grad_std,
            "grad_opacity_head": grad_opacity,
            "grad_value_head": grad_value,
            "grad_primitive_embed": _gradient_l2(primitive_grads),
            "grad_std_over_value": grad_std / max(grad_value, 1e-30),
            "grad_opacity_over_value": grad_opacity / max(grad_value, 1e-30),
        }
        return dict(self._last_gradient_stats)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = dict(self.gaussian_transport.last_stats or {})
        stats.update(self._last_stats)
        stats.update(self._last_gradient_stats)
        return [] if not stats else [{"layer": "hr_semidense", **stats}]

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
    ):
        input_sf = int(sf) if sf is not None else int(
            round(hr_msi.shape[-1] / lr_hsi.shape[-1])
        )
        target_size = hr_msi.shape[-2:]
        height, width = target_size
        stride = self.anchor_stride_hr
        if height % stride or width % stride:
            raise ValueError(
                f"HR size {(height, width)} must be divisible by stride {stride}"
            )
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        f_h = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_m = f_msi
        joint = torch.cat((f_h, f_m), dim=1)
        f_fused = self.conv0(joint)

        h_foot = F.pixel_unshuffle(f_h, downscale_factor=stride)
        m_foot = F.pixel_unshuffle(f_m, downscale_factor=stride)
        h_primitive = self.hsi_foot_embed(h_foot)
        m_primitive = self.msi_foot_embed(m_foot)
        primitive = self.primitive_fuse(
            torch.cat((h_primitive, m_primitive), dim=1)
        )
        gaussian_delta, gs_aux = self.gaussian_transport(
            primitive, target_size, return_aux=True
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual

        with torch.no_grad():
            fused_abs = f_fused.detach().abs().mean()
            delta_abs = gaussian_delta.detach().abs().mean()
            self._last_stats = {
                "input_sf": float(input_sf),
                "hr_height": float(height),
                "hr_width": float(width),
                "f_fused_abs_mean": float(fused_abs),
                "gaussian_delta_over_f_fused": float(
                    delta_abs / (fused_abs + 1e-8)
                ),
            }

        if not return_aux:
            return prediction
        return prediction, {
            "base": base,
            "F_H": f_h,
            "F_M": f_m,
            "F_fused": f_fused,
            "h_foot": h_foot,
            "m_foot": m_foot,
            "primitive": primitive,
            "gaussian_delta": gaussian_delta,
            "residual": residual,
            **gs_aux,
        }


__all__ = [
    "GSFusion",
    "HRSemiDenseGaussianTransport",
    "compute_loss",
    "sam_loss",
]
