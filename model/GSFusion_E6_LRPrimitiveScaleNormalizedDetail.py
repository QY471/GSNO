"""Reference-coordinate Gaussian detail transport for HSI-MSI fusion.

The field is defined in native HR-MSI pixel coordinates, not in a per-tensor
[-1, 1] canvas.  LR-HSI features are continuously sampled onto a fixed native
anchor lattice, while the same field can be queried on another output grid.
Changing the input resolution ratio therefore does not change primitive
density or sigma, and changing output sampling density does not redefine the
underlying field.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss
from model.important_model_support.GSFusion_LRPrimitiveTransportCommon import (
    MSIGuidedHSILocalRouting,
    ScaleSharedMSIFootprintSampler,
    _resolve_adaptive_gaussian_rasterizer,
)


class ReferenceGridAnchorSampler(nn.Module):
    """Sample any feature field at a fixed HR-MSI reference lattice."""

    def __init__(self, anchor_stride_hr: int = 2) -> None:
        super().__init__()
        if anchor_stride_hr <= 0:
            raise ValueError("anchor_stride_hr must be positive")
        self.anchor_stride_hr = int(anchor_stride_hr)

    def anchor_shape(self, reference_size: Tuple[int, int]) -> Tuple[int, int]:
        height, width = int(reference_size[0]), int(reference_size[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"reference_size must be positive, got {(height, width)}")
        stride = self.anchor_stride_hr
        return (height + stride - 1) // stride, (width + stride - 1) // stride

    def centers(
        self, reference_size: Tuple[int, int], device, dtype
    ) -> torch.Tensor:
        anchor_h, anchor_w = self.anchor_shape(reference_size)
        stride = self.anchor_stride_hr
        y = (torch.arange(anchor_h, device=device, dtype=dtype) + 0.5) * stride - 0.5
        x = (torch.arange(anchor_w, device=device, dtype=dtype) + 0.5) * stride - 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1)

    def forward(
        self, feature: torch.Tensor, reference_size: Tuple[int, int]
    ) -> torch.Tensor:
        batch = feature.shape[0]
        height, width = int(reference_size[0]), int(reference_size[1])
        centers = self.centers(reference_size, feature.device, feature.dtype)
        grid = torch.stack(
            (
                2.0 * (centers[..., 0] + 0.5) / width - 1.0,
                2.0 * (centers[..., 1] + 0.5) / height - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0).expand(batch, -1, -1, -1)
        return F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


class ScaleNormalizedDetailGaussianTransport(nn.Module):
    """Render a reference-coordinate detail field on any output grid."""

    def __init__(
        self,
        dim: int,
        anchor_stride_hr: int = 2,
        std_min_hr: float = 0.55,
        std_max_hr: float = 1.50,
        std_init_hr: float = 1.00,
        sigma_radius: float = 3.0,
        density_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        if anchor_stride_hr <= 0:
            raise ValueError("anchor_stride_hr must be positive")
        if not (std_min_hr < std_init_hr < std_max_hr):
            raise ValueError("std_init_hr must lie strictly inside the HR-pixel range")
        self.anchor_stride_hr = int(anchor_stride_hr)
        self.std_min_hr = float(std_min_hr)
        self.std_max_hr = float(std_max_hr)
        self.std_init_hr = float(std_init_hr)
        self.sigma_radius = float(sigma_radius)
        self.density_eps = float(density_eps)
        self.rasterizer = _resolve_adaptive_gaussian_rasterizer()(dim + 1)
        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, 2, 1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        fraction = (self.std_init_hr - self.std_min_hr) / (
            self.std_max_hr - self.std_min_hr
        )
        self.geometry_head[-1].bias.data[1] = math.log(fraction / (1.0 - fraction))
        nn.init.zeros_(self.value_head[-1].weight)

    def forward(
        self,
        primitive: torch.Tensor,
        reference_size: Tuple[int, int],
        out_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        batch, channels, anchor_h, anchor_w = primitive.shape
        reference_h, reference_w = int(reference_size[0]), int(reference_size[1])
        output_h, output_w = int(out_size[0]), int(out_size[1])
        stride = self.anchor_stride_hr
        if reference_h <= 0 or reference_w <= 0 or output_h <= 0 or output_w <= 0:
            raise ValueError(
                f"reference_size and out_size must be positive, got "
                f"{reference_size} and {out_size}"
            )
        expected_shape = (
            (reference_h + stride - 1) // stride,
            (reference_w + stride - 1) // stride,
        )
        if (anchor_h, anchor_w) != expected_shape:
            raise ValueError(
                f"primitive grid {(anchor_h, anchor_w)} != {expected_shape}"
            )
        raw = self.geometry_head(primitive).permute(0, 2, 3, 1).contiguous()
        raw = raw.view(batch, anchor_h * anchor_w, 2)
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_reference = self.std_min_hr + (
            self.std_max_hr - self.std_min_hr
        ) * torch.sigmoid(raw[..., 1:2])
        scale_x = output_w / reference_w
        scale_y = output_h / reference_h
        std_output = torch.cat(
            (std_reference * scale_x, std_reference * scale_y), dim=-1
        )
        value = self.value_head(primitive)
        value = value.permute(0, 2, 3, 1).contiguous().view(
            batch, anchor_h * anchor_w, channels
        )
        values_with_density = torch.cat(
            (value, value.new_ones(batch, anchor_h * anchor_w, 1)), dim=-1
        )
        sampler = ReferenceGridAnchorSampler(stride)
        means_reference = sampler.centers(
            reference_size, primitive.device, primitive.dtype
        ).view(1, anchor_h * anchor_w, 2).expand(batch, -1, -1)
        means_output = torch.stack(
            (
                (means_reference[..., 0] + 0.5) * scale_x - 0.5,
                (means_reference[..., 1] + 0.5) * scale_y - 0.5,
            ),
            dim=-1,
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_hr / max(reference_w, 1),
                self.sigma_radius * self.std_max_hr / max(reference_h, 1),
            ),
        )
        density_and_value = self.rasterizer(
            opacity.float(),
            means_output.float(),
            std_output.float(),
            std_output.new_zeros(batch, anchor_h * anchor_w, 1).float(),
            values_with_density.float(),
            output_h,
            output_w,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        ).permute(0, 3, 1, 2).contiguous()
        numerator = density_and_value[:, :channels]
        density = density_and_value[:, channels : channels + 1]
        delta = numerator / density.clamp_min(self.density_eps)
        with torch.no_grad():
            self.last_stats = {
                "anchor_stride_hr": float(stride),
                "primitive_count": float(anchor_h * anchor_w),
                "reference_height": float(reference_h),
                "reference_width": float(reference_w),
                "output_height": float(output_h),
                "output_width": float(output_w),
                "output_scale_x": float(scale_x),
                "output_scale_y": float(scale_y),
                "std_reference_mean": float(std_reference.detach().mean()),
                "std_output_x_mean": float(std_output[..., 0].detach().mean()),
                "std_output_y_mean": float(std_output[..., 1].detach().mean()),
                "opacity_mean": float(opacity.detach().mean()),
                "density_mean": float(density.detach().mean()),
                "density_min": float(density.detach().min()),
                "low_density_ratio": float((density.detach() < 1e-4).float().mean()),
                "value_abs_mean": float(value.detach().abs().mean()),
                "delta_abs_mean": float(delta.detach().abs().mean()),
            }
        if not return_aux:
            return delta
        return delta, {
            "density": density,
            "std_reference": std_reference,
            "std_output": std_output,
            "means_reference": means_reference,
            "means_output": means_output,
            "opacity": opacity,
            "value": value,
        }


class GSFusion(nn.Module):
    """E6 trunk plus a native-reference continuous Gaussian detail field."""

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
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])
        self.adci_msi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])
        self.conv0 = nn.Sequential(nn.Conv2d(2 * dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)
        self.anchor_sampler = ReferenceGridAnchorSampler(anchor_stride_hr)
        self.msi_footprint_sampler = ScaleSharedMSIFootprintSampler()
        self.msi_footprint_embed = nn.Sequential(
            nn.Conv2d(16 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.primitive_input = nn.Conv2d(2 * dim, dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.routing = MSIGuidedHSILocalRouting(dim, routing_dim=dim // 4)
        self.detail_direction_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.msi_contrast_gate = nn.Conv2d(dim, dim, 1, bias=False)
        self.gaussian_transport = ScaleNormalizedDetailGaussianTransport(
            dim, anchor_stride_hr=anchor_stride_hr
        )
        self._last_stats: Dict[str, float] = {}
        self.arch_summary = (
            "LR Primitive Scale-Normalized Detail v2: E6 ADCI/fusion trunk; "
            "native HR-MSI reference coordinates, fixed-stride anchor lattice, "
            "fixed reference-pixel sigma, and optional continuous output query; Gaussian "
            "values retain E6 primitive embedding and HSI routing as their spectral "
            "direction, but must be multiplicatively gated by fixed-HR local MSI "
            "contrast; normalized scatter "
            "adds a detail residual to the stable fused latent"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
        self.routing.reset_output_init()
        self.gaussian_transport.reset_custom_init()

    @staticmethod
    def fixed_hr_local_reference(feature: torch.Tensor) -> torch.Tensor:
        """A fixed 5x5 HR-pixel reference, independent of LR-HSI scale."""
        padded = F.pad(feature, (2, 2, 2, 2), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=5, stride=1)

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        del sf
        reference_size = tuple(int(v) for v in hr_msi.shape[-2:])
        query_size = (
            reference_size
            if output_size is None
            else (int(output_size[0]), int(output_size[1]))
        )
        if query_size[0] <= 0 or query_size[1] <= 0:
            raise ValueError(f"output_size must be positive, got {query_size}")
        base = F.interpolate(
            lr_hsi, size=query_size, mode="bicubic", align_corners=False
        )
        h_lr = self.shallow_encoder1(lr_hsi)
        f_m = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            h_lr = layer(h_lr)
        for layer in self.adci_msi_layers:
            f_m = layer(f_m)
        f_h = F.interpolate(
            h_lr, size=reference_size, mode="bicubic", align_corners=False
        )
        f_fused_reference = self.conv0(torch.cat((f_h, f_m), dim=1))

        footprint = self.msi_footprint_sampler(f_m, h_lr.shape[-2:])
        m_lr = self.msi_footprint_embed(footprint)
        e0_lr = self.primitive_input(torch.cat((h_lr, m_lr), dim=1))
        e_g_lr = e0_lr + self.primitive_residual(e0_lr)
        route_lr = self.routing(h_lr, m_lr)
        detail_direction_lr = self.detail_direction_proj(e_g_lr + route_lr)
        detail_direction = self.anchor_sampler(
            detail_direction_lr, reference_size
        )

        m_anchor = self.anchor_sampler(f_m, reference_size)
        local_reference = self.fixed_hr_local_reference(f_m)
        m_reference_anchor = self.anchor_sampler(
            local_reference, reference_size
        )
        m_contrast = m_anchor - m_reference_anchor
        gate = torch.tanh(self.msi_contrast_gate(m_contrast))
        primitive = detail_direction * gate
        gaussian_delta, aux = self.gaussian_transport(
            primitive,
            reference_size=reference_size,
            out_size=query_size,
            return_aux=True,
        )
        f_fused = (
            f_fused_reference
            if query_size == reference_size
            else F.interpolate(
                f_fused_reference,
                size=query_size,
                mode="bicubic",
                align_corners=False,
            )
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        with torch.no_grad():
            self._last_stats = {
                "msi_contrast_abs_mean": float(m_contrast.detach().abs().mean()),
                "primitive_abs_mean": float(primitive.detach().abs().mean()),
                "gaussian_delta_abs_mean": float(gaussian_delta.detach().abs().mean()),
                "gaussian_to_fused_ratio": float(gaussian_delta.detach().abs().mean() / (f_fused.detach().abs().mean() + 1e-8)),
                "reference_height": float(reference_size[0]),
                "reference_width": float(reference_size[1]),
                "query_height": float(query_size[0]),
                "query_width": float(query_size[1]),
            }
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {
            "H_lr": h_lr,
            "F_M": f_m,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "M_lr": m_lr,
            "E_g_lr": e_g_lr,
            "route_lr": route_lr,
            "detail_direction_lr": detail_direction_lr,
            "detail_direction": detail_direction,
            "m_anchor": m_anchor,
            "m_reference_anchor": m_reference_anchor,
            "m_contrast": m_contrast,
            "primitive": primitive,
            "gaussian_delta": gaussian_delta,
            **aux,
        }

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = dict(self.gaussian_transport.last_stats or {})
        stats.update(self._last_stats)
        return [] if not stats else [{"layer": "reference_detail_field", **stats}]


__all__ = [
    "GSFusion",
    "ReferenceGridAnchorSampler",
    "ScaleNormalizedDetailGaussianTransport",
    "compute_loss",
    "sam_loss",
]
