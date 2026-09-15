"""LR-cell Gaussian transport whose value is restricted to MSI contrast detail.

This is a research candidate for the old LR Primitive Transport failure mode:
the Gaussian branch may only transport a per-cell *relative MSI detail* signal,
not a complete fused latent.  It intentionally keeps the E6 ADCI/fusion trunk
and the density-normalized adaptive-3sigma CUDA renderer.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss
from model.important_model_support.GSFusion_LRPrimitiveTransportCommon import (
    _resolve_adaptive_gaussian_rasterizer,
)


class RelativeMSISubcellSampler(nn.Module):
    """Sample four normalized subcell locations inside every LR-HSI cell."""

    def __init__(self) -> None:
        super().__init__()
        offsets = torch.tensor(
            [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)],
            dtype=torch.float32,
        )
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, feature: torch.Tensor, lr_size: Tuple[int, int]) -> torch.Tensor:
        batch, channels, _height, _width = feature.shape
        h, w = int(lr_size[0]), int(lr_size[1])
        dtype, device = feature.dtype, feature.device
        cy = torch.arange(h, device=device, dtype=dtype) + 0.5
        cx = torch.arange(w, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        offsets = self.offsets.to(device=device, dtype=dtype)
        py = gy.unsqueeze(0) + offsets[:, 0].view(4, 1, 1)
        px = gx.unsqueeze(0) + offsets[:, 1].view(4, 1, 1)
        grid = torch.stack((2.0 * px / w - 1.0, 2.0 * py / h - 1.0), dim=-1)
        grid = grid.view(1, 4 * h, w, 2).expand(batch, -1, -1, -1)
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.view(batch, channels, 4, h, w)


class DetailSubcellGaussianTransport(nn.Module):
    """One Gaussian per 2x2 LR subcell, directly rasterized onto HR."""

    def __init__(
        self,
        dim: int,
        std_min_cell: float = 0.125,
        std_max_cell: float = 0.5,
        std_init_cell: float = 0.25,
        sigma_radius: float = 3.0,
        density_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.std_min_cell = float(std_min_cell)
        self.std_max_cell = float(std_max_cell)
        self.std_init_cell = float(std_init_cell)
        self.sigma_radius = float(sigma_radius)
        self.density_eps = float(density_eps)
        self.rasterizer = _resolve_adaptive_gaussian_rasterizer()(dim + 1)
        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, 2, 1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.register_buffer(
            "sub_offsets",
            torch.tensor(
                [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)
        fraction = (self.std_init_cell - self.std_min_cell) / (
            self.std_max_cell - self.std_min_cell
        )
        self.geometry_head[-1].bias.data[1] = math.log(fraction / (1.0 - fraction))
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def _centers(
        self, h: int, w: int, height: int, width: int, device, dtype
    ) -> torch.Tensor:
        offsets = self.sub_offsets.to(device=device, dtype=dtype)
        scale_y, scale_x = height / h, width / w
        cy = torch.arange(h, device=device, dtype=dtype) + 0.5
        cx = torch.arange(w, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(cy, cx, indexing="ij")
        centers = []
        for offset_y, offset_x in offsets:
            y = (gy + offset_y) * scale_y - 0.5
            x = (gx + offset_x) * scale_x - 0.5
            centers.append(torch.stack((x, y), dim=-1).reshape(-1, 2))
        return torch.cat(centers, dim=0).unsqueeze(0)

    def forward(
        self,
        primitive: torch.Tensor,
        out_size: Tuple[int, int],
        return_aux: bool = False,
    ):
        # primitive is B,C,4,h,w: the four entries carry only MSI-relative detail.
        batch, channels, subcells, h, w = primitive.shape
        if subcells != 4:
            raise ValueError(f"expected four subcells, got {subcells}")
        height, width = int(out_size[0]), int(out_size[1])
        flat = primitive.permute(0, 2, 1, 3, 4).reshape(batch * 4, channels, h, w)
        raw = self.geometry_head(flat).view(batch, 4, 2, h, w)
        raw = raw.permute(0, 1, 3, 4, 2).reshape(batch, 4 * h * w, 2)
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])
        std_cell = self.std_min_cell + (
            self.std_max_cell - self.std_min_cell
        ) * torch.sigmoid(raw[..., 1:2])
        scale_x, scale_y = width / w, height / h
        std_hr = torch.cat((std_cell * scale_x, std_cell * scale_y), dim=-1)
        value = self.value_head(flat).view(batch, 4, channels, h, w)
        value = value.permute(0, 1, 3, 4, 2).reshape(batch, 4 * h * w, channels)
        values_with_density = torch.cat(
            (value, value.new_ones(batch, 4 * h * w, 1)), dim=-1
        )
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_cell / max(w, 1),
                self.sigma_radius * self.std_max_cell / max(h, 1),
            ),
        )
        density_and_value = self.rasterizer(
            opacity.float(),
            self._centers(h, w, height, width, primitive.device, primitive.dtype)
            .expand(batch, -1, -1)
            .float(),
            std_hr.float(),
            std_hr.new_zeros(batch, 4 * h * w, 1).float(),
            values_with_density.float(),
            height,
            width,
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
                "primitive_count": float(4 * h * w),
                "std_cell_mean": float(std_cell.detach().mean()),
                "std_hr_mean": float(std_hr.detach().mean()),
                "opacity_mean": float(opacity.detach().mean()),
                "density_mean": float(density.detach().mean()),
                "low_density_ratio": float((density.detach() < 1e-4).float().mean()),
                "value_abs_mean": float(value.detach().abs().mean()),
                "delta_abs_mean": float(delta.detach().abs().mean()),
            }
        if not return_aux:
            return delta
        return delta, {"density": density, "std_hr": std_hr, "value": value}


class GSFusion(nn.Module):
    """E6 trunk plus contrast-gated, detail-only LR subcell transport."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
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
        self.subcell_sampler = RelativeMSISubcellSampler()
        self.hsi_detail_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.msi_contrast_gate = nn.Conv2d(dim, dim, 1, bias=False)
        self.gaussian_transport = DetailSubcellGaussianTransport(dim)
        self._last_stats: Dict[str, float] = {}
        self.arch_summary = (
            "LR Primitive Detail-Only v1: E6 ADCI/fusion trunk; each LR cell has "
            "four normalized MSI subcells; Gaussian values are HSI spectral directions "
            "multiplicatively gated by within-cell MSI contrast only; normalized scatter "
            "adds a detail residual to the stable fused latent"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.gaussian_transport.reset_custom_init()

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None, return_aux=False):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(lr_hsi, size=target_size, mode="bicubic", align_corners=False)
        h_lr = self.shallow_encoder1(lr_hsi)
        f_m = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            h_lr = layer(h_lr)
        for layer in self.adci_msi_layers:
            f_m = layer(f_m)
        f_h = F.interpolate(h_lr, size=target_size, mode="bicubic", align_corners=False)
        f_fused = self.conv0(torch.cat((f_h, f_m), dim=1))
        m_sub = self.subcell_sampler(f_m, h_lr.shape[-2:])
        m_contrast = m_sub - m_sub.mean(dim=2, keepdim=True)
        spectral = self.hsi_detail_proj(h_lr).unsqueeze(2)
        gate_in = m_contrast.permute(0, 2, 1, 3, 4).reshape(
            h_lr.shape[0] * 4, h_lr.shape[1], h_lr.shape[2], h_lr.shape[3]
        )
        gate = torch.tanh(self.msi_contrast_gate(gate_in)).view(
            h_lr.shape[0], 4, h_lr.shape[1], h_lr.shape[2], h_lr.shape[3]
        ).permute(0, 2, 1, 3, 4)
        primitive = spectral * gate
        gaussian_delta, aux = self.gaussian_transport(primitive, target_size, return_aux=True)
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        with torch.no_grad():
            self._last_stats = {
                "msi_contrast_abs_mean": float(m_contrast.detach().abs().mean()),
                "primitive_abs_mean": float(primitive.detach().abs().mean()),
                "gaussian_delta_abs_mean": float(gaussian_delta.detach().abs().mean()),
                "gaussian_to_fused_ratio": float(gaussian_delta.detach().abs().mean() / (f_fused.detach().abs().mean() + 1e-8)),
            }
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {"H_lr": h_lr, "F_M": f_m, "F_fused": f_fused, "m_sub": m_sub, "m_contrast": m_contrast, "primitive": primitive, "gaussian_delta": gaussian_delta, **aux}

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = dict(self.gaussian_transport.last_stats or {})
        stats.update(self._last_stats)
        return [] if not stats else [{"layer": "lr_detail_subcell", **stats}]


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
