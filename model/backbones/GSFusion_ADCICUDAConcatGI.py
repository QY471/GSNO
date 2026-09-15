"""Two-stream ADCI Exact backbone with a Galerkin-integration decoder.

The model intentionally removes the independent fused reconstruction path and
the spatial Gaussian branch. Both encoded modalities are concatenated at the
HR grid, and the complete latent reconstruction passes through GI blocks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ADCI_Exact import ADCIExact
from model.GSFusion_GSNO import compute_loss, sam_loss


class FeatureLayerNorm(nn.Module):
    """Layer normalization over the last feature dimension."""

    def __init__(self, heads: int, head_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, 1, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads, 1, head_dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * ((x - mean) / (std + self.eps)) + self.bias


class GalerkinIntegration(nn.Module):
    """Global linear-attention integral followed by bonded activation."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = self.channels // self.heads

        self.qkv_proj = nn.Conv2d(self.channels, 3 * self.channels, 1)
        self.out_proj1 = nn.Conv2d(self.channels, self.channels, 1)
        self.out_proj2 = nn.Conv2d(self.channels, self.channels, 1)
        self.key_norm = FeatureLayerNorm(self.heads, self.head_dim)
        self.value_norm = FeatureLayerNorm(self.heads, self.head_dim)

    @staticmethod
    def bonded_gelu(x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        expanded = F.interpolate(
            x, scale_factor=2.0, mode="bicubic", align_corners=False
        )
        return F.interpolate(
            F.gelu(expanded),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        point_count = height * width
        qkv = self.qkv_proj(x)
        qkv = qkv.permute(0, 2, 3, 1).reshape(
            batch, point_count, self.heads, 3 * self.head_dim
        )
        q, k, v = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)
        k = self.key_norm(k)
        v = self.value_norm(v)

        global_context = torch.matmul(k.transpose(-2, -1), v) / point_count
        integrated = torch.matmul(q, global_context)
        integrated = integrated.permute(0, 2, 1, 3).reshape(
            batch, height, width, channels
        )
        integrated = integrated.permute(0, 3, 1, 2).contiguous()

        transformed = self.out_proj1(integrated + x)
        transformed = self.out_proj2(self.bonded_gelu(transformed))
        return x + transformed


class GSFusion(nn.Module):
    """CUDA ADCI features reconstructed only through a GI decoder."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        gi_layers: int = 1,
        gi_heads: int = 8,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        joint_channels = 2 * self.dim

        self.shallow_encoder1 = nn.Conv2d(self.num_bands, self.dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, self.dim, 1)
        self.adci_hsi_layers = nn.ModuleList(
            [ADCIExact(self.dim, self.dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCIExact(self.dim, self.dim) for _ in range(adci_layers)]
        )
        self.gi_layers = nn.ModuleList(
            [GalerkinIntegration(joint_channels, gi_heads) for _ in range(gi_layers)]
        )
        self.decoder1 = nn.Conv2d(joint_channels, self.dim, 1)
        self.decoder2 = nn.Conv2d(self.dim, self.num_bands, 1)
        self._last_stats: Optional[Dict[str, float]] = None
        self.arch_summary = (
            "CUDA ADCI Exact x3 per modality; native HR concat without an "
            "independent fused path; one GI global reconstruction; 1x1 residual "
            "decoder; raw LR-HSI bicubic base; no spatial Gaussian branch"
        )

    @staticmethod
    def bonded_gelu(x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        expanded = F.interpolate(
            x, scale_factor=2.0, mode="bicubic", align_corners=False
        )
        return F.interpolate(
            F.gelu(expanded),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        )

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
    ):
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

        f_h = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        joint = torch.cat((f_h, f_msi), dim=1)
        gi_latent = joint
        for layer in self.gi_layers:
            gi_latent = layer(gi_latent)

        residual = self.decoder2(self.bonded_gelu(self.decoder1(gi_latent)))
        prediction = base + residual

        with torch.no_grad():
            self._last_stats = {
                "joint_abs_mean": float(joint.detach().abs().mean()),
                "gi_abs_mean": float(gi_latent.detach().abs().mean()),
                "gi_to_joint_ratio": float(
                    gi_latent.detach().abs().mean()
                    / (joint.detach().abs().mean() + 1e-8)
                ),
                "residual_abs_mean": float(residual.detach().abs().mean()),
                "residual_to_base_ratio": float(
                    residual.detach().abs().mean()
                    / (base.detach().abs().mean() + 1e-8)
                ),
            }

        if not return_aux:
            return prediction
        return prediction, {
            "base": base,
            "F_H": f_h,
            "F_M": f_msi,
            "joint": joint,
            "gi_latent": gi_latent,
            "residual": residual,
        }

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        if self._last_stats is None:
            return []
        return [{"layer": "adci_cuda_concat_gi", **self._last_stats}]


__all__ = ["GSFusion", "GalerkinIntegration", "compute_loss", "sam_loss"]
