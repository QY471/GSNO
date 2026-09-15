"""E3 with Center-Anchored Relative Integration encoders."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    compute_loss,
    sam_loss,
)


class ChannelLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            normalized * self.weight.view(1, -1, 1, 1)
            + self.bias.view(1, -1, 1, 1)
        )


class CARIBlock(nn.Module):
    """Center anchor plus group-dynamic integration of neighbour differences."""

    def __init__(
        self,
        dim: int = 64,
        num_groups: int = 8,
        score_hidden_dim: int = 110,
    ) -> None:
        super().__init__()
        if dim % num_groups:
            raise ValueError(f"dim={dim} must be divisible by groups={num_groups}")
        self.dim = int(dim)
        self.num_groups = int(num_groups)
        self.group_dim = self.dim // self.num_groups
        self.score_hidden_dim = int(score_hidden_dim)

        self.norm = ChannelLayerNorm(dim)
        self.anchor_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.score_mlp = nn.Sequential(
            nn.Linear(dim, self.score_hidden_dim),
            nn.LayerNorm(self.score_hidden_dim),
            nn.GELU(),
            nn.Linear(self.score_hidden_dim, self.num_groups),
        )
        self.channel_gate = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.anchor_gain = nn.Parameter(torch.full((dim,), 0.1))
        self.local_gain = nn.Parameter(torch.full((dim,), 0.1))
        self.last_stats: Optional[Dict[str, float]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm(x)
        anchor_update = self.anchor_proj(normalized)
        value = self.value_proj(normalized)
        neighbors = F.unfold(value, kernel_size=3, padding=1).view(
            batch, channels, 9, height, width
        )
        relative = neighbors - value.unsqueeze(2)
        relative_tokens = relative.permute(0, 3, 4, 2, 1).contiguous()
        weights = torch.softmax(self.score_mlp(relative_tokens), dim=3)
        grouped_relative = relative_tokens.view(
            batch,
            height,
            width,
            9,
            self.num_groups,
            self.group_dim,
        )
        local = (grouped_relative * weights.unsqueeze(-1)).sum(dim=3)
        local = local.reshape(batch, height, width, channels)
        local = local.permute(0, 3, 1, 2).contiguous()
        gated_local = local * torch.sigmoid(self.channel_gate(normalized))
        local_update = self.out_proj(gated_local)
        output = (
            x
            + self.anchor_gain.view(1, -1, 1, 1) * anchor_update
            + self.local_gain.view(1, -1, 1, 1) * local_update
        )

        with torch.no_grad():
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=3)
            input_abs = x.detach().abs().mean()
            anchor_abs = anchor_update.detach().abs().mean()
            local_abs = local_update.detach().abs().mean()
            self.last_stats = {
                "neighbor_entropy_normalized": float(
                    entropy.mean() / torch.log(torch.tensor(9.0, device=x.device))
                ),
                "center_weight_mean": float(weights[:, :, :, 4].mean()),
                "max_neighbor_weight_mean": float(weights.max(dim=3).values.mean()),
                "input_abs_mean": float(input_abs),
                "anchor_update_abs_mean": float(anchor_abs),
                "local_update_abs_mean": float(local_abs),
                "anchor_input_ratio": float(anchor_abs / (input_abs + 1e-8)),
                "local_input_ratio": float(local_abs / (input_abs + 1e-8)),
                "anchor_gain_abs_mean": float(self.anchor_gain.detach().abs().mean()),
                "local_gain_abs_mean": float(self.local_gain.detach().abs().mean()),
                "relative_abs_mean": float(relative.detach().abs().mean()),
            }
        return output


class GSFusion(E3GSFusion):
    """Formal E3 backend with independent three-layer CARI encoders."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        num_groups: int = 8,
        score_hidden_dim: int = 110,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        for module in (*self.adci_hsi_layers, *self.adci_msi_layers):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.cari_hsi_layers = nn.ModuleList(
            [
                CARIBlock(dim, num_groups, score_hidden_dim)
                for _ in range(adci_layers)
            ]
        )
        self.cari_msi_layers = nn.ModuleList(
            [
                CARIBlock(dim, num_groups, score_hidden_dim)
                for _ in range(adci_layers)
            ]
        )
        self.arch_summary = (
            "E3-CARI-v1: center-residual anchor plus group-dynamic integration "
            "of 3x3 neighbour differences in both independent modality "
            "encoders; formal E3 fusion, Gaussian backend and decoder retained"
        )

    def reset_custom_init(self) -> None:
        super().reset_custom_init()
        if not hasattr(self, "cari_hsi_layers"):
            return
        for layer in (*self.cari_hsi_layers, *self.cari_msi_layers):
            nn.init.constant_(layer.anchor_gain, 0.1)
            nn.init.constant_(layer.local_gain, 0.1)

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        records = super().collect_gs_stats()
        for branch, layers in (
            ("hsi", self.cari_hsi_layers),
            ("msi", self.cari_msi_layers),
        ):
            for index, layer in enumerate(layers, start=1):
                if layer.last_stats is not None:
                    records.append(
                        {
                            "layer": f"{branch}_cari_{index}",
                            **layer.last_stats,
                        }
                    )
        return records

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.cari_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.cari_msi_layers:
            f_msi = layer(f_msi)
        f_hsi_hr = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        joint = torch.cat([f_hsi_hr, f_msi], dim=1)
        fused = self.conv0(joint)
        primitive_base = self.primitive_input(joint)
        primitive = primitive_base + self.primitive_residual(primitive_base)
        gaussian_delta = self.gaussian_refine(primitive) - primitive
        residual = self.fc2(F.gelu(self.fc1(fused + gaussian_delta)))
        return base + residual


__all__ = [
    "GSFusion",
    "CARIBlock",
    "ChannelLayerNorm",
    "compute_loss",
    "sam_loss",
]
