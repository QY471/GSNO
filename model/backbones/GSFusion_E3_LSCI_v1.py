"""E3 with both ADCI encoders replaced by LSCI-v1 blocks.

LSCI-v1 (Local Spectral-Correlation Integration) performs multi-head,
mean-centered cosine correlation over the current feature grid's 3x3
neighbourhood.  It never reads scale, coordinates, or another modality.
"""

from __future__ import annotations

import math
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
    """Layer normalization over channels independently at every location."""

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


class LSCIBlock(nn.Module):
    """Multi-head local spectral-correlation integration on a 3x3 grid."""

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        ffn_hidden_dim: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.ffn_hidden_dim = int(ffn_hidden_dim or dim)
        self.eps = float(eps)

        self.norm1 = ChannelLayerNorm(dim)
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.norm2 = ChannelLayerNorm(dim)
        self.ffn1 = nn.Conv2d(dim, self.ffn_hidden_dim, kernel_size=1, bias=True)
        self.ffn2 = nn.Conv2d(self.ffn_hidden_dim, dim, kernel_size=1, bias=True)

        initial_temperature = 1.0 / math.sqrt(self.head_dim)
        inverse_softplus = math.log(math.expm1(initial_temperature))
        self.log_temperature = nn.Parameter(
            torch.full((self.num_heads,), inverse_softplus)
        )
        self.last_stats: Optional[Dict[str, float]] = None
        self._last_gradient_stats: Dict[str, float] = {}
        self._gradient_stat_calls = 0

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        normalized = self.norm1(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=1)
        q = q.view(batch, self.num_heads, self.head_dim, height, width)
        k_neighbours = F.unfold(k, kernel_size=3, padding=1).view(
            batch, self.num_heads, self.head_dim, 9, height, width
        )
        v_neighbours = F.unfold(v, kernel_size=3, padding=1).view(
            batch, self.num_heads, self.head_dim, 9, height, width
        )

        q_centered = q - q.mean(dim=2, keepdim=True)
        k_centered = k_neighbours - k_neighbours.mean(dim=2, keepdim=True)
        q_norm = F.normalize(q_centered, p=2.0, dim=2, eps=self.eps)
        k_norm = F.normalize(k_centered, p=2.0, dim=2, eps=self.eps)
        score = (q_norm.unsqueeze(3) * k_norm).sum(dim=2)
        score = score * self.temperature().view(1, self.num_heads, 1, 1, 1)
        attention = torch.softmax(score, dim=2)
        local = (attention.unsqueeze(2) * v_neighbours).sum(dim=3)
        local = local.reshape(batch, channels, height, width)
        local_projected = self.out_proj(local)
        x1 = x + local_projected
        ffn_update = self.ffn2(F.gelu(self.ffn1(self.norm2(x1))))
        output = x1 + ffn_update

        with torch.no_grad():
            entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=2)
            center = attention[:, :, 4]
            max_weight = attention.max(dim=2).values
            input_abs = x.detach().abs().mean()
            local_abs = local_projected.detach().abs().mean()
            update_abs = (output.detach() - x.detach()).abs().mean()
            q_raw_norm = q_centered.detach().norm(dim=2).mean()
            k_raw_norm = k_centered.detach().norm(dim=2).mean()
            q_unit_norm = q_norm.detach().norm(dim=2).mean()
            k_unit_norm = k_norm.detach().norm(dim=2).mean()
            temperatures = self.temperature().detach()
            self.last_stats = {
                "attention_entropy_mean": float(entropy.mean()),
                "attention_entropy_head_std": float(
                    entropy.mean(dim=(0, 2, 3)).std(unbiased=False)
                ),
                "center_weight_mean": float(center.mean()),
                "center_weight_head_std": float(
                    center.mean(dim=(0, 2, 3)).std(unbiased=False)
                ),
                "max_neighbor_weight_mean": float(max_weight.mean()),
                "temperature_mean": float(temperatures.mean()),
                "temperature_min": float(temperatures.min()),
                "temperature_max": float(temperatures.max()),
                "local_output_abs_mean": float(local_abs),
                "input_abs_mean": float(input_abs),
                "local_input_ratio": float(local_abs / (input_abs + 1e-8)),
                "residual_update_input_ratio": float(update_abs / (input_abs + 1e-8)),
                "q_centered_norm_mean": float(q_raw_norm),
                "k_centered_norm_mean": float(k_raw_norm),
                "q_normalized_norm_mean": float(q_unit_norm),
                "k_normalized_norm_mean": float(k_unit_norm),
                "attention_sum_max_error": float(
                    (attention.sum(dim=2) - 1.0).abs().max()
                ),
            }
        return output

    def collect_gradient_stats(self) -> Dict[str, float]:
        self._gradient_stat_calls += 1
        if self._gradient_stat_calls > 2 and self._gradient_stat_calls % 250:
            return dict(self._last_gradient_stats)

        def norm(parameter: torch.Tensor) -> float:
            if parameter.grad is None:
                return 0.0
            return float(parameter.grad.detach().float().norm())

        self._last_gradient_stats = {
            "grad_qkv_l2": norm(self.qkv.weight),
            "grad_out_proj_l2": norm(self.out_proj.weight),
            "grad_ffn1_l2": norm(self.ffn1.weight),
            "grad_ffn2_l2": norm(self.ffn2.weight),
            "grad_temperature_l2": norm(self.log_temperature),
        }
        return dict(self._last_gradient_stats)


class GSFusion(E3GSFusion):
    """Formal E3 backend with independent three-layer LSCI encoders."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        num_heads: int = 8,
        ffn_hidden_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        # Keep formal ADCI modules frozen only for paired-state compatibility.
        # They are never called by forward and are excluded from trainable count.
        for module in (*self.adci_hsi_layers, *self.adci_msi_layers):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        hidden = int(ffn_hidden_dim or dim)
        self.lsci_hsi_layers = nn.ModuleList([
            LSCIBlock(dim, num_heads, hidden) for _ in range(adci_layers)
        ])
        self.lsci_msi_layers = nn.ModuleList([
            LSCIBlock(dim, num_heads, hidden) for _ in range(adci_layers)
        ])
        self.arch_summary = (
            "E3-LSCI-v1: both independent ADCI encoders replaced by three "
            "current-grid 3x3 multi-head mean-centered spectral-correlation "
            "integration blocks; formal E3 fusion, circular normalized "
            "adaptive-3sigma Gaussian backend, decoder and bicubic base retained"
        )

    def collect_gradient_stats(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for branch, layers in (
            ("hsi", self.lsci_hsi_layers),
            ("msi", self.lsci_msi_layers),
        ):
            for index, layer in enumerate(layers, start=1):
                for key, value in layer.collect_gradient_stats().items():
                    result[f"{branch}_layer{index}_{key}"] = value
        return result

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        records = super().collect_gs_stats()
        for branch, layers in (
            ("hsi", self.lsci_hsi_layers),
            ("msi", self.lsci_msi_layers),
        ):
            for index, layer in enumerate(layers, start=1):
                if layer.last_stats is not None:
                    records.append({
                        "layer": f"{branch}_lsci_{index}",
                        **layer.last_stats,
                        **layer._last_gradient_stats,
                    })
        return records

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        target_size = hr_msi.shape[-2:]
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.lsci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.lsci_msi_layers:
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
    "LSCIBlock",
    "ChannelLayerNorm",
    "compute_loss",
    "sam_loss",
]
