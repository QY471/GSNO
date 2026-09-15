"""Parameter-matched three-layer local-convolution control for E3.

This control keeps the complete E3 ADCI trunk, fusion path, primitive
embedding and spectral decoder.  It replaces the three sequential Gaussian
residual layers with three ordinary residual local-convolution blocks.  The
control therefore tests whether the Three-Gaussian-Layers result comes from
Gaussian aggregation itself or merely from extra depth and a larger local
receptive field.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ADCI_Exact import ADCIExact
from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    compute_loss,
    sam_loss,
)


class LocalConvResidualBlock(nn.Module):
    """Pointwise bottleneck plus an ordinary depthwise 3x3 local convolution."""

    def __init__(self, dim: int, hidden_dim: int = 90) -> None:
        super().__init__()
        self.pre = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.local = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.post = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.post(F.gelu(self.local(F.gelu(self.pre(x)))))
        out = x + delta
        with torch.no_grad():
            input_abs = x.detach().abs().mean()
            delta_abs = delta.detach().abs().mean()
            self.last_stats = {
                "localconv_input_abs_mean": float(input_abs),
                "localconv_delta_abs_mean": float(delta_abs),
                "localconv_delta_input_ratio": float(delta_abs / (input_abs + 1e-8)),
            }
        return out


class ThreeLocalConvRefine(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 90) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [LocalConvResidualBlock(dim, hidden_dim) for _ in range(3)]
        )

    def reset_residual_init(self) -> None:
        for layer in self.layers:
            layer.reset_residual_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GSFusion(E3GSFusion):
    """E3 with the three Gaussian residuals replaced by local convolutions."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        local_hidden_dim: int = 90,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        # Keep the adopted ADCI mathematics and checkpoint keys unchanged,
        # while using the numerically aligned fused CUDA value aggregation.
        self.adci_hsi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        del self.gaussian_refine
        self.local_conv_refine = ThreeLocalConvRefine(dim, local_hidden_dim)
        self.arch_summary = (
            "E3 parameter-matched three-local-convolution control: exact CUDA ADCI, fusion, "
            "primitive embedding and decoder unchanged; three residual "
            "1x1-depthwise3x3-1x1 blocks replace three Gaussian residual layers"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
        # The E3 parent calls this virtual method before this subclass has
        # replaced its Gaussian module.  Handle both construction phases.
        if hasattr(self, "local_conv_refine"):
            self.local_conv_refine.reset_residual_init()
        elif hasattr(self, "gaussian_refine"):
            self.gaussian_refine.reset_residual_init()

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        result = []
        for index, layer in enumerate(self.local_conv_refine.layers, start=1):
            if layer.last_stats is not None:
                result.append({"layer": f"local_conv_{index}", **layer.last_stats})
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
        joint = torch.cat([f_hsi_hr, f_msi], dim=1)
        fused = self.conv0(joint)
        primitive_base = self.primitive_input(joint)
        primitive = primitive_base + self.primitive_residual(primitive_base)
        transported = self.local_conv_refine(primitive)
        local_delta = transported - primitive
        refined = fused + local_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = [
    "GSFusion",
    "LocalConvResidualBlock",
    "ThreeLocalConvRefine",
    "compute_loss",
    "sam_loss",
]
