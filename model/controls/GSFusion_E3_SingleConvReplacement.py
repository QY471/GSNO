"""E3 ablation with one ordinary 3x3 convolution replacing Gaussian refinement."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    GSFusion as E3GSFusion,
    compute_loss,
    sam_loss,
)


class SingleConvResidual(nn.Module):
    """One full-channel 3x3 residual convolution without extra nonlinear layers."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.conv(x)
        out = x + delta
        with torch.no_grad():
            input_abs = x.detach().abs().mean()
            delta_abs = delta.detach().abs().mean()
            self.last_stats = {
                "single_conv_input_abs_mean": float(input_abs),
                "single_conv_delta_abs_mean": float(delta_abs),
                "single_conv_delta_input_ratio": float(
                    delta_abs / (input_abs + 1e-8)
                ),
            }
        return out


class GSFusion(E3GSFusion):
    """Keep the complete E3 path and replace its Gaussian residual with one conv."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            **kwargs,
        )
        del self.gaussian_refine
        self.gaussian_refine = SingleConvResidual(dim)
        self.arch_summary = (
            "E3 single-convolution replacement: original ADCI, fusion, primitive "
            "embedding and decoder unchanged; one full-channel residual 3x3 "
            "convolution replaces Gaussian refinement"
        )
        self.reset_custom_init()

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = self.gaussian_refine.last_stats
        return [] if stats is None else [{"layer": "single_conv", **stats}]


__all__ = [
    "GSFusion",
    "SingleConvResidual",
    "compute_loss",
    "sam_loss",
]
