"""Mainstream CNN ablation paired with mandatory Gaussian Operator 1.

The complete HR latent must pass through a conventional depthwise-separable
3x3 CNN before the lightweight decoder.  ADCI, fusion, pointwise spectral
mixing, MSI conditioning, decoder, and raw bicubic base are retained.  Only
the Gaussian renderer is replaced, and there is no ``x + delta`` bypass.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from model.operators.GSFusion_E3_GaussianOperator1 import (
    GSFusion as GaussianOperatorGSFusion,
    compute_loss,
    sam_loss,
)


class ForcedCNNOperatorBlock(nn.Module):
    """Pointwise spectral mixing plus mandatory standard local CNN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.spectral_transform = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )
        # A conventional, widely used depthwise-separable CNN operator.
        # MSI is retained through a simple 1x1 channel adapter rather than a
        # newly invented dynamic weighting mechanism.
        self.msi_adapter = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_operator = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_operator_init(self) -> None:
        nn.init.zeros_(self.spectral_transform[-1].weight)
        nn.init.zeros_(self.spectral_transform[-1].bias)

    def forward(self, x: torch.Tensor, msi_feature: torch.Tensor) -> torch.Tensor:
        if msi_feature.shape != x.shape:
            raise ValueError(
                f"MSI condition shape {tuple(msi_feature.shape)} must equal "
                f"latent shape {tuple(x.shape)}"
            )
        values = x + self.spectral_transform(x)
        cnn_input = values + self.msi_adapter(msi_feature)
        # Mandatory spatial transformation: the original latent has no route
        # to the decoder around this CNN operator.
        out = self.spatial_operator(cnn_input)

        with torch.no_grad():
            input_abs = x.detach().abs().mean()
            pointwise_abs = (values - x).detach().abs().mean()
            operator_change_abs = (out - values).detach().abs().mean()
            self.last_stats = {
                "cop_input_abs_mean": float(input_abs),
                "cop_pointwise_update_abs_mean": float(pointwise_abs),
                "cop_operator_change_abs_mean": float(operator_change_abs),
                "cop_pointwise_input_ratio": float(
                    pointwise_abs / (input_abs + 1e-8)
                ),
                "cop_operator_input_ratio": float(
                    operator_change_abs / (input_abs + 1e-8)
                ),
                "cop_forced_cnn_operator": 1.0,
                "cop_kernel_size": 3.0,
                "cop_depthwise_separable": 1.0,
            }
        return out


class GSFusion(GaussianOperatorGSFusion):
    """Gaussian Operator topology with only the renderer replaced by CNN."""

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
        rng_state = torch.get_rng_state()
        self.operator_blocks = nn.ModuleList([ForcedCNNOperatorBlock(dim)])
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "ADCI and complete-latent mandatory operator topology retained; "
            "Gaussian renderer replaced by mainstream MSI-conditioned 3x3 "
            "depthwise-separable CNN; no operator bypass; 1x1 decoder"
        )
        self.reset_custom_init()

    def collect_gs_stats(self):
        result = []
        for index, block in enumerate(self.operator_blocks, start=1):
            if block.last_stats is not None:
                result.append({"layer": f"cnn_operator_{index}", **block.last_stats})
        return result


__all__ = ["GSFusion", "ForcedCNNOperatorBlock", "compute_loss", "sam_loss"]
