"""Mandatory local-operator control paired with Gaussian Operator 1.

The complete HR latent still has to pass through the operator before the
lightweight 1x1 decoder.  This control changes only the spatial integration:
Gaussian rasterization is replaced by an MSI-conditioned dynamic 3x3 convex
aggregation.  There is no latent-to-decoder bypass and no ``x + delta`` path.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.operators.GSFusion_E3_GaussianOperator1 import (
    GSFusion as GaussianOperatorGSFusion,
    compute_loss,
    sam_loss,
)


class ForcedLocalOperatorBlock(nn.Module):
    """Pointwise spectral mixing followed by mandatory dynamic 3x3 mixing."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.spectral_transform = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )

        # Choose the hidden width so this head closely matches the Gaussian
        # geometry head's parameter count.  The nine logits describe one
        # normalized 3x3 kernel per HR location, shared across channels just
        # as one Gaussian geometry is shared across latent channels.
        gaussian_head_parameters = 2 * dim * dim + 3 * dim + 2
        hidden = max(1, round((gaussian_head_parameters - 9) / (2 * dim + 10)))
        self.local_kernel_head = nn.Sequential(
            nn.Conv2d(2 * dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 9, kernel_size=1),
        )
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_operator_init(self) -> None:
        nn.init.zeros_(self.spectral_transform[-1].weight)
        nn.init.zeros_(self.spectral_transform[-1].bias)
        # Start from a stable uniform local average; training can move every
        # location away from it through the MSI-conditioned kernel logits.
        nn.init.zeros_(self.local_kernel_head[-1].weight)
        nn.init.zeros_(self.local_kernel_head[-1].bias)

    def forward(self, x: torch.Tensor, msi_feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if msi_feature.shape != x.shape:
            raise ValueError(
                f"MSI condition shape {tuple(msi_feature.shape)} must equal "
                f"latent shape {tuple(x.shape)}"
            )

        values = x + self.spectral_transform(x)
        logits = self.local_kernel_head(torch.cat([values, msi_feature], dim=1))
        weights = torch.softmax(logits, dim=1)
        padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
        patches = F.unfold(padded, kernel_size=3).view(
            batch, channels, 9, height, width
        )
        # Return the aggregated latent itself.  There is deliberately no
        # residual bypass around this mandatory local operator.
        out = (patches * weights.unsqueeze(1)).sum(dim=2)

        with torch.no_grad():
            input_abs = x.detach().abs().mean()
            pointwise_abs = (values - x).detach().abs().mean()
            operator_change_abs = (out - values).detach().abs().mean()
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
            self.last_stats = {
                "lop_input_abs_mean": float(input_abs),
                "lop_pointwise_update_abs_mean": float(pointwise_abs),
                "lop_operator_change_abs_mean": float(operator_change_abs),
                "lop_pointwise_input_ratio": float(
                    pointwise_abs / (input_abs + 1e-8)
                ),
                "lop_operator_input_ratio": float(
                    operator_change_abs / (input_abs + 1e-8)
                ),
                "lop_kernel_entropy_mean": float(entropy.detach().mean()),
                "lop_kernel_max_weight_mean": float(
                    weights.detach().amax(dim=1).mean()
                ),
                "lop_forced_local_operator": 1.0,
                "lop_kernel_size": 3.0,
            }
        return out


class GSFusion(GaussianOperatorGSFusion):
    """Gaussian Operator 1 topology with only its spatial operator replaced."""

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
        self.operator_blocks = nn.ModuleList([ForcedLocalOperatorBlock(dim)])
        torch.set_rng_state(rng_state)
        self.arch_summary = (
            "ADCI and complete-latent mandatory operator topology retained; "
            "Gaussian renderer replaced only by MSI-conditioned dynamic 3x3 "
            "convex aggregation; no operator bypass; lightweight 1x1 decoder"
        )
        self.reset_custom_init()

    def collect_gs_stats(self):
        result = []
        for index, block in enumerate(self.operator_blocks, start=1):
            if block.last_stats is not None:
                result.append({"layer": f"local_operator_{index}", **block.last_stats})
        return result


__all__ = ["GSFusion", "ForcedLocalOperatorBlock", "compute_loss", "sam_loss"]
