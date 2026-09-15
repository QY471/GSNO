"""DIM80 constrained elliptical E3 with checkpoint-compatible ADCI Exact."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ADCI_Exact import ADCIExact
from model.GSFusion_E3_ConstrainedEllipticalGaussian import (
    GSFusion as ConstrainedEllipticalGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(ConstrainedEllipticalGSFusion):
    """Constrained elliptical E3 whose six ADCI blocks use ADCI Exact."""

    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            max_axis_ratio=max_axis_ratio,
            **kwargs,
        )

        rng_state = torch.get_rng_state()
        self.adci_hsi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        torch.set_rng_state(rng_state)

        self.arch_summary = (
            "E3 DIM80 constrained elliptical Gaussian with checkpoint-compatible "
            "ADCI Exact CUDA/Triton value aggregation; all Gaussian geometry, "
            "density normalization, adaptive-3sigma support, fusion, and decoder "
            "paths are unchanged"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
