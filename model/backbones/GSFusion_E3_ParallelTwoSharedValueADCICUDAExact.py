"""CUDA ADCI Exact version of the parallel-two shared-value model."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ADCI_Exact import ADCIExact
from model.GSFusion_E3_ParallelTwoSharedValue import (
    GSFusion as ParallelTwoSharedValueGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(ParallelTwoSharedValueGSFusion):
    """Use Triton-exact ADCI while retaining tied Gaussian value parameters."""

    variant_name = "two_parallel_shared_value_gaussian_adci_cuda_exact"

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
            "E3 DIM80 two-expert parallel constrained elliptical Gaussian "
            "mixture with independent geometry, tied value parameters, and "
            "checkpoint-compatible ADCI Exact Triton aggregation"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
