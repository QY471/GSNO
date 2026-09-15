"""Continuous-output E3 with checkpoint-compatible ADCI Exact acceleration.

This model preserves the complete HR factorized reference-continuous path and
only replaces its six ADCI blocks with the mathematically equivalent fused
CUDA/Triton value aggregation. Parameter names and shapes remain compatible
with the paired initialization of the original continuous-output model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ADCI_Exact import ADCIExact
from model.GSFusion_E3_HRFactorizedReferenceContinuous import (
    GSFusion as HRFactorizedReferenceContinuous,
    compute_loss,
    sam_loss,
)


class GSFusion(HRFactorizedReferenceContinuous):
    """HR reference-continuous E3 whose six ADCI blocks use ADCI Exact."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        factorized_cap_ratio: float = 0.25,
        **kwargs: object,
    ) -> None:
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            factorized_cap_ratio=factorized_cap_ratio,
            **kwargs,
        )

        # ADCIExact intentionally has the same state-dict keys and tensor
        # shapes as ADCI. Preserve global RNG state so this implementation
        # change does not perturb initialization of unrelated parameters.
        rng_state = torch.get_rng_state()
        self.adci_hsi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(adci_layers)]
        )
        torch.set_rng_state(rng_state)

        self.arch_summary = (
            "E3 HR Factorized Reference Continuous with unchanged native-HR "
            "reference-coordinate Gaussian rendering, bounded factorized "
            "primitive content, arbitrary output_size, and checkpoint-compatible "
            "ADCI Exact CUDA/Triton value aggregation"
        )
        self.reset_custom_init()


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
