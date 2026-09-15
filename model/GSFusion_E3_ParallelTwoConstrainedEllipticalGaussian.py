"""DIM-configurable E3 with two parallel constrained Gaussian experts."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.GSFusion_E3_ParallelConstrainedEllipticalGaussian import (
    GSFusion as ParallelThreeGaussianGSFusion,
    compute_loss,
    sam_loss,
)


class GSFusion(ParallelThreeGaussianGSFusion):
    """Mix two Gaussian residuals that read the same primitive field."""

    gaussian_layers = 2
    variant_name = "two_parallel_constrained_elliptical_gaussian_experts"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gaussian_refine_extra = nn.ModuleList([self.gaussian_refine_extra[0]])
        self.gaussian_mixture_logits = nn.Parameter(torch.zeros(2))
        self.arch_summary = (
            "E3 DIM-configurable two-expert parallel constrained elliptical "
            "Gaussian mixture: both experts read the same primitive field; "
            "softmax-normalized global mixing; no sequential state cascade"
        )


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
