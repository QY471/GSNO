"""Frozen diagnostic for the spatial units used by continuous Gaussian sigma."""

from __future__ import annotations

import math

from model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous import (
    ContinuousConstrainedEllipticalGaussianResidual,
    GSFusion as ContinuousGSFusion,
    compute_loss,
    sam_loss,
)


class SigmaUnitProbeResidual(ContinuousConstrainedEllipticalGaussianResidual):
    def __init__(
        self,
        *args,
        observation_scale_exponent: float = 0.0,
        reference_scale_exponent: float = 0.0,
        training_ratio: float = 4.0,
        reference_baseline: float = 512.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.observation_scale_exponent = float(observation_scale_exponent)
        self.reference_scale_exponent = float(reference_scale_exponent)
        self.training_ratio = float(training_ratio)
        self.reference_baseline = float(reference_baseline)
        self.runtime_sigma_multiplier = 1.0

    def set_grid_context(self, lr_size, reference_size):
        lr_h, lr_w = map(float, lr_size)
        reference_h, reference_w = map(float, reference_size)
        ratio = math.sqrt((reference_h / lr_h) * (reference_w / lr_w))
        reference_extent = math.sqrt(reference_h * reference_w)
        self.runtime_sigma_multiplier = (
            (ratio / self.training_ratio) ** self.observation_scale_exponent
            * (reference_extent / self.reference_baseline)
            ** self.reference_scale_exponent
        )

    def forward(self, *args, **kwargs):
        original_min = self.std_min_px
        original_max = self.std_max_px
        self.std_min_px = original_min * self.runtime_sigma_multiplier
        self.std_max_px = original_max * self.runtime_sigma_multiplier
        try:
            result = super().forward(*args, **kwargs)
            self.last_stats["hrgs_sigma_unit_multiplier"] = float(
                self.runtime_sigma_multiplier
            )
            self.last_stats["hrgs_observation_scale_exponent"] = float(
                self.observation_scale_exponent
            )
            self.last_stats["hrgs_reference_scale_exponent"] = float(
                self.reference_scale_exponent
            )
            return result
        finally:
            self.std_min_px = original_min
            self.std_max_px = original_max


class GSFusion(ContinuousGSFusion):
    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        max_axis_ratio: float = 2.0,
        observation_scale_exponent: float = 0.0,
        reference_scale_exponent: float = 0.0,
        training_ratio: float = 4.0,
        reference_baseline: float = 512.0,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=adci_layers,
            max_axis_ratio=max_axis_ratio,
            **kwargs,
        )
        self.gaussian_refine = SigmaUnitProbeResidual(
            dim=dim,
            max_axis_ratio=max_axis_ratio,
            observation_scale_exponent=observation_scale_exponent,
            reference_scale_exponent=reference_scale_exponent,
            training_ratio=training_ratio,
            reference_baseline=reference_baseline,
        )
        self.reset_custom_init()
        self.arch_summary = (
            "Frozen continuous Gaussian sigma-unit probe; no new trainable "
            "parameters and native 4x/512 behavior preserved exactly"
        )

    def forward(self, lr_hsi, hr_msi, *args, **kwargs):
        self.gaussian_refine.set_grid_context(
            lr_hsi.shape[-2:], hr_msi.shape[-2:]
        )
        return super().forward(lr_hsi, hr_msi, *args, **kwargs)


__all__ = ["GSFusion", "SigmaUnitProbeResidual", "compute_loss", "sam_loss"]
