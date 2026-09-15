"""E7: combine HSI spectral anchoring with MSI-guided local HSI routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.important_model_support.GSFusion_HRFused_Circular_PrimitiveValueCommon import (
    HRFusedPrimitiveValueBase,
    MSIGuidedHSILocalRouting,
    compute_loss,
    sam_loss,
)


class GSFusion(HRFusedPrimitiveValueBase):
    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **_: object,
    ) -> None:
        super().__init__(dim, num_bands, num_msi, adci_layers)
        extra_rng_state = torch.get_rng_state()
        self.value_condition = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.routing = MSIGuidedHSILocalRouting(dim, routing_dim=dim // 4)
        torch.set_rng_state(extra_rng_state)
        self.arch_summary = (
            "E7: E3-common ADCI/fusion/primitive embedding; geometry reads E_g; "
            "value is F_H plus pointwise joint correction plus MSI-keyed local "
            "HSI routing; single circular normalized adaptive-3sigma residual"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        self.reset_common_init()
        nn.init.zeros_(self.value_condition[-1].weight)
        nn.init.zeros_(self.value_condition[-1].bias)
        self.routing.reset_output_init()

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        base, f_h, f_m, joint, f_fused, e_g = self._encode_common(
            lr_hsi, hr_msi
        )
        pointwise_delta = self.value_condition(joint)
        routing_delta = self.routing(f_h, f_m)
        value_source = f_h + pointwise_delta + routing_delta
        gaussian_delta = self.gaussian_refine(
            transport_x=e_g, value_x=value_source
        )
        self._record_primitive_stats(
            e_g,
            value_source,
            pointwise_delta=pointwise_delta,
            routing_delta=routing_delta,
            routing=self.routing,
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
