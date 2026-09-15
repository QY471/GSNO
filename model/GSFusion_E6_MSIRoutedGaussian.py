"""E6: MSI-guided selection of local HSI content for Gaussian value."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.GSFusion_HRFused_Circular_PrimitiveValueCommon import (
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
        self.routing = MSIGuidedHSILocalRouting(dim, routing_dim=dim // 4)
        torch.set_rng_state(extra_rng_state)
        self.arch_summary = (
            "E6: E3-common ADCI/fusion/primitive embedding; geometry reads E_g; "
            "MSI keys route a 3x3 neighborhood of HSI values; routed correction "
            "is added to E_g value source before circular normalized splatting"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        self.reset_common_init()
        self.routing.reset_output_init()

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        del sf
        base, f_h, f_m, _joint, f_fused, e_g = self._encode_common(
            lr_hsi, hr_msi
        )
        routing_delta = self.routing(f_h, f_m)
        value_source = e_g + routing_delta
        gaussian_delta = self.gaussian_refine(
            transport_x=e_g, value_x=value_source
        )
        self._record_primitive_stats(
            e_g,
            value_source,
            routing_delta=routing_delta,
            routing=self.routing,
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
