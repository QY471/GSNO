"""E8: frozen HR-fused E6 plus a zero-start HR local reconstruction branch."""

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
    """Add one depthwise 3x3 residual after E6's Gaussian injection.

    ``local_out`` is reset to exactly zero after the shared Xavier pass in
    Train_Cave.py. Consequently, at initialization this model computes the
    same function as the frozen HR-fused E6 while retaining a learnable HR
    local correction path.
    """

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **_: object,
    ) -> None:
        super().__init__(dim, num_bands, num_msi, adci_layers)

        # Preserve E6's routing path without changing its definition.
        extra_rng_state = torch.get_rng_state()
        self.routing = MSIGuidedHSILocalRouting(dim, routing_dim=dim // 4)

        # The only E8 addition: spatially local HR refinement after Gaussian
        # injection and before the unchanged pointwise residual decoder.
        self.local_dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.local_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        torch.set_rng_state(extra_rng_state)

        self.arch_summary = (
            "E8: complete HR-fused E6 routing and circular normalized "
            "Gaussian splatting; zero-start depthwise-3x3 plus pointwise-1x1 "
            "HR local residual after Gaussian injection; unchanged 1x1 "
            "residual decoder"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        self.reset_common_init()
        self.routing.reset_output_init()
        # local_dw may be random because zero local_out makes the branch zero.
        nn.init.zeros_(self.local_out.weight)
        nn.init.zeros_(self.local_out.bias)

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
        local_delta = self.local_out(F.gelu(self.local_dw(refined)))
        refined = refined + local_delta

        with torch.no_grad():
            refined_abs = refined.detach().abs().mean()
            local_abs = local_delta.detach().abs().mean()
            self._last_primitive_stats.update(
                {
                    "hr_local_delta_abs_mean": float(local_abs),
                    "hr_local_to_refined_ratio": float(
                        local_abs / (refined_abs + 1e-8)
                    ),
                }
            )

        residual = self.fc2(F.gelu(self.fc1(refined)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
