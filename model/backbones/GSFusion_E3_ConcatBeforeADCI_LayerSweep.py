"""E3 concat-before-ADCI ablation with an explicit shared layer count."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ADCI_Exact import ADCIExact
from model.GSFusion_GSNO import compute_loss, sam_loss
from model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous import (
    GSFusion as ContinuousBase,
)


class GSFusion(ContinuousBase):
    """HR concat followed by a configurable serial ADCI stack."""

    def __init__(
        self,
        dim: int = 80,
        num_bands: int = 31,
        num_msi: int = 3,
        concat_adci_layers: int = 6,
        max_axis_ratio: float = 2.0,
        **kwargs: object,
    ) -> None:
        layer_count = int(concat_adci_layers)
        if layer_count <= 0:
            raise ValueError(f"concat_adci_layers must be positive, got {layer_count}")
        super().__init__(
            dim=dim,
            num_bands=num_bands,
            num_msi=num_msi,
            adci_layers=3,
            max_axis_ratio=max_axis_ratio,
            **kwargs,
        )
        del self.adci_hsi_layers
        del self.adci_msi_layers
        rng_state = torch.get_rng_state()
        self.concat_adci_layers = nn.ModuleList(
            [ADCIExact(dim, dim) for _ in range(layer_count)]
        )
        torch.set_rng_state(rng_state)
        self.concat_adci_layers_count = layer_count
        self.arch_summary = (
            f"E3 HR concat-before-ADCI with {layer_count} serial ADCI Exact CUDA "
            "blocks and the unchanged continuous constrained-elliptical Gaussian "
            "backend"
        )

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        del sf
        reference_size = tuple(int(v) for v in hr_msi.shape[-2:])
        query_size = reference_size if output_size is None else tuple(map(int, output_size))
        if min(query_size) <= 0:
            raise ValueError(f"output_size must be positive, got {query_size}")

        base = F.interpolate(lr_hsi, size=query_size, mode="bicubic", align_corners=False)
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_hsi = F.interpolate(f_hsi, size=reference_size, mode="bicubic", align_corners=False)
        f_msi = self.shallow_encoder2(hr_msi)
        joint = torch.cat((f_hsi, f_msi), dim=1)
        f_fused_reference = self.conv0(joint)
        for layer in self.concat_adci_layers:
            f_fused_reference = layer(f_fused_reference)

        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)
        gaussian_delta, gaussian_aux = self.gaussian_refine(
            e_g, reference_size, query_size, return_aux=True
        )
        f_fused = f_fused_reference if query_size == reference_size else F.interpolate(
            f_fused_reference, size=query_size, mode="bicubic", align_corners=False
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual
        if not return_aux:
            return prediction
        return prediction, {
            "F_H": f_hsi,
            "F_M": f_msi,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "E0": e0,
            "E_g": e_g,
            "gaussian_delta": gaussian_delta,
            **gaussian_aux,
        }


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
