"""Transparent target-size adapter for the frozen unified AFNO epoch-1658 model.

The learned modules and their parameters are unchanged.  Only the two spatial
resizes that historically used ``scale_factor=sf`` are expressed using the
observed HR-MSI and LR-HSI tensor sizes.  On exactly divisible integer ratios
this must be numerically identical to the frozen training-time forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from analysis.afno_psf_sigma_transfer_20260816.source_snapshot.GSFusion_AFNO_ZhuJunweiUnified_64 import (
    GSFusion as FrozenAFNO,
)
from tools.Utils import make_coord


class GSFusion(FrozenAFNO):
    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
        return_aux: bool = False,
        output_size=None,
    ) -> torch.Tensor:
        del sf, return_aux
        target_size = tuple(int(value) for value in hr_msi.shape[-2:])
        if output_size is not None and tuple(output_size) != target_size:
            raise ValueError("AFNO adapter predicts on the supplied HR-MSI grid")

        lr_hsi_up = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )
        high_resolution_features = self.shallow_encoder2(
            torch.cat((hr_msi, lr_hsi_up), dim=1)
        )
        low_resolution_features = self.shallow_encoder1(lr_hsi)
        high_resolution_features = self.ADCI1_3(
            self.ADCI1_2(self.ADCI1_1(high_resolution_features))
        )
        low_resolution_features = self.ADCI2_3(
            self.ADCI2_2(self.ADCI2_1(low_resolution_features))
        )

        batch, _, output_height, output_width = hr_msi.shape
        coordinates = make_coord(
            (output_height, output_width), flatten=False
        ).to(hr_msi.device).unsqueeze(0).expand(
            batch, output_height, output_width, 2
        )
        return (
            self.infi(
                low_resolution_features,
                high_resolution_features,
                coordinates,
            )
            + lr_hsi_up
        )


__all__ = ["GSFusion"]
