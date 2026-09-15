"""E3 HR primitive content with a reference-coordinate continuous renderer.

This isolated control removes the LR-cell MSI footprint and E6 routing from
Gaussian content generation.  Primitive content is formed on the complete
native HR-MSI grid exactly from the E3 inputs (F_H, F_M), while rendering uses
the native HR-MSI pixel coordinate system and can be queried at another output
size.  The stable fused trunk and raw LR-HSI bicubic residual base are kept.

The file is intentionally not registered in Train_Cave.py.  It is a control
for separating continuous rendering from a new content-fusion contribution;
it must not be presented as a new main architecture.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_E6_LRPrimitiveScaleNormalizedDetail import (
    ScaleNormalizedDetailGaussianTransport,
)
from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


class GSFusion(nn.Module):
    """E3 content generation plus continuous rendering (control model)."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)

        # Keep the E3 reconstruction trunk unchanged.
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)
        self.adci_hsi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        # Dedicated E3 primitive embedding, evaluated on the full HR grid.
        extra_rng_state = torch.get_rng_state()
        self.primitive_input = nn.Conv2d(2 * dim, dim, 1)
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        torch.set_rng_state(extra_rng_state)

        # Stride one is deliberate: the first comparison must not mix the
        # content-source change with an S2/S4 primitive-density change.
        self.gaussian_transport = ScaleNormalizedDetailGaussianTransport(
            dim,
            anchor_stride_hr=1,
            std_min_hr=0.30,
            std_max_hr=1.50,
            std_init_hr=0.90,
        )
        # E3's residual value head uses biases in both 1x1 layers. Preserve
        # those parameter shapes so this candidate matches E3's capacity.
        self.gaussian_transport.value_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=True),
        )
        self._last_stats: Dict[str, float] = {}
        self.arch_summary = (
            "CONTROL - E3 HR Primitive Reference Continuous: E3 two-stream ADCI and "
            "full-HR E_g content; no LR footprint, no E6 routing, no extra "
            "MSI contrast gate; stride-1 circular density-normalized adaptive "
            "3-sigma renderer in native HR-MSI pixel coordinates; optional "
            "continuous output_size; stable fused trunk and bicubic base"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)
        self.gaussian_transport.reset_custom_init()
        nn.init.zeros_(self.gaussian_transport.value_head[-1].bias)

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
        query_size = (
            reference_size
            if output_size is None
            else (int(output_size[0]), int(output_size[1]))
        )
        if query_size[0] <= 0 or query_size[1] <= 0:
            raise ValueError(f"output_size must be positive, got {query_size}")

        base = F.interpolate(
            lr_hsi, size=query_size, mode="bicubic", align_corners=False
        )
        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_h = F.interpolate(
            f_hsi, size=reference_size, mode="bicubic", align_corners=False
        )
        f_m = f_msi
        joint = torch.cat((f_h, f_m), dim=1)
        f_fused_reference = self.conv0(joint)
        e0 = self.primitive_input(joint)
        e_g = e0 + self.primitive_residual(e0)

        gaussian_delta, gaussian_aux = self.gaussian_transport(
            e_g,
            reference_size=reference_size,
            out_size=query_size,
            return_aux=True,
        )
        f_fused = (
            f_fused_reference
            if query_size == reference_size
            else F.interpolate(
                f_fused_reference,
                size=query_size,
                mode="bicubic",
                align_corners=False,
            )
        )
        refined = f_fused + gaussian_delta
        residual = self.fc2(F.gelu(self.fc1(refined)))
        prediction = base + residual

        with torch.no_grad():
            self._last_stats = {
                "primitive_abs_mean": float(e_g.detach().abs().mean()),
                "gaussian_delta_abs_mean": float(
                    gaussian_delta.detach().abs().mean()
                ),
                "gaussian_to_fused_ratio": float(
                    gaussian_delta.detach().abs().mean()
                    / (f_fused.detach().abs().mean() + 1e-8)
                ),
                "reference_height": float(reference_size[0]),
                "reference_width": float(reference_size[1]),
                "query_height": float(query_size[0]),
                "query_width": float(query_size[1]),
            }

        if not return_aux:
            return prediction
        return prediction, {
            "F_H": f_h,
            "F_M": f_m,
            "F_fused_reference": f_fused_reference,
            "F_fused": f_fused,
            "E0": e0,
            "E_g": e_g,
            "gaussian_delta": gaussian_delta,
            **gaussian_aux,
        }

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        stats = dict(self.gaussian_transport.last_stats or {})
        stats.update(self._last_stats)
        return [] if not stats else [{"layer": "e3_hr_reference_field", **stats}]


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
