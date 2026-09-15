"""
GSNO strong backbone with v2 scatter Gaussian refinement.

Diagnostic intent:
  Start from the GSNO_NoGS_Identity backbone and insert only v2 scatter
  GSEncoder layers between the fused feature and the final decoder.

No DGP, no GSNO gather, no new loss, no extra routing tricks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss
from model.important_model_support.GSFusionv2 import GSEncoder


class GSFusion(nn.Module):
    def __init__(
        self,
        dim=64,
        num_bands=31,
        num_msi=3,
        num_basis=16,
        num_gs_layers=3,
        edsr_resblocks=6,
        adci_layers=3,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.dim = dim
        self.num_gs_layers = num_gs_layers
        self.arch_summary = (
            f"GSNOBackbone+ScatterGS: dim={dim}, ADCI={adci_layers}, "
            f"GS=v2 scatter * {num_gs_layers}"
        )

        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)

        self.adci_hsi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])
        self.adci_msi_layers = nn.ModuleList([ADCI(dim, dim) for _ in range(adci_layers)])

        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        self.gs_layers = nn.Sequential(
            *[GSEncoder(dim) for _ in range(num_gs_layers)]
        )

        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        self.reset_custom_init()

    def reset_custom_init(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def collect_gs_stats(self):
        stats = []
        for i, layer in enumerate(self.gs_layers):
            layer_stats = getattr(layer, "last_stats", None)
            if layer_stats is None:
                continue
            item = {"layer": i}
            item.update(layer_stats)
            stats.append(item)
        return stats

    def forward(self, lr_hsi, hr_msi, sf):
        lr_hsi_up = F.interpolate(
            lr_hsi, scale_factor=sf, mode="bicubic", align_corners=False
        )

        f_msi = self.shallow_encoder2(hr_msi)
        f_hsi = self.shallow_encoder1(lr_hsi)

        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)

        f_hsi = F.interpolate(f_hsi, scale_factor=sf, mode="bicubic", align_corners=False)

        feat = torch.cat([f_hsi, f_msi], dim=1)
        feat = self.conv0(feat)
        feat = self.gs_layers(feat)

        residual = self.fc2(F.gelu(self.fc1(feat)))
        return residual + lr_hsi_up
