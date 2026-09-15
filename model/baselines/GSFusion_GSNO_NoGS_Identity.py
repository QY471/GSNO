"""
GSNO backbone without Gaussian local gather.

This diagnostic keeps the GSNO data flow intact:
  LR-HSI -> LR-space ADCI stream -> feature upsample
  HR-MSI -> HR-space ADCI stream
  concat -> 1x1 fusion -> fc decoder -> residual + bicubic

Only the GaussianSplatEncoder stack is replaced by identity. This isolates the
contribution of GSNO's normalized local Gaussian gather operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


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
        self.num_gs_layers = 0
        self.arch_summary = (
            f"GSNO_NoGS_Identity: dim={dim}, ADCI={adci_layers}, "
            "GS=False, identity in place of GaussianSplatEncoder"
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

        self.gs_layers = nn.Identity()

        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        self.reset_custom_init()

    def reset_custom_init(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def collect_gs_stats(self):
        return []

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
