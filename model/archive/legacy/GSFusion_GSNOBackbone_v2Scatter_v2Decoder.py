"""
GSNO backbone with v2 dense scatter Gaussian layers.

This isolates the GSNO backbone/decoder from its local normalized gather GS:
  LR-HSI ADCI stack -> upsample -> concat with HR-MSI ADCI stack -> conv0
  -> v2 scatter GSEncoder blocks -> Conv3x3/ReLU/Conv3x3 decoder -> bicubic residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GSFusion_GSNO import ADCI
from model.important_model_support.GSFusionv2 import GSEncoder, compute_loss, sam_loss


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
        self.num_basis = num_basis
        self.edsr_resblocks = edsr_resblocks
        self.arch_summary = (
            f"GSNOBackbone+v2Scatter: dim={dim}, ADCI={adci_layers} per stream, "
            f"fusion=conv0 1x1 stack, GS=v2 scatter * {num_gs_layers}, "
            "decoder=Conv3x3+ReLU+Conv3x3"
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

        self.gs_layers = nn.ModuleList([GSEncoder(dim) for _ in range(num_gs_layers)])

        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_bands, 3, padding=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def reset_custom_init(self):
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

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

        f_hsi = self.shallow_encoder1(lr_hsi)
        f_msi = self.shallow_encoder2(hr_msi)

        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        f_hsi = F.interpolate(f_hsi, scale_factor=sf, mode="bicubic", align_corners=False)

        feat = torch.cat([f_hsi, f_msi], dim=1)
        feat = self.conv0(feat)

        for layer in self.gs_layers:
            feat = layer(feat)

        residual = self.decoder(feat)
        return lr_hsi_up + residual
