"""Strict EDSR-only control for the current Gaussian experiments.

The input, EDSR reconstruction encoder, decoder, bicubic residual base, and
training loss match the paired EDSR Gaussian experiments.  The only functional
change is that no Gaussian module or Gaussian residual path exists.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.support.EDSR import make_edsr_baseline


def sam_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
    cosine = (pred * gt).sum(dim=1) / (
        pred.norm(dim=1) * gt.norm(dim=1) + eps
    )
    return (1.0 - cosine).mean()


def compute_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    epoch: int,
    sam_warmup_epochs: int = 5,
    sam_weight: float = 0.1,
):
    l1 = F.l1_loss(pred, gt)
    if epoch < sam_warmup_epochs:
        return l1
    weight = min(
        sam_weight,
        sam_weight * (epoch - sam_warmup_epochs + 1) / 5.0,
    )
    return l1 + weight * sam_loss(pred, gt)


class GSFusion(nn.Module):
    """Bicubic HSI + MSI -> EDSR -> decoder -> bicubic residual base."""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        n_resblocks: int = 6,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_msi = int(num_msi)
        self.n_resblocks = int(n_resblocks)

        self.edsr_encoder = make_edsr_baseline(
            n_resblocks=self.n_resblocks,
            n_feats=self.dim,
            n_colors=self.num_bands + self.num_msi,
            no_upsampling=True,
        )
        self.fc1 = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.fc2 = nn.Conv2d(self.dim, self.num_bands, kernel_size=1)

        self.arch_summary = (
            "raw bicubic HSI + HR-MSI concat -> EDSR 3x3 encoder "
            f"({self.n_resblocks} residual blocks) -> 1x1/GELU/1x1 decoder "
            "-> raw bicubic residual base; no Gaussian module or path"
        )
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        sf=None,
    ) -> torch.Tensor:
        del sf
        base = F.interpolate(
            lr_hsi,
            size=hr_msi.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        joint = torch.cat([hr_msi, base], dim=1)
        fused = self.edsr_encoder(joint)
        residual = self.fc2(F.gelu(self.fc1(fused)))
        return base + residual


__all__ = ["GSFusion", "compute_loss", "sam_loss"]
