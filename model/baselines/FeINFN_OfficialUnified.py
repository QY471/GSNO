"""Minimal unified-protocol adapter for the official NeurIPS 2024 FeINFN."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "Efficient_MIF_official_134434a_src"


def _load_model_class():
    project_model = importlib.import_module("model")
    official_model_path = str(_OFFICIAL / "model")
    if official_model_path not in project_model.__path__:
        project_model.__path__.append(official_model_path)
    module = importlib.import_module("model.FeINFN")
    return module.FeINFNet


def _gaussian(window_size=11, sigma=1.5):
    values = [
        math.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
        for x in range(window_size)
    ]
    window = torch.tensor(values, dtype=torch.float32)
    return window / window.sum()


def _ssim_loss(prediction, target, window_size=11, sigma=1.5):
    one_d = _gaussian(window_size, sigma).to(prediction)
    two_d = one_d[:, None].mm(one_d[None, :])[None, None]
    channels = prediction.shape[1]
    window = two_d.expand(channels, 1, window_size, window_size).contiguous()
    mu1 = F.conv2d(prediction, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1 = F.conv2d(
        prediction.square(), window, padding=window_size // 2, groups=channels
    ) - mu1_sq
    sigma2 = F.conv2d(
        target.square(), window, padding=window_size // 2, groups=channels
    ) - mu2_sq
    sigma12 = F.conv2d(
        prediction * target, window, padding=window_size // 2, groups=channels
    ) - mu12
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return 1 - score.mean()


class GSFusion(nn.Module):
    def __init__(self, num_bands: int = 31, num_msi: int = 3):
        super().__init__()
        if num_bands <= 0 or num_msi != 3:
            raise ValueError(f"FeINFN expects positive HSI bands and 3 MSI channels, got {num_bands}/{num_msi}")
        model_class = _load_model_class()
        self.net = model_class(
            hsi_dim=num_bands,
            msi_dim=num_msi,
            feat_dim=128,
            guide_dim=128,
            spa_edsr_num=4,
            spe_edsr_num=4,
            mlp_dim=[256, 128],
            # Official 31-band setting uses 33: two auxiliary mixture logits
            # are retained, so the channel-general form is bands + 2.
            NIR_dim=num_bands + 2,
            d_model=2,
            scale=4,
            patch_merge=False,
        )

    def forward(self, lr_hsi, hr_msi, sf=4):
        del sf
        lms = F.interpolate(
            lr_hsi,
            size=hr_msi.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        return self.net._forward_implem(hr_msi, lms, lr_hsi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    # Official ``l1ssim`` recipe: L1 + 0.1 * (1 - SSIM).
    return F.l1_loss(prediction, target) + 0.1 * _ssim_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
