"""Unified-protocol wrapper around the authors' MIMO-SST implementation.

The upstream TGRS 2024 code hard-codes an 8x bilinear upsampler before the
scale-independent fusion backbone.  For the project's fair CAVE protocol we
perform the same bilinear interpolation to the *actual* HR-MSI size.  This is
the only scale-contract adaptation: the upstream Transformer, three-scale
decoder, and three-output L1+FFT objective are retained.

Upstream source snapshot:
  external_baselines/MIMO_SST_official_06aacc1/Model.py
  https://github.com/Freelancefangjian/MIMO-SST
  commit 06aacc193ad9951fb873359a29efc792f8c33dc3
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM_PATH = (
    _PROJECT_ROOT
    / "external_baselines"
    / "MIMO_SST_official_06aacc1"
    / "Model.py"
)


def _load_upstream_module():
    """Load the frozen upstream file with its one missing helper supplied."""
    if not _UPSTREAM_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen MIMO-SST source: {_UPSTREAM_PATH}")

    # The public repository imports PixelUnshuffle.py but does not include it.
    # Its Downsample call has the standard torch.pixel_unshuffle semantics.
    helper = types.ModuleType("PixelUnshuffle")
    helper.pixel_unshuffle = F.pixel_unshuffle
    sys.modules.setdefault("PixelUnshuffle", helper)

    module_name = "_mimo_sst_upstream_06aacc1"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _UPSTREAM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MIMO-SST source from {_UPSTREAM_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class GSFusion(nn.Module):
    """Expose MIMO-SST through the common ``(LR-HSI, HR-MSI, sf)`` API."""

    def __init__(self, num_bands: int = 31, num_msi: int = 3):
        super().__init__()
        if num_bands <= 0 or num_msi != 3:
            raise ValueError(f"MIMO-SST expects positive HSI bands and 3 MSI channels, got {num_bands}/{num_msi}")
        upstream = _load_upstream_module()

        # Upstream constructs one ReLU with `.cuda()` inside __init__.  Avoid a
        # hidden device allocation; the common trainer moves the complete model.
        original_cuda = nn.Module.cuda
        nn.Module.cuda = lambda module, *args, **kwargs: module
        try:
            self.net = upstream.Net()
        finally:
            nn.Module.cuda = original_cuda

        # The wrapper supplies a dynamically sized bilinear input below.
        self.net.upSample = nn.Identity()

        # The released network hard-codes 31 only in its HSI input/output
        # projections. Parameterize those boundary projections while leaving
        # every transformer and decoder block unchanged.
        if num_bands != 31:
            self.net.patch_embed = upstream.OverlapPatchEmbed(99 + num_bands, 48)
            self.net.Conv31_64 = nn.Conv2d(num_bands, 48, kernel_size=3, padding=1)
            self.net.Conv192_31 = nn.Conv2d(192, num_bands, kernel_size=3, padding=1)
            self.net.Conv96_31 = nn.Conv2d(96, num_bands, kernel_size=3, padding=1)

    def forward(self, lr_hsi, hr_msi, sf=4):
        del sf  # target size, rather than a hard-coded integer, defines the grid.
        lr_hsi_up = F.interpolate(
            lr_hsi,
            size=hr_msi.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse, middle, full = self.net(hr_msi, lr_hsi_up)
        if self.training:
            return coarse, middle, full
        return full


def compute_loss(outputs, target, epoch, **kwargs):
    """Authors' three-output L1 + 0.01 Fourier-magnitude objective."""
    del epoch, kwargs
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise TypeError("MIMO-SST training expects (coarse, middle, full) outputs")
    loss_l1 = target.new_zeros(())
    loss_fft = target.new_zeros(())
    for prediction in outputs:
        target_scale = F.interpolate(target, size=prediction.shape[-2:])
        loss_l1 = loss_l1 + F.l1_loss(prediction, target_scale)
        loss_fft = loss_fft + torch.mean(
            torch.abs(torch.fft.fft2(prediction) - torch.fft.fft2(target_scale))
        )
    return loss_l1 + 0.01 * loss_fft
