"""Unified CAVE/Harvard adapter for the official BHSR-Net release.

Upstream: https://github.com/Dou0405/BHSR-Net
Commit: d5fa43ae36b2ac2831856cbace0de1c9116a8749

The learnable ten-stage model and released composite loss are preserved.  Only
the dataset/tensor interface is adapted to the common GSNO experiment driver.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch.nn as nn


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "BHSR_Net_official_d5fa43a"


def _load_source(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]
    source = _OFFICIAL / filename
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen BHSR-Net source: {source}")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load BHSR-Net source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_modules():
    # The release uses ``from modules import *``.  Register that exact source
    # under its expected import name without changing the upstream file.
    modules = _load_source("_bhsr_official_modules_d5fa43a", "modules.py")
    previous = sys.modules.get("modules")
    sys.modules["modules"] = modules
    try:
        models = _load_source("_bhsr_official_models_d5fa43a", "models.py")
    finally:
        if previous is None:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = previous
    losses = _load_source("_bhsr_official_loss_d5fa43a", "loss.py")
    return models, losses


class GSFusion(nn.Module):
    """Expose the released full ten-stage BHSR-Net through the common API."""

    supported_scales = (4, 8, 16, 32)
    inference_adapter = (
        "official BHSR-Net ten-stage core; its released parameter-free bilinear "
        "LR-HSI alignment uses the actual HR-MSI target grid"
    )

    def __init__(
        self,
        num_bands: int = 31,
        num_msi: int = 3,
        training_scale: int = 4,
        stages: int = 10,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
    ):
        super().__init__()
        if num_bands <= 0 or num_msi != 3:
            raise ValueError(f"BHSR-Net expects positive HSI bands and 3 MSI channels, got {num_bands}/{num_msi}")
        if int(training_scale) not in self.supported_scales:
            raise ValueError(f"Unsupported BHSR-Net training scale: {training_scale}")
        models, losses = _load_official_modules()
        self.training_scale = int(training_scale)
        self.current_epoch = 0
        # The released train.py says stage_num, but the released constructor is
        # stage_nBHSR.  This is the sole typo correction; the class is unchanged.
        self.net = models.BHSR(
            stage_nBHSR=int(stages),
            C=num_bands,
            c=num_msi,
            sigma1=float(sigma1),
            sigma2=float(sigma2),
        )
        self.criterion = losses.Losses(
            scale=self.training_scale,
            model_name="BHSR",
            blur=0,
        )

    def set_training_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _check_grid(self, lr_hsi, hr_msi, scale: int) -> None:
        expected = (lr_hsi.shape[-2] * scale, lr_hsi.shape[-1] * scale)
        if tuple(hr_msi.shape[-2:]) != expected:
            raise RuntimeError(
                f"BHSR-Net x{scale} expects HR-MSI grid {expected}, got "
                f"{tuple(hr_msi.shape[-2:])}"
            )

    def forward(self, lr_hsi, hr_msi, sf=4):
        scale = int(sf)
        if scale not in self.supported_scales:
            raise RuntimeError(f"Unsupported BHSR-Net scale: {scale}")
        self._check_grid(lr_hsi, hr_msi, scale)
        stage_outputs, _corrected_msi = self.net(lr_hsi, hr_msi)
        return stage_outputs[-1]

    def training_step(self, lr_hsi, hr_msi, target, sf):
        scale = int(sf)
        if scale != self.training_scale:
            raise RuntimeError(
                f"BHSR-Net was configured for x{self.training_scale} training, got x{scale}"
            )
        self._check_grid(lr_hsi, hr_msi, scale)
        stage_outputs, corrected_msi = self.net(lr_hsi, hr_msi)
        return self.criterion(
            stage_outputs,
            target,
            corrected_msi,
            hr_msi,
            self.current_epoch,
        )

    def validation_step(self, lr_hsi, hr_msi, target, sf):
        """Return the prediction and the released composite validation loss."""
        scale = int(sf)
        if scale != self.training_scale:
            raise RuntimeError(
                f"BHSR-Net was configured for x{self.training_scale} validation, "
                f"got x{scale}"
            )
        self._check_grid(lr_hsi, hr_msi, scale)
        stage_outputs, corrected_msi = self.net(lr_hsi, hr_msi)
        loss = self.criterion(
            stage_outputs,
            target,
            corrected_msi,
            hr_msi,
            self.current_epoch,
        )
        return stage_outputs[-1], loss


__all__ = ["GSFusion"]
