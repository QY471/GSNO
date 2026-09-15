"""Unified-protocol adapter for the official CVPR 2026 EMR-Diff release.

The released trainer is hard-coded to 8x observations and 512-pixel targets.
This adapter preserves the released BAFUNet/diffusion/loss recipe while making
the non-learned HSI conditioning pyramid explicit and target-size based.  That
is the minimum interface change needed for 4x training and frozen evaluation.
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "EMR_Diff_official_cc851ec_src"


def _load_bafunet():
    official_path = str(_OFFICIAL)
    if official_path not in sys.path:
        sys.path.insert(0, official_path)
    # The release imports only ``Mlp`` from its bundled Swin file, while that
    # file pulls in the undeclared optional dependency ``timm``.  Supply the
    # same small feed-forward block locally so the official BAFUNet can run in
    # the shared, unmodified ``sqy`` environment.
    if "arch.swin_transformer" not in sys.modules:
        compatibility = types.ModuleType("arch.swin_transformer")

        class Mlp(nn.Module):
            def __init__(
                self,
                in_features,
                hidden_features=None,
                out_features=None,
                act_layer=nn.GELU,
                drop=0.0,
            ):
                super().__init__()
                hidden_features = hidden_features or in_features
                out_features = out_features or in_features
                self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
                self.act = act_layer()
                self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
                self.drop = nn.Dropout(drop)

            def forward(self, inputs):
                outputs = self.drop(self.act(self.fc1(inputs)))
                return self.drop(self.fc2(outputs))

        compatibility.Mlp = Mlp
        sys.modules["arch.swin_transformer"] = compatibility
    return importlib.import_module("arch.BAFUnet").BAFUNet


def _schedule(steps=5, power=0.3, min_noise=0.001, kappa=2.0, end=0.999):
    start = min(min_noise / kappa, min_noise)
    increaser = math.exp(math.log(end / start) / (steps - 1))
    base = np.ones((steps,), dtype=np.float64) * increaser
    positions = np.linspace(0, 1, steps, endpoint=True) ** power * (steps - 1)
    sqrt_etas = np.power(base, positions) * start
    etas = sqrt_etas**2
    previous = np.append(0.0, etas[:-1])
    alpha = etas - previous
    posterior_variance = kappa**2 * previous / etas * alpha
    posterior_variance_clipped = np.append(posterior_variance[1], posterior_variance[1:])
    return (
        etas,
        sqrt_etas,
        previous / etas,
        alpha / etas,
        np.log(posterior_variance_clipped),
    )


def _take(values, timesteps, reference):
    result = values[timesteps]
    while result.ndim < reference.ndim:
        result = result[..., None]
    return result.expand_as(reference)


class GSFusion(nn.Module):
    def __init__(self, num_bands: int = 31, num_msi: int = 3):
        super().__init__()
        if (num_bands, num_msi) != (31, 3):
            raise ValueError("Official EMR-Diff release requires 31/3 channels")
        model_class = _load_bafunet()
        self.net = model_class(
            image_size=512,
            in_channels=34,
            model_channels=34,
            out_channels=34,
            channel_mult=[1, 1, 1, 1, 1],
            num_res_blocks=[1, 1, 1, 1, 1],
            dims=2,
            lqrgb_channels=34,
        )
        arrays = _schedule()
        for name, array in zip(
            ("etas", "sqrt_etas", "posterior_c1", "posterior_c2", "posterior_logvar"),
            arrays,
        ):
            self.register_buffer(name, torch.tensor(array, dtype=torch.float32))
        self.steps = 5
        self.kappa = 2.0
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _edge(self, rgb):
        gray = rgb.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        flat = magnitude.flatten(1)
        low = flat.min(dim=1).values[:, None, None, None]
        high = flat.max(dim=1).values[:, None, None, None]
        return (magnitude - low) / (high - low).clamp_min(1e-12) + 1.0

    @staticmethod
    def _resize_hsi(lr_hsi, size):
        if tuple(lr_hsi.shape[-2:]) == tuple(size):
            return lr_hsi
        return F.interpolate(lr_hsi, size=size, mode="bicubic", align_corners=False)

    def _conditioners(self, lr_hsi, hr_msi):
        height, width = hr_msi.shape[-2:]
        hsi_hr = self._resize_hsi(lr_hsi, (height, width))
        pyramid = {
            factor: self._resize_hsi(lr_hsi, (height // factor, width // factor))
            for factor in (8, 4, 2)
        }
        return hsi_hr, pyramid

    def training_step(self, lr_hsi, hr_msi, target, sf=4):
        del sf
        hsi_hr, pyramid = self._conditioners(lr_hsi, hr_msi)
        condition = torch.cat((hsi_hr, hr_msi), dim=1)
        x_start = torch.cat((target, target[:, :3]), dim=1)
        timesteps = torch.randint(0, self.steps, (target.shape[0],), device=target.device)
        noise = torch.randn_like(condition)
        edge = self._edge(hr_msi)
        x_t = (
            _take(self.etas, timesteps, x_start) * (condition - x_start)
            + x_start
            + _take(self.sqrt_etas * self.kappa, timesteps, x_start) * noise * edge
        )
        prediction, intermediate = self.net(x_t, hr_msi, hsi_hr, timesteps)
        losses = [F.l1_loss(prediction + condition, x_start)]
        for output_index, factor in ((2, 8), (4, 4), (6, 2)):
            target_level = x_start[..., ::factor, ::factor]
            rgb_level = hr_msi[..., ::factor, ::factor]
            condition_level = torch.cat((pyramid[factor], rgb_level), dim=1)
            losses.append(F.l1_loss(intermediate[output_index] + condition_level, target_level))
        return sum(losses)

    def forward(self, lr_hsi, hr_msi, sf=4):
        hsi_hr, _ = self._conditioners(lr_hsi, hr_msi)
        condition = torch.cat((hsi_hr, hr_msi), dim=1)
        edge = self._edge(hr_msi)
        generator = torch.Generator(device=condition.device)
        generator.manual_seed(20260820 + int(sf))
        noise = torch.randn(condition.shape, device=condition.device, dtype=condition.dtype, generator=generator)
        last_t = torch.full((condition.shape[0],), self.steps - 1, device=condition.device, dtype=torch.long)
        x_t = condition + _take(self.kappa * self.sqrt_etas, last_t, condition) * noise * edge
        for step in reversed(range(self.steps)):
            timesteps = torch.full((condition.shape[0],), step, device=condition.device, dtype=torch.long)
            residual, _ = self.net(x_t, hr_msi, hsi_hr, timesteps)
            x_start = residual + condition
            step_noise = torch.randn(condition.shape, device=condition.device, dtype=condition.dtype, generator=generator)
            nonzero = (timesteps != 0).float()[:, None, None, None]
            x_t = (
                _take(self.posterior_c1, timesteps, x_t) * x_t
                + _take(self.posterior_c2, timesteps, x_t) * x_start
                + nonzero * torch.exp(0.5 * _take(self.posterior_logvar, timesteps, x_t)) * step_noise * edge
            )
        return x_t[:, :31]


def compute_loss(prediction, target, epoch, **kwargs):
    del target, epoch, kwargs
    if prediction.ndim != 0:
        raise RuntimeError("EMR-Diff training must use GSFusion.training_step")
    return prediction


__all__ = ["GSFusion", "compute_loss"]
