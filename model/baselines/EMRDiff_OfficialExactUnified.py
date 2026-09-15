"""Unified-protocol adapter for the official CVPR 2026 EMR-Diff release.

The released trainer is hard-coded to 8x observations and 512-pixel targets.
This adapter preserves the released BAFUNet/diffusion/loss recipe while making
the non-learned HSI conditioning pyramid explicit and target-size based.  That
is the minimum interface change needed for 4x training and frozen evaluation.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "EMR_Diff_official_cc851ec_clean_20260828"


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


def _load_official_diffusion():
    """Load the released diffusion implementation without rewriting its math."""
    name = "_emrdiff_official_cc851ec_clean"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _OFFICIAL / "EMRDiff.py")
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load the clean official EMR-Diff snapshot")
    module = importlib.util.module_from_spec(spec)
    # The release hard-codes cuda:1.  Construct its fixed Sobel modules on CPU;
    # normal model.to(device) then moves them to the selected experiment GPU.
    module.device = torch.device("cpu")
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.device = torch.device("cpu")
    return module


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
        official = _load_official_diffusion()
        diffusion_config = {
            "params": {
                "sf": 8,
                "schedule_name": "exponential",
                "schedule_kwargs": {"power": 0.3},
                "etas_end": 0.999,
                "steps": 5,
                "min_noise_level": 0.001,
                "kappa": 2.0,
                "band_dim": 31,
                "normalize_input": False,
                "latent_flag": None,
            }
        }
        self.diffusion = official.EMRDIFF(diffusion_config)
        # Register the exact released residual calculator so ordinary
        # model.to(device) moves its fixed Sobel kernels with the network.
        self.training_residual = self.diffusion.residual_calculator
        self.inference_edge = official.Edge()
        self.steps = 5

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
        x_t = self.diffusion.forward_addnoise(
            x_start=x_start,
            y=condition,
            t=timesteps,
            noise=noise,
            rgb_hr=hr_msi,
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
        edge = self.inference_edge(hr_msi)
        generator = torch.Generator(device=condition.device)
        generator.manual_seed(20260820 + int(sf))
        noise = torch.randn(condition.shape, device=condition.device, dtype=condition.dtype, generator=generator)
        x_t = self.diffusion.prior_sample(condition, noise, edge_map=edge)
        for step in reversed(range(self.steps)):
            timesteps = torch.full((condition.shape[0],), step, device=condition.device, dtype=torch.long)
            residual, _ = self.net(x_t, hr_msi, hsi_hr, timesteps)
            x_start = residual + condition
            step_noise = torch.randn(condition.shape, device=condition.device, dtype=condition.dtype, generator=generator)
            x_t = self.diffusion.inverse_denoise(
                x_start=x_start,
                x_t=x_t,
                t=timesteps,
                noise=step_noise,
                edge_map=edge,
            )
        return x_t[:, :31]


def compute_loss(prediction, target, epoch, **kwargs):
    del target, epoch, kwargs
    if prediction.ndim != 0:
        raise RuntimeError("EMR-Diff training must use GSFusion.training_step")
    return prediction


__all__ = ["GSFusion", "compute_loss"]
