"""Unified-protocol adapter for the official CVPR 2025 SRLF-Net.

The released CAVE/Harvard implementation is hard-coded for x8.  The authors'
GF5 implementation shows the same SRLF block parameterized for x2.  This
adapter applies that released scale-specific construction pattern at x4:
one stride-4 pyramid convolution, PixelShuffle(4), and x4 routing grids.  No
new learned module or cross-scale training is introduced.

At frozen x8/x16/x32 evaluation, LR-HSI is bicubically aligned to the x4
network's canonical quarter grid.  This non-learned adapter is explicit in
the evaluation artifact; SRLF-Net must not be described as natively
arbitrary-scale.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL = _ROOT / "external_baselines" / "SRLF_Net_official_a23bd193"
_RUNTIME = (
    _ROOT
    / "external_baselines"
    / "_runtime"
    / "srlfnet_mamba_ssm_2_2_3_post2_torch2_6_cu12"
)


def _fit_paired_response(dataset: str, data_path: Path):
    """Recover the exact fixed HSI-to-paired-MSI response from training only."""

    data_path = data_path.resolve()
    if dataset == "cave":
        split_path = data_path / "Train.txt"
        names = [name for name in split_path.read_text().splitlines() if name]
        if not names:
            raise ValueError(f"Empty CAVE training split: {split_path}")
        hsi = sio.loadmat(data_path / "HSI" / f"{names[0]}.mat")["hsi"]
        msi = sio.loadmat(data_path / "RGB" / f"{names[0]}.mat")["rgb"]
        stride = 2
    elif dataset == "harvard":
        paired = sio.loadmat(data_path / "1.mat")
        hsi, msi = paired["HS"], paired["HRMS"]
        stride = 4
    else:
        raise ValueError(f"Unsupported SRLF-Net dataset: {dataset!r}")

    x = np.asarray(hsi[::stride, ::stride, :], dtype=np.float64).reshape(-1, 31)
    y = np.asarray(msi[::stride, ::stride, :], dtype=np.float64).reshape(-1, 3)
    normal = x.T @ x
    ridge = 1e-12 * float(np.trace(normal)) / normal.shape[0]
    response = np.linalg.solve(normal + ridge * np.eye(31), x.T @ y)
    relative_error = float(np.linalg.norm(x @ response - y) / np.linalg.norm(y))
    if not np.isfinite(relative_error) or relative_error > 1e-6:
        raise ValueError(
            "Paired HSI/MSI is not explained by one fixed response: "
            f"dataset={dataset} relative_error={relative_error:.3e}"
        )
    return np.asarray(response.T, dtype=np.float32), relative_error


def _project_gaussian_psf(scale: int = 4, sigma: float = 2.0):
    """Match cv2.getGaussianKernel(scale, sigma) used by para_setting."""

    coordinate = np.arange(scale, dtype=np.float64) - (scale - 1.0) / 2.0
    kernel_1d = np.exp(-(coordinate**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return np.asarray(np.outer(kernel_1d, kernel_1d), dtype=np.float32)


def _install_official_import_context():
    if not _RUNTIME.is_dir():
        raise FileNotFoundError(f"Missing isolated SRLF-Net Mamba runtime: {_RUNTIME}")
    runtime = str(_RUNTIME)
    official = str(_OFFICIAL)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    if official not in sys.path:
        sys.path.insert(0, official)

    # Avoid mamba_ssm.__init__, which imports unrelated language-model extras.
    if "mamba_ssm" not in sys.modules:
        package = types.ModuleType("mamba_ssm")
        package.__path__ = [str(_RUNTIME / "mamba_ssm")]
        sys.modules["mamba_ssm"] = package

    # These packages are imported by the released files only for profiling or
    # unused helpers.  Stubbing them keeps the private runtime minimal.
    if "thop" not in sys.modules:
        thop = types.ModuleType("thop")
        thop.profile = None
        thop.clever_format = None
        sys.modules["thop"] = thop
    if "timm.models.layers" not in sys.modules:
        timm = types.ModuleType("timm")
        timm_models = types.ModuleType("timm.models")
        timm_layers = types.ModuleType("timm.models.layers")
        timm_layers.DropPath = nn.Identity
        timm_layers.to_2tuple = lambda value: (value, value)
        timm_layers.trunc_normal_ = nn.init.trunc_normal_
        timm.models = timm_models
        timm_models.layers = timm_layers
        sys.modules["timm"] = timm
        sys.modules["timm.models"] = timm_models
        sys.modules["timm.models.layers"] = timm_layers
    if "utils" not in sys.modules:
        utility_stub = types.ModuleType("utils")
        utility_stub.Gaussian_downsample = None
        utility_stub.fspecial = None
        utility_stub.create_F = None
        sys.modules["utils"] = utility_stub


def _load_official_module():
    _install_official_import_context()
    name = "_srlfnet_official_a23bd193"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _OFFICIAL / "deep_select_mamba.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load official SRLF-Net source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scale4_block_forward(self, fufea_hr, response, coeff, sf, width, gt_msi, gt_hsi):
    batch, channels, height, spatial_width = fufea_hr.shape
    feature_hr = self.mamba_64(fufea_hr)
    feature_lr = self.mamba_8(self.down(feature_hr))
    up_feature_lr = self.ps(self.up(feature_lr))

    pre_fusion = self.mambafusion(
        self.mambacat(torch.cat([feature_hr, up_feature_lr], dim=1))
    )
    pre_fusion = self.layernorm(pre_fusion.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    _, predicted_msi = self.to_hsi_and_msi(pre_fusion, response, coeff, sf, width)
    predicted_msi = predicted_msi.clamp(0, 1)

    ssim_score = self.ssim(predicted_msi, gt_msi)
    keep_ratio = 0.3 * (1 - self.to_r_ssim(ssim_score).reshape(-1))
    flat_ssim = ssim_score.reshape(batch, -1)
    ssim_index = torch.argsort(flat_ssim, dim=1, descending=False)
    keep_count = int(torch.mean(flat_ssim.shape[1] * keep_ratio))
    bad_ssim_index = ssim_index[:, :keep_count]

    msi_feature = self.pancat(torch.cat([pre_fusion, gt_msi], dim=1))
    selected_msi = self.ssim_refine(
        batch_index_select_proxy(
            msi_feature.reshape(batch, channels, -1).permute(0, 2, 1),
            bad_ssim_index,
        )
    )
    ssim_out = torch.zeros_like(fufea_hr)
    ssim_out = batch_index_fill_proxy(
        ssim_out.reshape(batch, channels, -1).permute(0, 2, 1),
        selected_msi,
        bad_ssim_index,
    )
    ssim_out = ssim_out.permute(0, 2, 1).reshape(
        batch, channels, height, spatial_width
    ) + msi_feature

    predicted_lr, _ = self.to_hsi_and_msi(ssim_out, response, coeff, sf, width)
    sam_score = self.sam(predicted_lr, gt_hsi).reshape(
        batch, height // 4, spatial_width // 4, -1
    )
    flat_sam = sam_score.reshape(batch, -1)
    sam_index = torch.argsort(flat_sam, dim=1, descending=True)
    keep_count = int(torch.mean(flat_sam.shape[1] * keep_ratio))
    bad_sam_index = sam_index[:, :keep_count]

    hsi_feature = self.hsicat(torch.cat([predicted_lr, gt_hsi], dim=1))
    selected_hsi = self.sam_refine(
        batch_index_select_proxy(
            hsi_feature.reshape(batch, channels, -1).permute(0, 2, 1),
            bad_sam_index,
        )
    )
    sam_out = batch_index_fill_proxy(
        torch.zeros_like(feature_lr).reshape(batch, channels, -1).permute(0, 2, 1),
        selected_hsi,
        bad_sam_index,
    )
    sam_out = sam_out.permute(0, 2, 1).reshape(
        batch, channels, height // 4, spatial_width // 4
    ) + hsi_feature
    sam_out = self.ps(self.up2(sam_out))

    fused = self.convout(torch.cat([ssim_out, sam_out], dim=1))
    relearned = self.mambafusion2(fused) + fused
    return self.convout2(torch.cat([relearned, fufea_hr], dim=1))


# Filled after the official module is loaded; keeping these proxies at module
# scope avoids copying or altering the released index helper implementations.
batch_index_select_proxy = None
batch_index_fill_proxy = None


class GSFusion(nn.Module):
    supported_scales = (4, 8, 16, 32)
    def __init__(
        self,
        num_bands: int = 31,
        num_msi: int = 3,
        dataset: str = "cave",
        calibration_data_path: str | Path | None = None,
        training_scale: int = 4,
    ):
        super().__init__()
        if (num_bands, num_msi) != (31, 3):
            raise ValueError("Official SRLF-Net requires 31 HSI and 3 MSI channels")
        if calibration_data_path is None:
            raise ValueError("SRLF-Net paired-response calibration path is required")
        self.training_scale = int(training_scale)
        if self.training_scale not in (4, 8):
            raise ValueError("Official SRLF-Net adapter supports x4 or native x8 training")
        self.inference_adapter = (
            f"official-style x{self.training_scale} scale parameterization; frozen unseen "
            f"scales use explicit non-learned bicubic alignment to the canonical "
            f"1/{self.training_scale} observation grid"
        )
        dataset = dataset.lower()
        response, response_error = _fit_paired_response(
            dataset, Path(calibration_data_path)
        )
        psf = _project_gaussian_psf(scale=self.training_scale, sigma=2.0)
        module = _load_official_module()
        global batch_index_select_proxy, batch_index_fill_proxy
        batch_index_select_proxy = module.batch_index_select
        batch_index_fill_proxy = module.batch_index_fill2

        self.net = module.SRLF_Net(
            31, torch.from_numpy(response), torch.from_numpy(psf), self.training_scale
        )
        del self.net.R
        del self.net.PSF
        self.net.register_buffer("R", torch.from_numpy(response))
        self.net.register_buffer("PSF", torch.from_numpy(psf))
        self.register_buffer(
            "response_calibration_relative_error",
            torch.tensor(response_error, dtype=torch.float32),
        )

        if self.training_scale == 4:
            for block in (self.net.icp1, self.net.icp2, self.net.icp3, self.net.icp4):
                block.down = nn.Sequential(
                    nn.Conv2d(31, 31, kernel_size=6, stride=4, padding=2, bias=False)
                )
                block.up = nn.Sequential(nn.Conv2d(31, 31 * 16, 1))
                block.ps = nn.PixelShuffle(4)
                block.down2 = nn.Sequential(
                    nn.Conv2d(31, 31, kernel_size=6, stride=4, padding=2, bias=False)
                )
                block.up2 = nn.Sequential(nn.Conv2d(31, 31 * 16, 1))
                block.ps2 = nn.PixelShuffle(4)
                block.forward = types.MethodType(_scale4_block_forward, block)

    def reset_custom_init(self):
        """Reproduce the explicit Xavier/LayerNorm initialization in train_cave.py."""

        for module in self.net.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

    def forward(self, lr_hsi, hr_msi, sf=4):
        scale = int(sf)
        if scale not in self.supported_scales:
            raise RuntimeError(f"Unsupported SRLF-Net evaluation scale: {scale}")
        canonical_size = (
            hr_msi.shape[-2] // self.training_scale,
            hr_msi.shape[-1] // self.training_scale,
        )
        if tuple(lr_hsi.shape[-2:]) != canonical_size:
            lr_hsi = F.interpolate(
                lr_hsi,
                size=canonical_size,
                mode="bicubic",
                align_corners=False,
            )
        return self.net(lr_hsi, hr_msi)


def compute_loss(prediction, target, epoch, **kwargs):
    del epoch, kwargs
    return F.l1_loss(prediction, target)


__all__ = ["GSFusion", "compute_loss"]
