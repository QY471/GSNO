"""Utilities used by the GSNO training and evaluation pipelines."""

from __future__ import annotations

import glob
import os
import random
import re

import cv2
import numpy as np
import scipy.io as sio
import torch
from pypher import pypher as Pypher
from skimage.metrics import structural_similarity


def compute_ssim(im1: np.ndarray, im2: np.ndarray) -> float:
    """Average SSIM over the spectral channels of two HWC images."""

    scores = [
        structural_similarity(im1[:, :, channel], im2[:, :, channel], data_range=1.0)
        for channel in range(im1.shape[2])
    ]
    return float(np.mean(scores))


def cal_psnr(im1: np.ndarray, im2: np.ndarray) -> float:
    """Compute the mean per-band PSNR for two HWC images in [0, 1]."""

    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    mse = np.mean(np.square(im1 - im2), axis=0)
    return float(np.mean(10 * np.log10(1 / mse)))


def compute_ergas(out: np.ndarray, gt: np.ndarray, scale: int) -> float:
    """Compute ERGAS for two HWC images."""

    num_spectral = out.shape[-1]
    out = np.reshape(out, (-1, num_spectral))
    gt = np.reshape(gt, (-1, num_spectral))
    mse = np.mean(np.square(gt - out), axis=0)
    gt_mean = np.mean(gt, axis=0)
    return float(100 / scale * np.sqrt(np.mean(mse / (gt_mean ** 2 + 1e-6))))


def compute_sam(im1: np.ndarray, im2: np.ndarray) -> float:
    """Compute the mean spectral angle in degrees for two HWC images."""

    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    numerator = np.sum(im1 * im2, axis=1)
    denominator = np.linalg.norm(im1, axis=1) * np.linalg.norm(im2, axis=1)
    angles = np.rad2deg(np.arccos(numerator / (denominator + 1e-7)))
    return float(np.mean(angles))


def para_setting(kernel_type: str, sf: int, sz: list[int], sigma: float):
    """Build the blur PSF and its optical transfer function."""

    if kernel_type == "uniform_blur":
        psf = np.ones([sf, sf]) / (sf * sf)
    elif kernel_type == "gaussian_blur":
        kernel = cv2.getGaussianKernel(sf, sigma)
        psf = kernel @ kernel.T
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    fft_b = Pypher.psf2otf(psf, sz)
    return fft_b, np.conj(fft_b)


def dataparallel(model: torch.nn.Module, ngpus: int, gpu0: int = 0):
    """Move a model to CUDA and wrap it when multiple GPUs are requested."""

    if ngpus <= 0:
        raise ValueError("GSNO training requires at least one GPU")
    gpu_list = list(range(gpu0, gpu0 + ngpus))
    if torch.cuda.device_count() < gpu0 + ngpus:
        raise RuntimeError(f"Requested {ngpus} GPU(s), but fewer are available")
    if ngpus > 1 and not isinstance(model, torch.nn.DataParallel):
        return torch.nn.DataParallel(model, gpu_list).cuda()
    return model.cuda()


def findLastCheckpoint(save_dir: str) -> int:
    """Return the largest epoch encoded by a ``model_<epoch>.pth`` file."""

    epochs = []
    for path in glob.glob(os.path.join(save_dir, "model_*.pth")):
        match = re.search(r"model_(\d+)\.pth$", path)
        if match:
            epochs.append(int(match.group(1)))
    return max(epochs, default=0)


def prepare_data(path: str, file_list: list[str], file_num: int | None = None):
    """Load CAVE HSI/RGB MAT files into channel-last arrays."""

    file_num = len(file_list) if file_num is None else file_num
    if file_num <= 0 or file_num > len(file_list):
        raise ValueError(f"file_num={file_num} is outside the file-list range")

    names = file_list[:file_num]
    first_hsi_path = os.path.join(path, "HSI", names[0] + ".mat")
    first_rgb_path = os.path.join(path, "RGB", names[0] + ".mat")
    first_hsi = sio.loadmat(first_hsi_path)["hsi"]
    first_rgb = sio.loadmat(first_rgb_path)["rgb"]

    if first_hsi.ndim != 3 or first_hsi.shape[-1] != 31:
        raise ValueError(f"Expected HSI HxWx31 in {first_hsi_path}, got {first_hsi.shape}")
    if first_rgb.ndim != 3 or first_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB HxWx3 in {first_rgb_path}, got {first_rgb.shape}")
    if first_hsi.shape[:2] != first_rgb.shape[:2]:
        raise ValueError(
            f"HSI/RGB spatial size mismatch for {names[0]}: "
            f"{first_hsi.shape[:2]} vs {first_rgb.shape[:2]}"
        )

    height, width, hsi_channels = first_hsi.shape
    msi_channels = first_rgb.shape[-1]
    hr_hsi = np.empty((height, width, hsi_channels, file_num), dtype=np.float32)
    hr_msi = np.empty((height, width, msi_channels, file_num), dtype=np.float32)
    hr_hsi[:, :, :, 0] = first_hsi.astype(np.float32)
    hr_msi[:, :, :, 0] = first_rgb.astype(np.float32)

    for index, name in enumerate(names[1:], start=1):
        hsi_path = os.path.join(path, "HSI", name + ".mat")
        msi_path = os.path.join(path, "RGB", name + ".mat")
        hsi = sio.loadmat(hsi_path)["hsi"]
        msi = sio.loadmat(msi_path)["rgb"]
        if hsi.shape != hr_hsi.shape[:3]:
            raise ValueError(f"HSI shape mismatch in {hsi_path}: {hsi.shape}")
        if msi.shape != hr_msi.shape[:3]:
            raise ValueError(f"RGB shape mismatch in {msi_path}: {msi.shape}")
        hr_hsi[:, :, :, index] = hsi.astype(np.float32)
        hr_msi[:, :, :, index] = msi.astype(np.float32)

    return hr_hsi, hr_msi


def loadpath(pathlistfile: str, shuffle: bool = True) -> list[str]:
    """Read scene names from a text file, optionally shuffling them."""

    with open(pathlistfile, encoding="utf-8") as handle:
        pathlist = handle.read().splitlines()
    if shuffle:
        random.shuffle(pathlist)
    return pathlist


def make_coord(shape: tuple[int, int], ranges=None, flatten: bool = True) -> torch.Tensor:
    """Create coordinates at grid centers in the normalized [-1, 1] domain."""

    coordinate_sequences = []
    for index, size in enumerate(shape):
        start, end = (-1, 1) if ranges is None else ranges[index]
        step = (end - start) / (2 * size)
        coordinate_sequences.append(
            start + step + (2 * step) * torch.arange(size).float()
        )
    coordinates = torch.stack(
        torch.meshgrid(*coordinate_sequences, indexing="ij"), dim=-1
    )
    return coordinates.view(-1, coordinates.shape[-1]) if flatten else coordinates
