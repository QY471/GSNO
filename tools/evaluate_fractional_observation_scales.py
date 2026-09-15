#!/usr/bin/env python3
"""Evaluate frozen GSNO/PSRT/DSPNet on fractional CAVE observation grids.

This is a separate antialiased target-grid protocol. It does not claim to be
bitwise identical to the legacy integer FFT blur-plus-stride degradation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--gsno-checkpoint", required=True)
    parser.add_argument("--psrt-model-source", required=True)
    parser.add_argument("--psrt-checkpoint", required=True)
    parser.add_argument("--dspnet-model-source", required=True)
    parser.add_argument("--dspnet-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requested-scales", nargs="+", type=float, default=[3.2, 4.0, 5.7, 8.0])
    parser.add_argument("--gsno-dim", type=int, default=80)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_from_source(module_name: str, source: Path, add_root: Path | None = None):
    if add_root is not None and str(add_root) not in sys.path:
        sys.path.insert(0, str(add_root))
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def clean_state(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(type(payload).__name__)
    return {key.removeprefix("module.").removeprefix("model."): value for key, value in payload.items()}


def load_models(args, device):
    gsno_module = importlib.import_module(
        "model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous"
    )
    gsno = gsno_module.GSFusion(dim=args.gsno_dim, num_bands=31, num_msi=3, adci_layers=3)
    gsno.load_state_dict(clean_state(torch.load(args.gsno_checkpoint, map_location="cpu")), strict=True)

    psrt_source = Path(args.psrt_model_source).resolve()
    psrt_module = import_from_source("psrt_official_model_sr", psrt_source, psrt_source.parent.parent)
    psrt = psrt_module.PSRTnet(SimpleNamespace())
    psrt_payload = torch.load(args.psrt_checkpoint, map_location="cpu")
    if not isinstance(psrt_payload, dict) or "state_dict" not in psrt_payload:
        raise ValueError("PSRT checkpoint must contain state_dict")
    psrt.load_state_dict(clean_state(psrt_payload["state_dict"]), strict=True)

    dspnet_source = Path(args.dspnet_model_source).resolve()
    import_from_source("DSPNet", dspnet_source)
    try:
        dspnet = torch.load(args.dspnet_checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        dspnet = torch.load(args.dspnet_checkpoint, map_location="cpu")
    if not isinstance(dspnet, torch.nn.Module):
        raise TypeError("DSPNet checkpoint must contain a serialized model")
    return {
        "GSNO formal x4 frozen": gsno.to(device).eval(),
        "PSRT official x4 frozen": psrt.to(device).eval(),
        "DSPNet official x4 frozen plus canonical-quarter adapter": dspnet.to(device).eval(),
    }


def load_scenes(data_path: str):
    from tools.Utils import loadpath, prepare_data

    names = loadpath(os.path.join(data_path, "Test.txt"), shuffle=False)
    hr_hsi, hr_msi = prepare_data(data_path, names, len(names))
    scenes = []
    for index, raw_name in enumerate(names):
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        hsi = torch.from_numpy(np.ascontiguousarray(hr_hsi[..., index].transpose(2, 0, 1))).float()
        msi = torch.from_numpy(np.ascontiguousarray(hr_msi[..., index].transpose(2, 0, 1))).float()
        scenes.append((Path(name).stem, hsi, msi))
    return scenes


def target_lr_size(hr_size: tuple[int, int], requested_scale: float):
    if requested_scale <= 1.0:
        raise ValueError(f"requested scale must exceed one, got {requested_scale}")
    target = tuple(max(1, int(round(value / requested_scale))) for value in hr_size)
    actual = tuple(value / low for value, low in zip(hr_size, target))
    if abs(actual[0] - actual[1]) > 1e-12:
        raise ValueError(f"anisotropic actual scales are unsupported: {actual}")
    return target, actual[0]


def degrade_antialiased(hr_hsi: torch.Tensor, lr_size: tuple[int, int]):
    return F.interpolate(
        hr_hsi.unsqueeze(0),
        size=lr_size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def stable_sam(pred: np.ndarray, target: np.ndarray) -> float:
    channels = pred.shape[-1]
    left = pred.reshape(-1, channels)
    right = target.reshape(-1, channels)
    cosine = np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-7
    )
    return float(np.mean(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))))


def forward_method(label, model, lr_hsi, hr_msi, actual_scale):
    if label.startswith("GSNO"):
        return model(lr_hsi, hr_msi, actual_scale)
    if label.startswith("PSRT"):
        lr_up = F.interpolate(lr_hsi, size=hr_msi.shape[-2:], mode="bicubic", align_corners=False)
        return model(hr_msi, lr_up)
    canonical = F.interpolate(
        lr_hsi,
        size=(hr_msi.shape[-2] // 4, hr_msi.shape[-1] // 4),
        mode="bicubic",
        align_corners=False,
    )
    return model(canonical, hr_msi)


@torch.inference_mode()
def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if len(set(args.requested_scales)) != len(args.requested_scales):
        raise ValueError("requested scales must be unique")

    from tools.Utils import cal_psnr, compute_ergas, compute_ssim

    output = Path(args.output).resolve()
    csv_output = output.with_suffix(".per_image.csv")
    if output.exists() or csv_output.exists():
        raise FileExistsError("refusing to overwrite existing artifacts")
    device = torch.device("cuda")
    models = load_models(args, device)
    scenes = load_scenes(args.data_path)
    rows = []
    grid_records = []
    for requested_scale in args.requested_scales:
        lr_size, actual_scale = target_lr_size((512, 512), requested_scale)
        grid_records.append({
            "requested_scale": requested_scale,
            "lr_size": list(lr_size),
            "actual_scale": actual_scale,
            "absolute_scale_error": abs(actual_scale - requested_scale),
        })
        for scene_index, (scene_name, hr_hsi_cpu, hr_msi_cpu) in enumerate(scenes, start=1):
            lr_hsi = degrade_antialiased(hr_hsi_cpu, lr_size).to(device)
            hr_hsi = hr_hsi_cpu.unsqueeze(0).to(device)
            hr_msi = hr_msi_cpu.unsqueeze(0).to(device)
            target = hr_hsi_cpu.numpy().transpose(1, 2, 0)
            for label, model in models.items():
                prediction = forward_method(label, model, lr_hsi, hr_msi, actual_scale).clamp(0.0, 1.0)
                if tuple(prediction.shape) != tuple(hr_hsi.shape):
                    raise RuntimeError(f"{label} output {tuple(prediction.shape)} != {tuple(hr_hsi.shape)}")
                pred = prediction[0].cpu().numpy().transpose(1, 2, 0)
                row = {
                    "method": label,
                    "requested_scale": requested_scale,
                    "actual_scale": actual_scale,
                    "lr_height": lr_size[0],
                    "lr_width": lr_size[1],
                    "scene_index": scene_index,
                    "scene_name": scene_name,
                    "psnr": float(cal_psnr(pred, target)),
                    "sam": stable_sam(pred, target),
                    "ergas_actual_scale": float(compute_ergas(pred, target, actual_scale)),
                    "ssim": float(compute_ssim(pred, target)),
                }
                if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
                    raise FloatingPointError(row)
                rows.append(row)
            print(
                f"scale={requested_scale:g} actual={actual_scale:.8f} "
                f"scene={scene_index}/{len(scenes)}",
                flush=True,
            )

    summaries = []
    for label in models:
        for grid in grid_records:
            selected = [
                row for row in rows
                if row["method"] == label and row["requested_scale"] == grid["requested_scale"]
            ]
            summaries.append({
                "method": label,
                **grid,
                "images": len(selected),
                **{
                    metric: float(np.mean([row[metric] for row in selected], dtype=np.float64))
                    for metric in ("psnr", "sam", "ergas_actual_scale", "ssim")
                },
            })

    checkpoint_paths = {
        "GSNO formal x4 frozen": Path(args.gsno_checkpoint).resolve(),
        "PSRT official x4 frozen": Path(args.psrt_checkpoint).resolve(),
        "DSPNet official x4 frozen plus canonical-quarter adapter": Path(args.dspnet_checkpoint).resolve(),
    }
    report = {
        "protocol": {
            "name": "CAVE fixed-HR fractional observation target-grid diagnostic",
            "hr_msi_gt_output_size": [512, 512],
            "lr_grid_rule": "round(512/requested_scale) per axis",
            "degradation": "PyTorch bicubic resize with antialias=True from native GT-HSI",
            "legacy_integer_fft_protocol_equivalent": False,
            "interpretation": (
                "separate antialiased continuous-grid observation diagnostic; "
                "integer anchor rows quantify its difference from legacy FFT blur-plus-stride"
            ),
            "training_or_tuning": False,
            "dspnet_adapter": "frozen canonical-quarter input adapter; official forward unchanged",
        },
        "grids": grid_records,
        "checkpoints": {
            label: {"path": str(path), "sha256": sha256(path)}
            for label, path in checkpoint_paths.items()
        },
        "summaries": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output}", flush=True)
    print(f"WROTE {csv_output}", flush=True)


if __name__ == "__main__":
    main()
