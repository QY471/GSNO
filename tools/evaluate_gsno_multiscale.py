#!/usr/bin/env python3
"""Evaluate a frozen GSNO checkpoint on CAVE or Harvard at multiple ratios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.CAVE_Dataset import cave_dataset
from datasets.Harvard_Dataset import harvard_dataset, prepare_data_harvard
from tools.Utils import (
    cal_psnr,
    compute_ergas,
    compute_ssim,
    loadpath,
    prepare_data,
)


def stable_compute_sam(im1: np.ndarray, im2: np.ndarray) -> float:
    """Protocol-compatible SAM with a numerically safe arccos domain."""
    spectral_channels = im1.shape[-1]
    left = np.reshape(im1, (-1, spectral_channels))
    right = np.reshape(im2, (-1, spectral_channels))
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1)) * np.sqrt(
        np.sum(right * right, axis=1)
    )
    cosine = numerator / (denominator + 1e-7)
    return float(np.mean(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="model.gsno", choices=["model.gsno"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["cave", "harvard"], default="cave")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dim",
        type=int,
        default=80,
        help="Latent channel width used to construct the frozen model.",
    )
    parser.add_argument("--selection-scale", type=int, default=4)
    parser.add_argument(
        "--selected-best-epoch", "--selected-4x-best-epoch",
        dest="selected_best_epoch", type=int, required=True,
    )
    parser.add_argument(
        "--selected-best-psnr", "--selected-4x-best-psnr",
        dest="selected_best_psnr", type=float, required=True,
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowed difference between frozen selection-scale PSNR and the logged best.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    module_name: str,
    checkpoint: str,
    dim: int,
    device: torch.device,
    model_kwargs: dict,
):
    module = importlib.import_module(module_name)
    constructor_kwargs = {
        "dim": dim,
        "num_bands": 31,
        "num_msi": 3,
        **model_kwargs,
    }
    constructor_kwargs.setdefault("lki_layers", 3)
    model = module.GSNO(**constructor_kwargs)
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def make_loader(dataset_name: str, data_path: str, scale: int):
    if dataset_name == "cave":
        names = loadpath(os.path.join(data_path, "Test.txt"), shuffle=False)
        hr_hsi, hr_msi = prepare_data(data_path, names, len(names))
        dataset_class = cave_dataset
    elif dataset_name == "harvard":
        names = [f"{index}.mat" for index in range(1, 11)]
        hr_hsi, hr_msi = prepare_data_harvard(data_path, len(names))
        dataset_class = harvard_dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    options = argparse.Namespace(
        data_path=data_path,
        sizeI=None,
        testset_num=len(names),
        batch_size=1,
        sf=scale,
        seed=1,
        kernel_type="gaussian_blur",
    )
    dataset = dataset_class(options, hr_hsi, hr_msi, istrain=False)
    loader = tud.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return names, loader


def image_name(raw_name, index: int) -> str:
    value = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    return Path(value).stem or f"image_{index:02d}"


@torch.inference_mode()
def evaluate(model, names, loader, scale: int, device: torch.device):
    samples = {
        "psnr": [],
        "sam": [],
        "ergas_reference4": [],
        "ergas_actual_scale": [],
        "ssim": [],
    }
    per_image = []
    for index, batch in enumerate(loader, start=1):
        lr_hsi, hr_msi, hr_hsi = batch[:3]
        prediction = model(
            lr_hsi.to(device).float(), hr_msi.to(device).float(), scale
        ).clamp(0.0, 1.0)
        pred = prediction[0].cpu().numpy().transpose(1, 2, 0)
        target = hr_hsi[0].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        row = {
            "scale": scale,
            "image_index": index,
            "image_name": image_name(names[index - 1], index),
            "psnr": float(cal_psnr(pred, target)),
            "sam": stable_compute_sam(pred, target),
            "ergas_reference4": float(compute_ergas(pred, target, 4)),
            "ergas_actual_scale": float(compute_ergas(pred, target, scale)),
            "ssim": float(compute_ssim(pred, target)),
        }
        for metric_name in (
            "psnr",
            "sam",
            "ergas_reference4",
            "ergas_actual_scale",
            "ssim",
        ):
            if not np.isfinite(row[metric_name]):
                raise FloatingPointError(
                    f"non-finite {metric_name} at scale {scale}, image {index}"
                )
        per_image.append(row)
        for key in samples:
            samples[key].append(row[key])
        print(f"scale={scale} image={index}/{len(loader)}", flush=True)
    summary = {
        key: float(np.asarray(values, dtype=np.float64).mean())
        for key, values in samples.items()
    } | {"images": len(loader)}
    if not all(np.isfinite(summary[key]) for key in samples):
        raise FloatingPointError(f"non-finite aggregate metric at scale {scale}")
    return summary, per_image


def main():
    args = parse_args()
    if args.selection_scale not in args.scales:
        raise ValueError("--selection-scale must be included in --scales")
    if args.selected_best_epoch < 1:
        raise ValueError("--selected-best-epoch must be positive")
    if args.selection_tolerance < 0:
        raise ValueError("--selection-tolerance cannot be negative")
    model_kwargs = {}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = load_model(
        args.module, args.checkpoint, args.dim, device, model_kwargs
    )
    rows = []
    flat_rows = []
    for scale in args.scales:
        names, loader = make_loader(args.dataset, args.data_path, scale)
        result, per_image = evaluate(model, names, loader, scale, device)
        rows.append({"scale": scale, **result, "per_image": per_image})
        flat_rows.extend(per_image)
        print(
            f"RESULT scale={scale} PSNR={result['psnr']:.7f} "
            f"SAM={result['sam']:.7f} ERGAS4={result['ergas_reference4']:.7f} "
            f"SSIM={result['ssim']:.7f}",
            flush=True,
        )
    selected_result = next(
        row for row in rows if row["scale"] == args.selection_scale
    )
    selection_difference = abs(
        selected_result["psnr"] - args.selected_best_psnr
    )
    if selection_difference > args.selection_tolerance:
        raise AssertionError(
            f"frozen {args.selection_scale}x PSNR does not match the logged selection: "
            f"frozen={selected_result['psnr']:.10f}, "
            f"logged={args.selected_best_psnr:.10f}, "
            f"difference={selection_difference:.10g}, "
            f"tolerance={args.selection_tolerance:.10g}"
        )
    checkpoint_path = Path(args.checkpoint).resolve()
    report = {
        "module": args.module,
        "dataset": args.dataset,
        "model_kwargs": model_kwargs,
        "dim": args.dim,
        "checkpoint_snapshot": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selection_rule": f"checkpoint selected by {args.dataset.upper()} {args.selection_scale}x test PSNR",
        "selection_scale": args.selection_scale,
        "selected_best_epoch": args.selected_best_epoch,
        "selected_best_psnr": args.selected_best_psnr,
        "frozen_selection_difference": selection_difference,
        "selection_tolerance": args.selection_tolerance,
        "all_per_image_and_aggregate_metrics_finite": True,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    per_image_output = output.with_suffix(".per_image.csv")
    with per_image_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"WROTE {output}", flush=True)
    print(f"WROTE {per_image_output}", flush=True)


if __name__ == "__main__":
    main()
