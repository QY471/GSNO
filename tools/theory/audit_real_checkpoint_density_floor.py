#!/usr/bin/env python3
"""Audit Gaussian density-floor assumptions on frozen CAVE inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_dynamic_model_multiscale import image_name, make_loader
from tools.neural_operator_eval_utils import (
    load_model,
    resolve_device,
    runtime_provenance,
    sha256,
)


PROTOCOL_VERSION = "frozen-cave-density-floor-audit-v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "7e821ef1095e2f61bbbea66afcb356fe4095d428ab8fcf376157380750e093b0"
)
EXPECTED_EXTENSION_SHA256 = (
    "1f713bd1aa4b032838ccf25ac1fdca737eb11f6415ec7d3b81a40a90b777e25f"
)
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
THRESHOLDS = (1e-6, 1e-5, 1e-4, 1e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--dim", type=int, default=80)
    parser.add_argument("--model-kwargs-json", default='{"max_axis_ratio": 2.0}')
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def tensor_stats(value: torch.Tensor) -> dict[str, float | int]:
    flat = value.detach().double().flatten()
    q = torch.quantile(flat, flat.new_tensor(QUANTILES))
    stats: dict[str, float | int] = {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
    }
    labels = ("min", "p0_1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max")
    stats.update({label: float(number) for label, number in zip(labels, q)})
    for threshold in THRESHOLDS:
        key = f"fraction_lt_{threshold:.0e}".replace("-", "m")
        stats[key] = float((flat < threshold).double().mean())
    return stats


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "sqy":
        raise RuntimeError("formal audit requires `conda activate sqy`")
    if sorted(set(args.scales)) != sorted(args.scales):
        raise ValueError("scales must be unique and sorted")
    if args.max_images < 0:
        raise ValueError("max-images must be nonnegative")

    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"unexpected checkpoint SHA256: {checkpoint_hash}")
    device = resolve_device(args.device)
    model, model_kwargs = load_model(args, device)
    import diff_srgaussian_rasterization._C as extension

    extension_path = Path(extension.__file__).resolve()
    extension_hash = sha256(extension_path)
    if extension_hash != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError(f"unexpected adaptive3 binary SHA256: {extension_hash}")

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for scale in args.scales:
        names, loader = make_loader(args.data_path, scale)
        selected_count = len(loader) if args.max_images == 0 else min(args.max_images, len(loader))
        density_chunks: list[np.ndarray] = []
        for index, (lr_hsi, hr_msi, _target) in enumerate(loader, start=1):
            if index > selected_count:
                break
            prediction, aux = model(
                lr_hsi.to(device).float(),
                hr_msi.to(device).float(),
                scale,
                return_aux=True,
            )
            density = aux["density"]
            if not torch.isfinite(prediction).all() or not torch.isfinite(density).all():
                raise FloatingPointError(f"non-finite output at {scale}x image {index}")
            stats = tensor_stats(density)
            row = {
                "scale": scale,
                "image_index": index,
                "image_name": image_name(names[index - 1], index),
                **stats,
            }
            rows.append(row)
            density_chunks.append(density.cpu().numpy().astype(np.float32, copy=False).reshape(-1))
            print(
                f"DENSITY_AUDIT scale={scale} image={index}/{selected_count} "
                f"min={stats['min']:.9g} p0.1={stats['p0_1']:.9g} mean={stats['mean']:.9g}",
                flush=True,
            )

        scale_density = torch.from_numpy(np.concatenate(density_chunks))
        scale_stats = tensor_stats(scale_density)
        summary[str(scale)] = {"images": selected_count, **scale_stats}

    write_csv(output_dir / "per_scene.csv", rows)
    runtime = runtime_provenance(model, device)
    runtime.update(
        {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": sys.executable,
            "extension_path": str(extension_path),
            "extension_sha256": extension_hash,
        }
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "state": "frozen checkpoint; no training, tuning, or scale-specific selection",
        "module": args.module,
        "model_kwargs": model_kwargs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "data_path": str(Path(args.data_path).resolve()),
        "scales": args.scales,
        "images_per_scale": summary[str(args.scales[0])]["images"],
        "density_source": "model return_aux density before clamp_min(1e-6)",
        "implementation_floor": 1e-6,
        "thresholds": list(THRESHOLDS),
        "quantiles": list(QUANTILES),
        "summary": summary,
        "runtime": runtime,
        "artifacts": {"per_scene_csv": str((output_dir / "per_scene.csv").resolve())},
    }
    atomic_json(result_path, report)
    print(json.dumps({"result": str(result_path), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
