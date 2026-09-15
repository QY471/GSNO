#!/usr/bin/env python3
"""Evaluate one frozen GSFusion checkpoint on Chikusei at several scales."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.Chikusei_AFNO_Dataset import ChikuseiAFNODataset
from tools.Utils import cal_psnr, compute_ergas, compute_sam, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--num-bands", type=int, default=128)
    parser.add_argument("--num-msi", type=int, default=3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--adci-layers", type=int, default=3)
    parser.add_argument("--max-axis-ratio", type=float, default=2.0)
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--selected-4x-best-epoch", type=int, required=True)
    parser.add_argument("--selected-4x-best-psnr", type=float, required=True)
    return parser.parse_args()


def unwrap_state(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    module_name,
    checkpoint,
    num_bands,
    num_msi,
    dim,
    adci_layers,
    max_axis_ratio,
    device,
):
    module = importlib.import_module(module_name)
    model = module.GSFusion(
        dim=dim,
        num_bands=num_bands,
        num_msi=num_msi,
        adci_layers=adci_layers,
        max_axis_ratio=max_axis_ratio,
    )
    model.load_state_dict(
        unwrap_state(torch.load(checkpoint, map_location="cpu")), strict=True
    )
    return model.to(device).eval()


def make_loader(test_file, scale):
    dataset = ChikuseiAFNODataset(test_file, scale, False)
    return tud.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


@torch.inference_mode()
def evaluate(model, loader, scale, device):
    rows = []
    for index, batch in enumerate(loader, start=1):
        lr_hsi, hr_msi, hr_hsi = batch[:3]
        prediction = model(
            lr_hsi.to(device).float(), hr_msi.to(device).float(), scale
        ).clamp(0.0, 1.0)
        pred = prediction[0].cpu().numpy().transpose(1, 2, 0)
        target = hr_hsi[0].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        lr_height, lr_width = lr_hsi.shape[-2:]
        hr_height, hr_width = hr_hsi.shape[-2:]
        row = {
            "scale": scale,
            "image": index,
            "lr_height": int(lr_height),
            "lr_width": int(lr_width),
            "hr_height": int(hr_height),
            "hr_width": int(hr_width),
            "effective_scale_y": float(hr_height / lr_height),
            "effective_scale_x": float(hr_width / lr_width),
            "psnr": float(cal_psnr(pred, target)),
            "sam": float(compute_sam(pred, target)),
            "ergas_reference4": float(compute_ergas(pred, target, 4)),
            "ergas_actual_scale": float(compute_ergas(pred, target, scale)),
            "ssim": float(compute_ssim(pred, target)),
        }
        rows.append(row)
        print(
            f"scale={scale} image={index}/{len(loader)} PSNR={row['psnr']:.7f}",
            flush=True,
        )
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "psnr",
            "sam",
            "ergas_reference4",
            "ergas_actual_scale",
            "ssim",
        )
    }
    summary.update(
        {
            "scale": scale,
            "images": len(rows),
            "lr_size": [rows[0]["lr_height"], rows[0]["lr_width"]],
            "hr_size": [rows[0]["hr_height"], rows[0]["hr_width"]],
            "effective_scale_y": rows[0]["effective_scale_y"],
            "effective_scale_x": rows[0]["effective_scale_x"],
        }
    )
    return summary, rows


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    checkpoint = Path(args.checkpoint).resolve()
    test_file = Path(args.test_file).resolve()
    model = load_model(
        args.module,
        checkpoint,
        args.num_bands,
        args.num_msi,
        args.dim,
        args.adci_layers,
        args.max_axis_ratio,
        device,
    )
    summaries, per_image = [], []
    for scale in args.scales:
        summary, rows = evaluate(
            model, make_loader(str(test_file), scale), scale, device
        )
        summaries.append(summary)
        per_image.extend(rows)
        print(
            f"RESULT scale={scale} PSNR={summary['psnr']:.7f} "
            f"SAM={summary['sam']:.7f} "
            f"ERGAS4={summary['ergas_reference4']:.7f} "
            f"SSIM={summary['ssim']:.7f}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": args.label,
        "module": args.module,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "selection_rule": args.selection_rule,
        "selected_4x_best_epoch": args.selected_4x_best_epoch,
        "selected_4x_best_psnr": args.selected_4x_best_psnr,
        "dataset": "Chikusei supplied AFNO-compatible GT/RGB protocol",
        "test_file": str(test_file),
        "num_bands": args.num_bands,
        "num_msi": args.num_msi,
        "dim": args.dim,
        "adci_layers": args.adci_layers,
        "max_axis_ratio": args.max_axis_ratio,
        "scales": args.scales,
        "summary": summaries,
        "per_image": per_image,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".per_image.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    print(f"WROTE {output}", flush=True)
    print(f"WROTE {csv_path}", flush=True)


if __name__ == "__main__":
    main()
