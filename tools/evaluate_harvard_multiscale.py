#!/usr/bin/env python3
"""Evaluate one frozen GSFusion checkpoint on Harvard at several scales."""

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

from datasets.Harvard_Dataset import harvard_dataset, prepare_data_harvard
from tools.Utils import cal_psnr, compute_ergas, compute_sam, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-bands", type=int, default=31)
    parser.add_argument("--num-msi", type=int, default=3)
    parser.add_argument("--adci-layers", type=int, default=3)
    parser.add_argument("--max-axis-ratio", type=float, default=None)
    parser.add_argument("--dataset-manifest", required=True)
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args, checkpoint, device):
    module_name = args.module
    module = importlib.import_module(module_name)
    model_kwargs = {
        "dim": args.dim,
        "num_bands": args.num_bands,
        "num_msi": args.num_msi,
        "adci_layers": args.adci_layers,
    }
    if args.max_axis_ratio is not None:
        model_kwargs["max_axis_ratio"] = args.max_axis_ratio
    model = module.GSFusion(**model_kwargs)
    model.load_state_dict(
        unwrap_state(torch.load(checkpoint, map_location="cpu")), strict=True
    )
    return model.to(device).eval(), model_kwargs


def make_loader(hr_hsi, hr_msi, test_path, scale, spatial_crop):
    options = argparse.Namespace(
        data_path=test_path,
        sizeI=None,
        testset_num=10,
        batch_size=1,
        sf=scale,
        seed=1,
        kernel_type="gaussian_blur",
        eval_crop_size=int(spatial_crop["shape"][0]),
        eval_crop_top=int(spatial_crop["row_slice"][0]),
        eval_crop_left=int(spatial_crop["column_slice"][0]),
    )
    dataset = harvard_dataset(options, hr_hsi, hr_msi, istrain=False)
    return tud.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


@torch.inference_mode()
def evaluate(model, loader, scale, device, file_records, expected_hr_shape):
    rows = []
    for index, batch in enumerate(loader, start=1):
        lr_hsi, hr_msi, hr_hsi = batch[:3]
        if tuple(hr_hsi.shape[-2:]) != tuple(expected_hr_shape):
            raise ValueError(
                f"unexpected Harvard HR-HSI shape {tuple(hr_hsi.shape[-2:])}; "
                f"expected {tuple(expected_hr_shape)}"
            )
        if tuple(hr_msi.shape[-2:]) != tuple(expected_hr_shape):
            raise ValueError("Harvard HR-MSI/HSI crop shapes are not paired")
        prediction = model(
            lr_hsi.to(device).float(), hr_msi.to(device).float(), scale
        ).clamp(0.0, 1.0)
        pred = prediction[0].cpu().numpy().transpose(1, 2, 0)
        target = hr_hsi[0].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        row = {
            "scale": scale,
            "image": index,
            "image_name": file_records[index - 1]["name"],
            "image_file_sha256": file_records[index - 1]["sha256"],
            "psnr": float(cal_psnr(pred, target)),
            "sam": float(compute_sam(pred, target)),
            "ergas_reference4": float(compute_ergas(pred, target, 4)),
            "ergas_actual_scale": float(compute_ergas(pred, target, scale)),
            "ssim": float(compute_ssim(pred, target)),
        }
        rows.append(row)
        print(
            f"scale={scale} image={index}/{len(loader)} "
            f"PSNR={row['psnr']:.7f}",
            flush=True,
        )
    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "psnr", "sam", "ergas_reference4", "ergas_actual_scale", "ssim"
        )
    }
    summary.update({"scale": scale, "images": len(rows)})
    return summary, rows


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    checkpoint = str(Path(args.checkpoint).resolve())
    output = Path(args.output).resolve()
    csv_path = output.with_suffix(".per_image.csv")
    if output.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite a Harvard frozen diagnostic")
    test_path = str(Path(args.data_root).resolve() / "Test")
    manifest_path = Path(args.dataset_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_version"] != "harvard-supplied-paired-top-left512-v1":
        raise ValueError("unexpected Harvard dataset manifest protocol")
    file_records = manifest["files"]
    spatial_crop = manifest["spatial_crop"]
    expected_hr_shape = spatial_crop["shape"]
    if expected_hr_shape != [512, 512]:
        raise ValueError("primary Harvard diagnostic must preserve the 512 HR grid")
    if [item["name"] for item in file_records] != [f"{i}.mat" for i in range(1, 11)]:
        raise ValueError("Harvard dataset manifest has unexpected file order")
    for item in file_records:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"Harvard asset changed after manifest: {item['path']}")
    print(f"Loading Harvard test data from {test_path}", flush=True)
    hr_hsi, hr_msi = prepare_data_harvard(test_path, 10)
    model, model_kwargs = load_model(args, checkpoint, device)

    summaries, per_image = [], []
    for scale in args.scales:
        summary, rows = evaluate(
            model,
            make_loader(hr_hsi, hr_msi, test_path, scale, spatial_crop),
            scale,
            device,
            file_records,
            expected_hr_shape,
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

    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_version": "cave4-frozen-to-harvard-paired-grid512-v1",
        "label": args.label,
        "module": args.module,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_kwargs": model_kwargs,
        "selection_rule": args.selection_rule,
        "dataset": "Harvard supplied paired MAT protocol",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_asset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "spatial_crop": manifest["spatial_crop"],
        "pairing": manifest["pairing"],
        "lr_hsi_generation": manifest["lr_hsi_generation"],
        "interpretation_boundary": (
            "Frozen CAVE-checkpoint transfer diagnostic only. No Harvard sample is "
            "used for training, checkpoint selection, or tuning. Stored paired HRMS "
            "is used without applying or inventing an SRF. This result does not prove "
            "unknown-sensor transfer or the original acquisition provenance of HRMS. "
            "The HR reference grid is fixed at 512x512 to avoid confounding dataset "
            "transfer with absolute reference-grid transfer."
        ),
        "scales": args.scales,
        "images_per_scale": len(file_records),
        "runtime": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
        },
        "summary": summaries,
        "per_image": per_image,
    }
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    temporary_csv.replace(csv_path)
    serialized = json.dumps(report, indent=2)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(serialized, encoding="utf-8")
    temporary_output.replace(output)
    print(f"WROTE {output}", flush=True)
    print(f"WROTE {csv_path}", flush=True)


if __name__ == "__main__":
    main()
