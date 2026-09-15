#!/usr/bin/env python3
"""Frozen CAVE evaluation with variable input and output sampling grids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.Utils import cal_psnr, compute_ergas, compute_ssim, loadpath, prepare_data
from tools.cave_operator_grid_protocol import (
    build_operator_grid_sample,
    parse_grid_size,
    parse_grid_sizes,
)
from tools.evaluate_dynamic_model_multiscale import stable_compute_sam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--model-kwargs-json", default="{}")
    parser.add_argument(
        "--input-grid-sizes",
        nargs="+",
        default=["256x256", "384x384", "512x512"],
    )
    parser.add_argument("--continuous-reference-size", default="256x256")
    parser.add_argument(
        "--query-grid-sizes",
        nargs="*",
        default=["256x256", "384x384", "512x512"],
    )
    parser.add_argument(
        "--skip-output-query",
        action="store_true",
        help="Evaluate only variable LR/HR input grids for a non-continuous model.",
    )
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--selected-4x-best-psnr", type=float)
    parser.add_argument("--selected-native-size", default="512x512")
    parser.add_argument("--selection-tolerance", type=float, default=1e-4)
    parser.add_argument("--native-equivalence-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(args, device: torch.device):
    try:
        model_kwargs = json.loads(args.model_kwargs_json)
    except json.JSONDecodeError as error:
        raise ValueError("--model-kwargs-json must be valid JSON") from error
    if not isinstance(model_kwargs, dict):
        raise ValueError("--model-kwargs-json must decode to an object")
    module = importlib.import_module(args.module)
    model = module.GSFusion(
        dim=args.dim,
        num_bands=31,
        num_msi=3,
        adci_layers=3,
        **model_kwargs,
    )
    payload = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    state = {key.removeprefix("module."): value for key, value in payload.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), model_kwargs


def metric_row(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = prediction[0].detach().cpu().numpy().transpose(1, 2, 0)
    truth = target[0].detach().cpu().numpy().transpose(1, 2, 0)
    return {
        "psnr": float(cal_psnr(pred, truth)),
        "sam": stable_compute_sam(pred, truth),
        "ergas4": float(compute_ergas(pred, truth, 4)),
        "ssim": float(compute_ssim(pred, truth)),
    }


def finite_tensor_psnr(left: torch.Tensor, right: torch.Tensor) -> float:
    mse = float(F.mse_loss(left.float(), right.float()))
    if mse <= 1e-10:
        return 100.0
    return float(-10.0 * np.log10(mse))


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> dict[str, float]:
    summary = {
        key: float(np.mean([float(row[key]) for row in rows], dtype=np.float64))
        for key in keys
    }
    if not all(np.isfinite(value) for value in summary.values()):
        raise FloatingPointError(f"non-finite aggregate metrics: {summary}")
    return summary


def scene_tensor(array: np.ndarray, index: int) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(array[..., index].transpose(2, 0, 1))
    ).float()


def model_capabilities(model) -> dict[str, bool]:
    parameters = inspect.signature(model.forward).parameters
    return {
        "output_size": "output_size" in parameters,
        "return_aux": "return_aux" in parameters,
    }


def forward_query(model, lr_hsi, hr_msi, factor, query_size, capabilities):
    kwargs = {"output_size": query_size}
    if capabilities["return_aux"]:
        kwargs["return_aux"] = True
    result = model(lr_hsi, hr_msi, factor, **kwargs)
    if capabilities["return_aux"]:
        prediction, aux = result
    else:
        prediction, aux = result, {}
    return prediction, aux


@torch.inference_mode()
def main():
    args = parse_args()
    input_sizes = parse_grid_sizes(args.input_grid_sizes)
    query_sizes = (
        [] if args.skip_output_query else parse_grid_sizes(args.query_grid_sizes)
    )
    reference_size = parse_grid_size(args.continuous_reference_size)
    selected_native_size = parse_grid_size(args.selected_native_size)
    if args.factor <= 0 or args.blur_sigma <= 0:
        raise ValueError("factor and blur sigma must be positive")
    if args.max_images < 0:
        raise ValueError("--max-images cannot be negative")
    for size in set(input_sizes + query_sizes + [reference_size]):
        if size[0] % args.factor or size[1] % args.factor:
            raise ValueError(f"grid {size} must be divisible by factor {args.factor}")

    device = resolve_device(args.device)
    model, model_kwargs = load_model(args, device)
    capabilities = model_capabilities(model)
    if query_sizes and not capabilities["output_size"]:
        raise TypeError(
            f"{args.module}.GSFusion.forward does not expose output_size; "
            "use a continuous model or pass an empty --query-grid-sizes list"
        )

    names = loadpath(os.path.join(args.data_path, "Test.txt"), shuffle=False)
    hr_hsi, hr_msi = prepare_data(args.data_path, names, len(names))
    image_count = len(names) if args.max_images == 0 else min(args.max_images, len(names))
    input_rows = []
    input_summaries = []
    prediction_cache: dict[tuple[int, tuple[int, int]], torch.Tensor] = {}

    for target_size in input_sizes:
        size_rows = []
        for image_index in range(image_count):
            sample = build_operator_grid_sample(
                scene_tensor(hr_hsi, image_index),
                scene_tensor(hr_msi, image_index),
                target_size,
                factor=args.factor,
                sigma=args.blur_sigma,
            )
            lr = sample.lr_hsi.unsqueeze(0).to(device)
            msi = sample.hr_msi.unsqueeze(0).to(device)
            target = sample.target_hsi.unsqueeze(0).to(device)
            prediction = model(lr, msi, args.factor).clamp(0.0, 1.0)
            if tuple(prediction.shape[-2:]) != target_size:
                raise RuntimeError(
                    f"native output {tuple(prediction.shape[-2:])} != {target_size}"
                )
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError(
                    f"non-finite native output at image {image_index + 1}, grid {target_size}"
                )
            row = {
                "experiment": "input_grid_transfer",
                "image_index": image_index + 1,
                "image_name": Path(str(names[image_index])).stem,
                "lr_height": int(lr.shape[-2]),
                "lr_width": int(lr.shape[-1]),
                "hr_height": target_size[0],
                "hr_width": target_size[1],
                **metric_row(prediction, target),
            }
            input_rows.append(row)
            size_rows.append(row)
            prediction_cache[(image_index, target_size)] = prediction.cpu()
            print(
                f"INPUT_GRID image={image_index + 1}/{image_count} "
                f"grid={target_size[0]}x{target_size[1]} psnr={row['psnr']:.7f}",
                flush=True,
            )
        input_summaries.append(
            {
                "hr_height": target_size[0],
                "hr_width": target_size[1],
                "lr_height": target_size[0] // args.factor,
                "lr_width": target_size[1] // args.factor,
                "images": len(size_rows),
                **aggregate(size_rows, ("psnr", "sam", "ergas4", "ssim")),
            }
        )

    common_size = max(input_sizes, key=lambda size: size[0] * size[1])
    consistency_rows = []
    for image_index in range(image_count):
        common_prediction = prediction_cache[(image_index, common_size)]
        for source_size in input_sizes:
            source_prediction = prediction_cache[(image_index, source_size)]
            resized = F.interpolate(
                source_prediction,
                size=common_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0)
            consistency_rows.append(
                {
                    "experiment": "input_grid_consistency",
                    "image_index": image_index + 1,
                    "image_name": Path(str(names[image_index])).stem,
                    "source_height": source_size[0],
                    "source_width": source_size[1],
                    "common_height": common_size[0],
                    "common_width": common_size[1],
                    "psnr_vs_common_grid_prediction": finite_tensor_psnr(
                        resized, common_prediction
                    ),
                }
            )

    query_rows = []
    query_summaries = []
    native_explicit_errors = []
    if query_sizes:
        per_query: dict[tuple[int, int], list[dict]] = {size: [] for size in query_sizes}
        for image_index in range(image_count):
            full_hsi = scene_tensor(hr_hsi, image_index)
            full_msi = scene_tensor(hr_msi, image_index)
            reference = build_operator_grid_sample(
                full_hsi,
                full_msi,
                reference_size,
                factor=args.factor,
                sigma=args.blur_sigma,
            )
            lr = reference.lr_hsi.unsqueeze(0).to(device)
            msi = reference.hr_msi.unsqueeze(0).to(device)
            native = model(lr, msi, args.factor).clamp(0.0, 1.0)
            explicit_native, _ = forward_query(
                model, lr, msi, args.factor, reference_size, capabilities
            )
            explicit_native = explicit_native.clamp(0.0, 1.0)
            native_error = float((native - explicit_native).abs().max())
            native_explicit_errors.append(native_error)
            if native_error > args.native_equivalence_tolerance:
                raise AssertionError(
                    f"native explicit query error {native_error} exceeds "
                    f"{args.native_equivalence_tolerance}"
                )

            for query_size in query_sizes:
                prediction, aux = forward_query(
                    model, lr, msi, args.factor, query_size, capabilities
                )
                prediction = prediction.clamp(0.0, 1.0)
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError(
                        f"non-finite query output at image {image_index + 1}, {query_size}"
                    )
                query_target = build_operator_grid_sample(
                    full_hsi,
                    full_msi,
                    query_size,
                    factor=args.factor,
                    sigma=args.blur_sigma,
                ).target_hsi.unsqueeze(0).to(device)
                native_resampled = (
                    native
                    if query_size == reference_size
                    else F.interpolate(
                        native,
                        size=query_size,
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ).clamp(0.0, 1.0)
                )
                resampled_native_metrics = metric_row(native_resampled, query_target)
                cycle = F.interpolate(
                    prediction,
                    size=reference_size,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ).clamp(0.0, 1.0)
                density = aux.get("density")
                row = {
                    "experiment": "output_grid_query",
                    "image_index": image_index + 1,
                    "image_name": Path(str(names[image_index])).stem,
                    "reference_height": reference_size[0],
                    "reference_width": reference_size[1],
                    "query_height": query_size[0],
                    "query_width": query_size[1],
                    **metric_row(prediction, query_target),
                    "resampled_native_psnr": resampled_native_metrics["psnr"],
                    "resampled_native_sam": resampled_native_metrics["sam"],
                    "resampled_native_ergas4": resampled_native_metrics["ergas4"],
                    "resampled_native_ssim": resampled_native_metrics["ssim"],
                    "query_vs_resampled_native_psnr": finite_tensor_psnr(
                        prediction, native_resampled
                    ),
                    "query_vs_resampled_native_max_abs_error": float(
                        (prediction - native_resampled).abs().max()
                    ),
                    "cycle_psnr_vs_native": finite_tensor_psnr(cycle, native),
                    "cycle_max_abs_error_vs_native": float(
                        (cycle - native).abs().max()
                    ),
                    "density_min": float(density.min()) if density is not None else None,
                    "density_mean": float(density.mean()) if density is not None else None,
                }
                if density is not None and not bool(torch.isfinite(density).all()):
                    raise FloatingPointError(
                        f"non-finite density at image {image_index + 1}, {query_size}"
                    )
                query_rows.append(row)
                per_query[query_size].append(row)
                print(
                    f"OUTPUT_GRID image={image_index + 1}/{image_count} "
                    f"reference={reference_size[0]}x{reference_size[1]} "
                    f"query={query_size[0]}x{query_size[1]} psnr={row['psnr']:.7f}",
                    flush=True,
                )
        for query_size in query_sizes:
            rows = per_query[query_size]
            summary = {
                "reference_height": reference_size[0],
                "reference_width": reference_size[1],
                "query_height": query_size[0],
                "query_width": query_size[1],
                "images": len(rows),
                **aggregate(
                    rows,
                    (
                        "psnr",
                        "sam",
                        "ergas4",
                        "ssim",
                        "resampled_native_psnr",
                        "resampled_native_sam",
                        "resampled_native_ergas4",
                        "resampled_native_ssim",
                        "query_vs_resampled_native_psnr",
                        "cycle_psnr_vs_native",
                    ),
                ),
            }
            summary["psnr_minus_resampled_native"] = (
                summary["psnr"] - summary["resampled_native_psnr"]
            )
            query_summaries.append(summary)

    selection_difference = None
    if args.selected_4x_best_psnr is not None:
        selected = next(
            (
                row
                for row in input_summaries
                if (row["hr_height"], row["hr_width"]) == selected_native_size
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"selected native size {selected_native_size} is not in input grids"
            )
        if image_count == len(names):
            selection_difference = abs(
                selected["psnr"] - args.selected_4x_best_psnr
            )
            if selection_difference > args.selection_tolerance:
                raise AssertionError(
                    f"frozen native PSNR {selected['psnr']} does not match logged "
                    f"{args.selected_4x_best_psnr}"
                )

    checkpoint = Path(args.checkpoint).resolve()
    report = {
        "protocol": "CAVE bidirectional sampling-grid neural-operator evaluation",
        "module": args.module,
        "model_kwargs": model_kwargs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "device": str(device),
        "factor": args.factor,
        "blur_sigma": args.blur_sigma,
        "source_grid": [int(hr_hsi.shape[0]), int(hr_hsi.shape[1])],
        "evaluated_images": image_count,
        "full_test_images": len(names),
        "input_grid_sizes": [list(size) for size in input_sizes],
        "continuous_reference_size": list(reference_size),
        "query_grid_sizes": [list(size) for size in query_sizes],
        "model_capabilities": capabilities,
        "input_grid_transfer": input_summaries,
        "input_grid_consistency": consistency_rows,
        "output_grid_query": query_summaries,
        "native_default_vs_explicit_max_abs_error": (
            max(native_explicit_errors) if native_explicit_errors else None
        ),
        "selection_check": {
            "selected_native_size": list(selected_native_size),
            "logged_4x_best_psnr": args.selected_4x_best_psnr,
            "difference": selection_difference,
            "tolerance": args.selection_tolerance,
            "checked": selection_difference is not None,
        },
        "interpretation": {
            "input_grid_transfer": (
                "The same 512x512 scene extent is antialiased onto each HR grid; "
                "LR-HSI is regenerated with the existing Gaussian factor-4 protocol."
            ),
            "output_grid_query": (
                "The LR-HSI and HR-MSI inputs remain fixed at the reference grid; "
                "only the continuous output query grid changes. Each query is also "
                "compared with bicubic resampling of the native model prediction."
            ),
            "boundary": (
                "Targets at or below the original 512x512 source are resampled-reference "
                "diagnostics. Queries above the source grid must not be described as true GT."
            ),
        },
        "per_image": {
            "input_grid_transfer": input_rows,
            "output_grid_query": query_rows,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    flat_rows = input_rows + consistency_rows + query_rows
    csv_output = output.with_suffix(".per_image.csv")
    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"WROTE {output}", flush=True)
    print(f"WROTE {csv_output}", flush=True)


if __name__ == "__main__":
    main()
