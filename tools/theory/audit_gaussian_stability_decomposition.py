#!/usr/bin/env python3
"""Feature-space audit of value/kernel perturbation stability across scales."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_gaussian_geometry_causal_swap import (
    encode_reference,
    gaussian_fields,
    load_scale_records,
    relative_l2,
    render_fields,
)
from tools.neural_operator_eval_utils import (
    load_model,
    resolve_device,
    runtime_provenance,
    sha256,
)


PROTOCOL_VERSION = "paired-scale-sampled-normalized-kernel-decomposition-v1"
EXPECTED_CHECKPOINT_SHA256 = (
    "7e821ef1095e2f61bbbea66afcb356fe4095d428ab8fcf376157380750e093b0"
)
EXPECTED_EXTENSION_SHA256 = (
    "1f713bd1aa4b032838ccf25ac1fdca737eb11f6415ec7d3b81a40a90b777e25f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--sample-grid-side", type=int, default=16)
    parser.add_argument("--dim", type=int, default=80)
    parser.add_argument("--model-kwargs-json", default='{"max_axis_ratio": 2.0}')
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--mirror-tolerance", type=float, default=5e-5)
    parser.add_argument("--bound-tolerance", type=float, default=5e-6)
    return parser.parse_args()


def sample_query_indices(height: int, width: int, side: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if side < 2 or side > min(height, width):
        raise ValueError("sample-grid-side must be in [2,min(H,W)]")
    rows = torch.linspace(0, height - 1, side, device=device).round().long().unique()
    cols = torch.linspace(0, width - 1, side, device=device).round().long().unique()
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1)


def sampled_kernel(
    fields: dict[str, torch.Tensor],
    query_rows: torch.Tensor,
    query_cols: torch.Tensor,
    height: int,
    width: int,
    sigma_radius: float,
    std_max_px: float,
) -> dict[str, torch.Tensor]:
    """Evaluate exact formal point weights on a conservative local candidate set."""
    if fields["value"].shape[0] != 1:
        raise ValueError("sampled kernel audit currently requires batch size one")
    candidate_radius = math.ceil(sigma_radius * std_max_px)
    offsets = torch.arange(-candidate_radius, candidate_radius + 1, device=query_rows.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    source_rows = query_rows[:, None] + offset_y.reshape(1, -1)
    source_cols = query_cols[:, None] + offset_x.reshape(1, -1)
    valid_grid = (
        (source_rows >= 0)
        & (source_rows < height)
        & (source_cols >= 0)
        & (source_cols < width)
    )
    safe_rows = source_rows.clamp(0, height - 1)
    safe_cols = source_cols.clamp(0, width - 1)
    source_index = safe_rows * width + safe_cols

    std = fields["std"][0][source_index]
    rho = fields["rho"][0, source_index, 0]
    opacity = fields["opacity"][0, source_index, 0]
    value = fields["value"][0][source_index]
    delta_x = query_cols[:, None].float() - safe_cols.float()
    delta_y = query_rows[:, None].float() - safe_rows.float()
    support = (
        valid_grid
        & (delta_x.abs() < sigma_radius * std[..., 0])
        & (delta_y.abs() < sigma_radius * std[..., 1])
    )
    beta = 1.0 - rho.square()
    exponent = -0.5 / beta * (
        delta_x.square() / std[..., 0].square()
        + delta_y.square() / std[..., 1].square()
        - 2.0 * rho * delta_x * delta_y / (std[..., 0] * std[..., 1])
    )
    gaussian = torch.exp(exponent) / (
        2.0 * math.pi * std[..., 0] * std[..., 1] * torch.sqrt(beta)
    )
    raw_weight = torch.where(support, opacity * gaussian, torch.zeros_like(gaussian))
    mass = raw_weight.sum(dim=1, keepdim=True)
    normalized_weight = raw_weight / mass.clamp_min(1e-6)
    output = torch.sum(normalized_weight[..., None] * value, dim=1)
    return {
        "raw_weight": raw_weight,
        "mass": mass[:, 0],
        "normalized_weight": normalized_weight,
        "value": value,
        "output": output,
        "candidate_count": support.sum(dim=1),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def aggregate(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for scale in sorted({row["scale"] for row in rows}):
        selected = [row for row in rows if row["scale"] == scale]
        keys = [
            key for key, value in selected[0].items()
            if key not in {"image_index", "image_name", "scale"} and isinstance(value, (int, float))
        ]
        summary[str(scale)] = {
            "scenes": len(selected),
            **{f"{key}_mean": float(np.mean([row[key] for row in selected])) for key in keys},
            **{f"{key}_max": float(np.max([row[key] for row in selected])) for key in keys},
        }
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "sqy":
        raise RuntimeError("formal audit requires `conda activate sqy`")
    if args.scales[0] != 4 or sorted(set(args.scales)) != args.scales:
        raise ValueError("scales must be sorted, unique, and start with 4")
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

    records, names = load_scale_records(args.data_path, args.scales)
    if args.max_images:
        names = names[: args.max_images]
    scene_rows: list[dict] = []
    sample_rows: list[dict] = []
    mirror_max_abs = 0.0
    bound_violation_max = 0.0
    for image_index, name in enumerate(names, start=1):
        scale_fields: dict[int, dict[str, torch.Tensor]] = {}
        scale_deltas: dict[int, torch.Tensor] = {}
        height = width = 0
        for scale in args.scales:
            lr_hsi, hr_msi, _target = records[scale][name]
            lr_hsi = lr_hsi.to(device).float()
            hr_msi = hr_msi.to(device).float()
            _base, _fused, primitive = encode_reference(model, lr_hsi, hr_msi)
            fields = gaussian_fields(model.gaussian_refine, primitive)
            height, width = map(int, hr_msi.shape[-2:])
            delta, _density = render_fields(model.gaussian_refine, fields, height, width)
            scale_fields[scale] = fields
            scale_deltas[scale] = delta

        query_rows, query_cols = sample_query_indices(
            height, width, args.sample_grid_side, device
        )
        sampled = {
            scale: sampled_kernel(
                fields,
                query_rows,
                query_cols,
                height,
                width,
                model.gaussian_refine.sigma_radius,
                model.gaussian_refine.std_max_px,
            )
            for scale, fields in scale_fields.items()
        }
        for scale in args.scales:
            cuda_samples = scale_deltas[scale][0, :, query_rows, query_cols].transpose(0, 1)
            mirror_error = float((cuda_samples - sampled[scale]["output"]).abs().max())
            mirror_max_abs = max(mirror_max_abs, mirror_error)
            if mirror_error > args.mirror_tolerance:
                raise AssertionError(
                    f"sampled kernel mirror error {mirror_error} exceeds tolerance "
                    f"for {name} {scale}x"
                )

        reference_fields = scale_fields[4]
        reference_sample = sampled[4]
        for scale in args.scales[1:]:
            fields = scale_fields[scale]
            current_sample = sampled[scale]
            kernel_l1 = (
                current_sample["normalized_weight"]
                - reference_sample["normalized_weight"]
            ).abs().sum(dim=1)
            local_value_difference = (
                current_sample["value"] - reference_sample["value"]
            ).abs().amax(dim=(1, 2))
            local_reference_value_bound = reference_sample["value"].abs().amax(dim=(1, 2))
            value_term = local_value_difference
            kernel_term = local_reference_value_bound * kernel_l1
            bound = value_term + kernel_term
            lhs = (current_sample["output"] - reference_sample["output"]).abs().amax(dim=1)
            violation = lhs - bound
            bound_violation_max = max(bound_violation_max, float(violation.max()))
            if float(violation.max()) > args.bound_tolerance:
                raise AssertionError(
                    f"normalized-kernel stability bound violation {float(violation.max())} "
                    f"for {name} {scale}x"
                )

            value_from_4 = torch.sum(
                current_sample["normalized_weight"][..., None]
                * reference_sample["value"],
                dim=1,
            )
            geometry_from_4 = torch.sum(
                reference_sample["normalized_weight"][..., None]
                * current_sample["value"],
                dim=1,
            )
            content_effect = (current_sample["output"] - value_from_4).abs().amax(dim=1)
            geometry_effect = (current_sample["output"] - geometry_from_4).abs().amax(dim=1)

            for sample_index in range(query_rows.numel()):
                sample_rows.append(
                    {
                        "image_index": image_index,
                        "image_name": name,
                        "scale": scale,
                        "query_index": sample_index,
                        "query_row": int(query_rows[sample_index]),
                        "query_col": int(query_cols[sample_index]),
                        "kernel_l1": float(kernel_l1[sample_index]),
                        "value_term_linf": float(value_term[sample_index]),
                        "kernel_term_linf": float(kernel_term[sample_index]),
                        "bound_linf": float(bound[sample_index]),
                        "output_difference_linf": float(lhs[sample_index]),
                        "bound_slack": float(bound[sample_index] - lhs[sample_index]),
                        "content_swap_effect_linf": float(content_effect[sample_index]),
                        "geometry_swap_effect_linf": float(geometry_effect[sample_index]),
                        "mass_4x": float(reference_sample["mass"][sample_index]),
                        "mass_current": float(current_sample["mass"][sample_index]),
                        "candidate_count_4x": int(reference_sample["candidate_count"][sample_index]),
                        "candidate_count_current": int(current_sample["candidate_count"][sample_index]),
                    }
                )

            scene_rows.append(
                {
                    "image_index": image_index,
                    "image_name": name,
                    "scale": scale,
                    "value_relative_l2": relative_l2(fields["value"], reference_fields["value"]),
                    "value_linf": float((fields["value"] - reference_fields["value"]).abs().max()),
                    "opacity_mae": float((fields["opacity"] - reference_fields["opacity"]).abs().mean()),
                    "std_relative_l2": relative_l2(fields["std"], reference_fields["std"]),
                    "rho_mae": float((fields["rho"] - reference_fields["rho"]).abs().mean()),
                    "gaussian_delta_relative_l2": relative_l2(scale_deltas[scale], scale_deltas[4]),
                    "kernel_l1_sample_mean": float(kernel_l1.mean()),
                    "kernel_l1_sample_max": float(kernel_l1.max()),
                    "value_term_sample_mean": float(value_term.mean()),
                    "kernel_term_sample_mean": float(kernel_term.mean()),
                    "output_difference_sample_mean": float(lhs.mean()),
                    "output_difference_sample_max": float(lhs.max()),
                    "content_swap_effect_sample_mean": float(content_effect.mean()),
                    "geometry_swap_effect_sample_mean": float(geometry_effect.mean()),
                    "bound_violation_max": float(violation.max()),
                }
            )
            print(
                f"STABILITY image={image_index}/{len(names)} name={name} scale={scale} "
                f"value_rel_l2={scene_rows[-1]['value_relative_l2']:.7g} "
                f"kernel_l1={scene_rows[-1]['kernel_l1_sample_mean']:.7g}",
                flush=True,
            )

    write_csv(output_dir / "per_scene.csv", scene_rows)
    write_csv(output_dir / "per_sample.csv", sample_rows)
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
        "state": "frozen checkpoint; paired same-scene feature-space diagnostic",
        "module": args.module,
        "model_kwargs": model_kwargs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "scales": args.scales,
        "images": len(names),
        "sample_grid_side": args.sample_grid_side,
        "sample_queries_per_scene": args.sample_grid_side * args.sample_grid_side,
        "kernel_definition": "formal point-evaluated adaptive axis-aligned 3sigma support, normalized by local mass",
        "bound": "||K(p_s,v_s)-K(p_4,v_4)||inf <= ||v_s-v_4||inf + ||v_4||inf ||p_s-p_4||1, evaluated locally per sampled query",
        "interpretation_boundary": "The sampled normalized-weight diagnostic validates the Gaussian-stage decomposition; it is not a whole-network CDE bound.",
        "mirror_max_abs_vs_formal_cuda": mirror_max_abs,
        "bound_violation_max": bound_violation_max,
        "summary": aggregate(scene_rows),
        "runtime": runtime,
        "artifacts": {
            "per_scene_csv": str((output_dir / "per_scene.csv").resolve()),
            "per_sample_csv": str((output_dir / "per_sample.csv").resolve()),
        },
    }
    atomic_json(result_path, report)
    print(json.dumps({"result": str(result_path), "summary": report["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
