#!/usr/bin/env python3
"""Controlled CUDA audit of Gaussian quadrature and density normalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.GSFusion_HRFused_Circular_PrimitiveEmbedding import (
    _resolve_adaptive_gaussian_rasterizer,
)


PROTOCOL_VERSION = "fixed-query-uniform-source-quadrature-v2"
LEGAL_OPACITY_MULTIPLIERS = (0.25, 0.5, 1.0, 1.25)
EXPECTED_EXTENSION_SHA256 = (
    "1f713bd1aa4b032838ccf25ac1fdca737eb11f6415ec7d3b81a40a90b777e25f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-grids",
        nargs="+",
        type=int,
        default=[16, 24, 32, 48, 64, 96, 128, 192],
    )
    parser.add_argument("--reference-grid", type=int, default=256)
    parser.add_argument("--query-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--skip-full", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_coordinates(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-cell centers and their output-pixel coordinates."""
    axis = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def continuous_values(x: torch.Tensor, y: torch.Tensor, channels: int) -> torch.Tensor:
    fields = [
        torch.full_like(x, 0.25),
        0.50 + 0.20 * torch.sin(2.0 * math.pi * x) + 0.10 * torch.cos(2.0 * math.pi * y),
        0.15 + 0.70 * x * y,
        0.50 + 0.20 * torch.sin(4.0 * math.pi * x) * torch.cos(2.0 * math.pi * y),
        0.30 + 0.15 * torch.cos(2.0 * math.pi * (x + y)),
        0.40 + 0.10 * torch.sin(6.0 * math.pi * x) * torch.sin(2.0 * math.pi * y),
        0.20 + 0.60 * x.square() * (1.0 - y),
        0.50 + 0.20 * torch.cos(4.0 * math.pi * y),
    ]
    if channels < 1 or channels > len(fields):
        raise ValueError(f"channels must be in [1,{len(fields)}], got {channels}")
    return torch.stack(fields[:channels], dim=-1)


def continuous_kernel_fields(
    case: str, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if case == "isotropic_constant":
        opacity = torch.full_like(x, 0.70)
        sigma_1 = torch.full_like(x, 0.065)
        sigma_2 = sigma_1
        theta = torch.zeros_like(x)
    elif case == "anisotropic_smooth":
        opacity = torch.full_like(x, 0.70)
        base = 0.060 * (1.0 + 0.15 * torch.sin(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * y))
        stretch = 0.22 * torch.sin(2.0 * math.pi * x) * torch.cos(2.0 * math.pi * y)
        sigma_1 = base * torch.exp(stretch)
        sigma_2 = base * torch.exp(-stretch)
        theta = 0.45 * math.pi * torch.sin(2.0 * math.pi * x) * torch.sin(2.0 * math.pi * y)
    elif case == "opacity_smooth":
        opacity = 0.60 + 0.25 * torch.sin(2.0 * math.pi * x) * torch.cos(2.0 * math.pi * y)
        sigma_1 = torch.full_like(x, 0.070)
        sigma_2 = torch.full_like(x, 0.050)
        theta = 0.30 * math.pi * torch.sin(2.0 * math.pi * (x + y))
    else:
        raise KeyError(case)

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    var_1 = sigma_1.square()
    var_2 = sigma_2.square()
    var_x = var_1 * cos_theta.square() + var_2 * sin_theta.square()
    var_y = var_1 * sin_theta.square() + var_2 * cos_theta.square()
    covariance = (var_1 - var_2) * sin_theta * cos_theta
    std_x = torch.sqrt(var_x)
    std_y = torch.sqrt(var_y)
    rho = covariance / (std_x * std_y).clamp_min(1e-12)
    return opacity, std_x, std_y, rho


def make_fields(
    source_size: int,
    query_size: int,
    channels: int,
    case: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x, y = source_coordinates(source_size, device)
    opacity, std_x, std_y, rho = continuous_kernel_fields(case, x, y)
    means = torch.stack((x * query_size - 0.5, y * query_size - 0.5), dim=-1)
    stds = torch.stack((std_x * query_size, std_y * query_size), dim=-1)
    values = continuous_values(x, y, channels)
    return {
        "opacity": opacity.view(1, -1, 1),
        "means": means.unsqueeze(0),
        "stds": stds.unsqueeze(0),
        "rho": rho.view(1, -1, 1),
        "values": values.unsqueeze(0),
    }


def render(
    rasterizer,
    fields: dict[str, torch.Tensor],
    query_size: int,
    adaptive: bool,
    opacity_multiplier: float = 1.0,
    values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    current_values = fields["values"] if values is None else values
    packed_values = torch.cat(
        (current_values, torch.ones_like(current_values[..., :1])), dim=-1
    )
    output = rasterizer(
        fields["opacity"].float() * opacity_multiplier,
        fields["means"].float(),
        fields["stds"].float(),
        fields["rho"].float(),
        packed_values.float(),
        query_size,
        query_size,
        1,
        1.0,
        debug=False,
        adaptive_window=adaptive,
        sigma_radius=3.0,
    ).permute(0, 3, 1, 2).contiguous()
    channels = current_values.shape[-1]
    numerator = output[:, :channels]
    density = output[:, channels : channels + 1]
    return {
        "normalized": numerator / density.clamp_min(1e-6),
        "raw": numerator,
        "density": density,
    }


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm((value - reference).double())
        / torch.linalg.vector_norm(reference.double()).clamp_min(1e-15)
    )


def metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = (value - reference).double()
    return {
        "relative_l2": relative_l2(value, reference),
        "mae": float(difference.abs().mean()),
        "linf": float(difference.abs().max()),
        "mean_abs": float(value.double().abs().mean()),
    }


def density_stats(density: torch.Tensor) -> dict[str, float]:
    flat = density.double().flatten()
    quantiles = torch.quantile(
        flat, flat.new_tensor([0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    )
    return {
        "min": float(quantiles[0]),
        "p0_1": float(quantiles[1]),
        "p1": float(quantiles[2]),
        "p5": float(quantiles[3]),
        "p50": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(quantiles[7]),
        "mean": float(flat.mean()),
        "fraction_lt_1e_6": float((flat < 1e-6).double().mean()),
    }


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_plot(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted({row["case"] for row in rows})
    figure, axes = plt.subplots(len(cases), 2, figsize=(10, 3.2 * len(cases)), squeeze=False)
    for row_index, case in enumerate(cases):
        selected = [row for row in rows if row["case"] == case and row["support"] == "adaptive_3sigma"]
        grids = np.asarray([row["source_grid"] for row in selected])
        axes[row_index, 0].loglog(grids, [row["normalized_relative_l2"] for row in selected], "o-", label="Normalized")
        axes[row_index, 0].loglog(grids, [row["area_raw_relative_l2"] for row in selected], "s-", label="Area-corrected raw")
        axes[row_index, 0].loglog(grids, [row["raw_to_integral_relative_l2"] for row in selected], "^-", label="RawScatter")
        axes[row_index, 0].set_title(f"{case}: error")
        axes[row_index, 0].set_xlabel("source grid side")
        axes[row_index, 0].set_ylabel("relative L2")
        axes[row_index, 0].grid(True, which="both", alpha=0.3)
        axes[row_index, 0].legend()
        axes[row_index, 1].loglog(grids, [row["normalized_mean_abs"] for row in selected], "o-", label="Normalized")
        axes[row_index, 1].loglog(grids, [row["raw_mean_abs"] for row in selected], "^-", label="RawScatter")
        axes[row_index, 1].loglog(grids, [row["area_raw_mean_abs"] for row in selected], "s-", label="Area-corrected raw")
        axes[row_index, 1].set_title(f"{case}: amplitude")
        axes[row_index, 1].set_xlabel("source grid side")
        axes[row_index, 1].set_ylabel("mean absolute output")
        axes[row_index, 1].grid(True, which="both", alpha=0.3)
        axes[row_index, 1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def convergence_summary(rows: list[dict]) -> list[dict]:
    summary = []
    methods = {
        "normalized": "normalized_relative_l2",
        "area_corrected_raw": "area_raw_relative_l2",
        "raw_scatter_vs_integral": "raw_to_integral_relative_l2",
    }
    for case in sorted({row["case"] for row in rows}):
        for support in sorted({row["support"] for row in rows}):
            selected = [
                row for row in rows
                if row["case"] == case and row["support"] == support
            ]
            selected.sort(key=lambda row: row["source_grid"])
            h = 1.0 / np.asarray([row["source_grid"] for row in selected], dtype=np.float64)
            for method, error_key in methods.items():
                errors = np.asarray([row[error_key] for row in selected], dtype=np.float64)
                positive = np.isfinite(errors) & (errors > 0)
                slope = float(np.polyfit(np.log(h[positive]), np.log(errors[positive]), 1)[0]) if positive.sum() >= 2 else None
                summary.append(
                    {
                        "case": case,
                        "support": support,
                        "method": method,
                        "coarsest_grid": selected[0]["source_grid"],
                        "finest_test_grid": selected[-1]["source_grid"],
                        "coarsest_relative_l2": float(errors[0]),
                        "finest_relative_l2": float(errors[-1]),
                        "empirical_log_error_vs_log_h_slope": slope,
                    }
                )
    return summary


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "sqy":
        raise RuntimeError("formal experiment requires `conda activate sqy`")
    if not torch.cuda.is_available():
        raise RuntimeError("formal CUDA audit requires an available CUDA device")
    if args.query_size < 8 or args.reference_grid < max(args.source_grids):
        raise ValueError("query_size must be >=8 and reference_grid >= every source grid")
    if len(set(args.source_grids)) != len(args.source_grids):
        raise ValueError("source grids must be unique")

    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed result: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()
    rasterizer = GaussianRasterizer(args.channels + 1).to(device)
    import diff_srgaussian_rasterization._C as extension

    extension_path = Path(extension.__file__).resolve()
    extension_sha = sha256(extension_path)
    if extension_sha != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError(
            f"unexpected adaptive3 binary {extension_sha}; expected {EXPECTED_EXTENSION_SHA256}"
        )

    source_grids = sorted(args.source_grids)
    cases = ["isotropic_constant", "anisotropic_smooth", "opacity_smooth"]
    supports = ["adaptive_3sigma"] if args.skip_full else ["full_domain", "adaptive_3sigma"]
    references: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for case in cases:
        fields = make_fields(args.reference_grid, args.query_size, args.channels, case, device)
        for support in supports:
            rendered = render(rasterizer, fields, args.query_size, support == "adaptive_3sigma")
            cell_area = 1.0 / (args.reference_grid * args.reference_grid)
            rendered["area_raw"] = rendered["raw"] * cell_area
            references[(case, support)] = rendered

    rows: list[dict] = []
    denominator_rows: list[dict] = []
    rendered_cache: dict[tuple[str, str, int], dict[str, torch.Tensor]] = {}
    for case in cases:
        for source_grid in source_grids:
            fields = make_fields(source_grid, args.query_size, args.channels, case, device)
            cell_area = 1.0 / (source_grid * source_grid)
            for support in supports:
                rendered = render(rasterizer, fields, args.query_size, support == "adaptive_3sigma")
                rendered_cache[(case, support, source_grid)] = rendered
                reference = references[(case, support)]
                area_raw = rendered["raw"] * cell_area
                normalized_metric = metrics(rendered["normalized"], reference["normalized"])
                area_metric = metrics(area_raw, reference["area_raw"])
                raw_metric = metrics(rendered["raw"], reference["area_raw"])
                constant_error = float((rendered["normalized"][:, 0] - 0.25).abs().max())
                row = {
                    "case": case,
                    "support": support,
                    "source_grid": source_grid,
                    "source_points": source_grid * source_grid,
                    "cell_area": cell_area,
                    "normalized_relative_l2": normalized_metric["relative_l2"],
                    "normalized_mae": normalized_metric["mae"],
                    "normalized_linf": normalized_metric["linf"],
                    "normalized_mean_abs": normalized_metric["mean_abs"],
                    "area_raw_relative_l2": area_metric["relative_l2"],
                    "area_raw_mae": area_metric["mae"],
                    "area_raw_linf": area_metric["linf"],
                    "area_raw_mean_abs": area_metric["mean_abs"],
                    "raw_to_integral_relative_l2": raw_metric["relative_l2"],
                    "raw_mean_abs": raw_metric["mean_abs"],
                    "constant_channel_max_abs_error": constant_error,
                }
                rows.append(row)
                denominator_rows.append(
                    {"case": case, "support": support, "source_grid": source_grid, **density_stats(rendered["density"])}
                )

    property_grid = source_grids[-1]
    property_fields = make_fields(property_grid, args.query_size, args.channels, "anisotropic_smooth", device)
    base = render(rasterizer, property_fields, args.query_size, True)
    opacity_errors = {}
    maximum_base_opacity = float(property_fields["opacity"].max())
    for multiplier in LEGAL_OPACITY_MULTIPLIERS:
        maximum_scaled_opacity = maximum_base_opacity * multiplier
        if maximum_scaled_opacity > 1.0:
            raise AssertionError(
                f"opacity multiplier {multiplier} leaves the legal [0,1] range"
            )
        scaled = render(rasterizer, property_fields, args.query_size, True, opacity_multiplier=multiplier)
        opacity_errors[str(multiplier)] = {
            "max_abs": float((scaled["normalized"] - base["normalized"]).abs().max()),
            "relative_l2": relative_l2(scaled["normalized"], base["normalized"]),
            "density_min": float(scaled["density"].min()),
            "maximum_scaled_opacity": maximum_scaled_opacity,
        }

    values = property_fields["values"]
    x, y = source_coordinates(property_grid, device)
    perturbation_scalar = 0.01 * torch.sin(2.0 * math.pi * x) * torch.cos(2.0 * math.pi * y)
    perturbation = perturbation_scalar.view(1, -1, 1).expand_as(values)
    perturbed = render(rasterizer, property_fields, args.query_size, True, values=values + perturbation)
    output_linf = float((perturbed["normalized"] - base["normalized"]).abs().max())
    input_linf = float(perturbation.abs().max())

    truncation_rows = []
    if not args.skip_full:
        for case in cases:
            for source_grid in source_grids:
                full = rendered_cache[(case, "full_domain", source_grid)]["normalized"]
                adaptive = rendered_cache[(case, "adaptive_3sigma", source_grid)]["normalized"]
                truncation_rows.append(
                    {
                        "case": case,
                        "source_grid": source_grid,
                        "adaptive_vs_full_relative_l2": relative_l2(adaptive, full),
                        "adaptive_vs_full_linf": float((adaptive - full).abs().max()),
                    }
                )

    config = {
        "protocol_version": PROTOCOL_VERSION,
        "state": "isolated synthetic operator diagnostic; no training and no checkpoint",
        "source_grids": source_grids,
        "reference_grid": args.reference_grid,
        "query_size": args.query_size,
        "channels": args.channels,
        "cases": cases,
        "supports": supports,
        "coordinate_convention": {
            "continuous_domain": "[0,1]^2",
            "source_cell_centers": "x=(i+0.5)/N, y=(j+0.5)/N",
            "output_pixel_coordinate": "mean_px=x*W_out-0.5, y*H_out-0.5",
            "sigma_conversion": "sigma_px=sigma_cont*W_out (and H_out)",
            "query_grid_fixed_while_source_density_changes": True,
        },
        "cell_area": "1/N^2 for every uniform source grid",
        "continuous_values": [
            "v0=0.25",
            "v1=0.50+0.20sin(2pi x)+0.10cos(2pi y)",
            "v2=0.15+0.70xy",
            "v3=0.50+0.20sin(4pi x)cos(2pi y)",
        ],
        "formal_support": "strict axis-aligned rectangle |dx|<3sigma_x and |dy|<3sigma_y",
        "denominator_floor": 1e-6,
        "seed": args.seed,
    }

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_status = None, None
    environment = {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "git_commit": git_commit,
        "git_status_short": git_status,
        "extension_path": str(extension_path),
        "extension_sha256": extension_sha,
    }

    summary_rows = convergence_summary(rows)
    for row in summary_rows:
        row["reference_grid"] = args.reference_grid
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_grid.csv", rows)
    write_csv(output_dir / "denominator_stats.csv", denominator_rows)
    if truncation_rows:
        write_csv(output_dir / "truncation_vs_full.csv", truncation_rows)
    atomic_write_text(output_dir / "config.json", json.dumps(config, indent=2))
    atomic_write_text(output_dir / "environment.json", json.dumps(environment, indent=2))
    atomic_write_text(
        output_dir / "environment.txt",
        "\n".join(f"{key}={value}" for key, value in environment.items()) + "\n",
    )
    make_plot(rows, output_dir / "convergence_plot.png")

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "config": config,
        "environment": environment,
        "per_grid": rows,
        "summary": summary_rows,
        "denominator_stats": denominator_rows,
        "truncation_vs_full": truncation_rows,
        "property_tests": {
            "opacity_common_scaling": {
                "note": "direct positive rasterizer-input scaling; every tested opacity remains within [0,1]",
                "multipliers": list(LEGAL_OPACITY_MULTIPLIERS),
                "maximum_base_opacity": maximum_base_opacity,
                "results": opacity_errors,
            },
            "value_non_expansiveness": {
                "input_perturbation_linf": input_linf,
                "output_difference_linf": output_linf,
                "satisfied_with_2e_6_tolerance": output_linf <= input_linf + 2e-6,
            },
            "constant_channel_max_abs_error": max(row["constant_channel_max_abs_error"] for row in rows),
        },
        "artifacts": {
            "config": str((output_dir / "config.json").resolve()),
            "per_grid": str((output_dir / "per_grid.csv").resolve()),
            "summary": str((output_dir / "summary.csv").resolve()),
            "denominator_stats": str((output_dir / "denominator_stats.csv").resolve()),
            "plot": str((output_dir / "convergence_plot.png").resolve()),
        },
    }
    atomic_write_text(result_path, json.dumps(report, indent=2))
    print(json.dumps({
        "result": str(result_path),
        "rows": len(rows),
        "max_constant_error": report["property_tests"]["constant_channel_max_abs_error"],
        "value_non_expansive": report["property_tests"]["value_non_expansiveness"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
