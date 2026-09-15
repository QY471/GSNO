#!/usr/bin/env python3
"""Strictly load the curated strong checkpoints with the canonical trainer."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Train_Cave
from tools.audit_registered_models import default_options


CHECKPOINTS = {
    "baseline": ("GSFusion_Baseline_Fixed_CAVE_4", {"dim": 32}),
    "gsno": ("GSFusion_GSNO_CAVE_4", {}),
    "gsno_cell_gaussian_difference": (
        "GSNO_DSSGR_CellDiff_FromScratch_CAVE4_400EP_S1_GPU0_FORMAL",
        {},
    ),
    "gsno_nogs_identity": ("GSFusion_GSNO_NoGS_Identity_CAVE_4_seed1_ep400", {}),
    "msi_guided_hsi_gs_scale_consistent": (
        "GSFusion_MSI_Guided_HSIGS_ScaleConsistentNorm3Sigma_CAVE4_400EP_GPU0",
        {},
    ),
}


def extract_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    results = []
    for model_name, (directory, option_overrides) in CHECKPOINTS.items():
        path = ROOT / "Checkpoint" / directory / "best_model.pth"
        try:
            options = default_options(model_name)
            for name, value in option_overrides.items():
                setattr(options, name, value)
            model = Train_Cave.build_model_bundle(options)[0]
            state = extract_state(torch.load(path, map_location="cpu", weights_only=False))
            if state and all(key.startswith("module.") for key in state):
                state = {key[7:]: value for key, value in state.items()}
            model.load_state_dict(state, strict=True)
            result = {"model": model_name, "checkpoint": str(path), "status": "ok"}
        except Exception as exc:
            result = {
                "model": model_name,
                "checkpoint": str(path),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    failed = [result for result in results if result["status"] != "ok"]
    print(f"SUMMARY total={len(results)} ok={len(results) - len(failed)} failed={len(failed)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
