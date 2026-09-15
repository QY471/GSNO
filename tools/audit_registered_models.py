#!/usr/bin/env python3
"""Import and instantiate every model registered by Train_Cave.py."""

import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Train_Cave


def default_options(model_name):
    return SimpleNamespace(
        model=model_name,
        sf=4,
        dim=64,
        num_bands=31,
        num_msi=3,
        num_basis=16,
        num_gs_layers=3,
        edsr_resblocks=6,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    names = sorted(Train_Cave.MODEL_SPECS)
    if args.only:
        requested = set(args.only)
        names = [name for name in names if name in requested]

    results = []
    for name in names:
        try:
            bundle = Train_Cave.build_model_bundle(default_options(name))
            model, label = bundle[0], bundle[1]
            params = sum(parameter.numel() for parameter in model.parameters())
            result = {
                "model": name,
                "status": "ok",
                "label": label,
                "parameters": params,
            }
            del model
            gc.collect()
        except Exception as exc:  # Audit must continue after an isolated failure.
            result = {
                "model": name,
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
