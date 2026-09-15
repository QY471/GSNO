#!/usr/bin/env python3
"""Run a real CUDA forward/backward smoke test for every registered model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import Train_Cave
from tools.audit_registered_models import default_options


MODEL_OPTION_OVERRIDES = {
    "baseline": {"dim": 32},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--sf", type=int, default=4)
    parser.add_argument("--hr-size", type=int, default=64)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Gaussian rasterizer")
    if args.hr_size % args.sf:
        raise ValueError("--hr-size must be divisible by --sf")

    names = sorted(Train_Cave.MODEL_SPECS)
    if args.models:
        unknown = sorted(set(args.models) - set(names))
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")
        names = args.models

    results = []
    for index, name in enumerate(names):
        torch.manual_seed(1000 + index)
        torch.cuda.manual_seed_all(1000 + index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        options = default_options(name)
        for key, value in MODEL_OPTION_OVERRIDES.get(name, {}).items():
            setattr(options, key, value)

        model, label, _loss = Train_Cave.build_model_bundle(options)
        model = model.cuda().train(not args.forward_only)
        lr_size = args.hr_size // args.sf
        lr_hsi = torch.rand(
            1, options.num_bands, lr_size, lr_size,
            device="cuda", requires_grad=not args.forward_only,
        )
        hr_msi = torch.rand(
            1, options.num_msi, args.hr_size, args.hr_size,
            device="cuda",
        )

        torch.cuda.synchronize()
        started = time.perf_counter()
        output = model(lr_hsi, hr_msi, args.sf)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{name} returned {type(output).__name__}, expected Tensor")
        if output.shape != (1, options.num_bands, args.hr_size, args.hr_size):
            raise RuntimeError(f"{name} returned unexpected shape {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{name} forward produced NaN/Inf")

        if not args.forward_only:
            output.square().mean().backward()
            if lr_hsi.grad is None or not torch.isfinite(lr_hsi.grad).all():
                raise RuntimeError(f"{name} backward produced an invalid input gradient")

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "model": name,
            "label": label,
            "status": "ok",
            "shape": list(output.shape),
            "forward_backward_ms": round(elapsed_ms, 3),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

        del model, lr_hsi, hr_msi, output
        gc.collect()
        torch.cuda.empty_cache()

    print(f"SUMMARY total={len(results)} ok={len(results)} failed=0")


if __name__ == "__main__":
    main()
