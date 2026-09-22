# Reproducible launchers

Run the launchers from the repository root. They require Bash, Python,
CUDA-enabled PyTorch, the compiled rasterizer, and prepared datasets.

```bash
export DATA_ROOT=/path/to/datasets
export GPU=0
bash scripts/run_train_cave_x4.sh
```

For frozen cross-scale evaluation, provide the checkpoint-selection metadata
recorded by the training run:

```bash
export DATA_ROOT=/path/to/datasets
export CHECKPOINT=/path/to/best_model.pth
export BEST_EPOCH=555
export BEST_PSNR=52.6838439
bash scripts/run_eval_cave_cross_scale.sh
```

These values correspond to the paper's CAVE `4x` checkpoint. If a different
run is evaluated, replace both values with the metadata printed by that run.
The launcher refuses to guess checkpoint provenance.
