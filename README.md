# GSNO: Gaussian Spatial-Spectral Neural Operator

This repository contains the research code for **GSNO**, a neural-operator
framework for spatial-spectral fusion of a low-resolution hyperspectral image
(LR-HSI) and a high-resolution multispectral image (HR-MSI).

GSNO is evaluated under a single-ratio training protocol: the model is trained
at `4x` and the same frozen parameters are evaluated at `4x`, `8x`, `16x`, and
`32x`. The paper result reported as `52.68 dB` on CAVE is produced by
`e3_constrained_elliptical_gaussian`, using the native ADCI path and the CUDA
Gaussian renderer. It is not the later `ADCICUDAExactContinuous` candidate.

> **Release status.** This branch is the public-release preparation branch.
> Dataset files and trained weights are intentionally excluded. The CUDA
> rasterizer files retain their upstream non-commercial research license; see
> [`third_party/README.md`](third_party/README.md) before redistribution.

## Abstract

Spatial-spectral fusion (SSF) reconstructs a high-resolution hyperspectral
image by fusing a low-resolution hyperspectral image with a high-resolution
multispectral image. However, spatial interactions learned over discrete
observations remain tied to the observation grid, limiting the transfer of the
fusion mapping across fusion ratios. In this work, we propose the Gaussian
Spatial-Spectral Neural Operator (GSNO), an SSF framework that learns fusion
mappings between functions sampled on different grids. Specifically, Local
Kernel Interaction (LKI) parameterizes neighborhood aggregation through
center-neighbor feature relations, and Gaussian Spatial Integral Operator
(GSIO) predicts anisotropic Gaussian kernels from the fused representation for
spatial integration over the reference domain. Extensive experiments on CAVE
and Harvard show that GSNO trained only at $4\times$ consistently outperforms
competing methods at unseen fusion ratios from $8\times$ to $32\times$,
exceeding the second-best method by 2.07 dB in PSNR on CAVE at $32\times$.

Paper: [current manuscript repository](https://github.com/QY471/icassp)

## Repository Layout

```text
GSFusion/
├── Train_Cave.py                 # shared CAVE/Harvard training entry point
├── Train_Harvard.py              # Harvard convenience entry point
├── datasets/                     # dataset loaders and degradation protocol
├── model/                        # GSNO model and controlled variants
├── extensions/                   # CUDA/Triton acceleration modules
├── tools/                        # metrics, evaluation, and audit utilities
├── configs/datasets.yaml         # local dataset path template
├── scripts/                      # reproducible train/evaluation commands
├── checkpoints/README.md         # checkpoint release table
├── requirements-public.txt       # minimal public dependency list
├── CITATION.cff                  # machine-readable citation
└── third_party/                  # provenance and license boundaries
```

The formal paper model is registered as
`e3_constrained_elliptical_gaussian`. In-repository
controls remain in `model/` for the main ablation protocol. Official baseline
implementations and their private source snapshots are intentionally omitted;
comparison methods should be obtained from their own releases.

The paper's CAVE main result uses this model with `dim=80`, `seed=1`,
`ep_total=1000`, and `sf=4`. The selected checkpoint is at epoch `555` and
reports `52.6838439 dB` at `4x` (rounded to `52.68 dB` in the paper).

## Installation

The main model uses PyTorch, CUDA, and Triton. Use a Python environment that
matches the installed CUDA toolkit, then install the public dependencies:

```bash
conda create -n gsno python=3.10 -y
conda activate gsno
pip install --upgrade pip
pip install -r requirements-public.txt
```

The adaptive Gaussian rasterizer is compiled on first use. A CUDA compiler and
the PyTorch CUDA build must be available. CPU-only execution is supported only
for inspecting the pure-PyTorch utilities; it is not a reproduction of the
reported training protocol.

## Data Preparation

The repository does not redistribute CAVE or Harvard. Download the datasets
from their official sources and arrange the local paths as follows:

```text
<DATA_ROOT>/Cave/
├── Train/
└── Test/

<DATA_ROOT>/Harvard/
├── Train/
└── Test/
```

The CAVE loader expects the standard `Train.txt` and `Test.txt` lists together
with the HSI/MSI files. The Harvard loader expects numbered MAT files with
`HS` and `HRMS` arrays. Set the paths in
[`configs/datasets.yaml`](configs/datasets.yaml), or pass them directly on the
command line.

The released degradation protocol applies a Gaussian blur with standard
deviation `2.0` followed by phase-aligned downsampling. Training patches are
sampled from the HR reference image; evaluation uses the complete test scenes
or the documented evaluation crop.

## Training

The paper-facing configuration uses the same model parameters for CAVE and
Harvard, with only the dataset paths and split changing.

```bash
python Train_Cave.py \
  --dataset cave \
  --data_path /path/to/Cave/Train \
  --test_data_path /path/to/Cave/Test \
  --model e3_constrained_elliptical_gaussian \
  --sf 4 --dim 80 --ep_total 1000 --e_every 5 \
  --checkpoint_root Checkpoint_CAVE
```

For Harvard:

```bash
python Train_Cave.py \
  --dataset harvard \
  --data_path /path/to/Harvard/Train \
  --test_data_path /path/to/Harvard/Test \
  --model e3_constrained_elliptical_gaussian \
  --sf 4 --dim 80 --ep_total 1000 --e_every 5 \
  --checkpoint_root Checkpoint_Harvard
```

Convenience launchers are provided in `scripts/`. Each run writes its
configuration, logs, and checkpoints under the selected checkpoint root.

## Evaluation

To evaluate a frozen CAVE checkpoint at several ratios, use the public
multiscale evaluator:

```bash
python tools/evaluate_dynamic_model_multiscale.py \
  --module model.GSFusion_E3_ConstrainedEllipticalGaussian \
  --checkpoint /path/to/model_best.pth \
  --data-path /path/to/Cave/Test \
  --scales 4 8 16 32 \
  --dim 80 \
  --selected-4x-best-epoch 555 \
  --selected-4x-best-psnr 52.6838439 \
  --output results/cave_cross_scale.json
```

The `selected-4x` arguments record the checkpoint-selection provenance. For a
released checkpoint, replace them with the values stored in its run metadata.
The evaluator reports PSNR, SAM, ERGAS with the fixed reference factor `4`,
SSIM, and per-image CSV results.

## Checkpoints and Results

Weights are distributed separately from the source tree so that the Git
repository stays small and dataset licenses are respected. The checkpoint
table and SHA256 fields belong in [`checkpoints/README.md`](checkpoints/README.md)
when the final assets are approved.

## Citation

```bibtex
@inproceedings{shi2027gsno,
  title     = {Gaussian Spatial-Spectral Neural Operator},
  author    = {Shi, Qinyi and Zhu, Junwei and Zhang, Mouyi and Xu, Honghui and Zheng, Jianwei},
  booktitle = {2027 IEEE International Conference on Acoustics, Speech, and Signal Processing},
  year      = {2027}
}
```

## License

The original GSNO source is intended for academic research. Files copied or
adapted from external projects keep their own license and attribution notices;
the most visible case is the Inria/MPII Gaussian rasterizer under
`extensions/`. See [`third_party/README.md`](third_party/README.md) before
redistributing a release archive.

## Contact

For reproduction questions, open an issue in the release repository or contact
the corresponding author listed in the paper.
