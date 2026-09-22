# GSNO: Gaussian Spatial-Spectral Neural Operator

This repository contains the research code for **GSNO**, a neural-operator
framework for spatial-spectral fusion of a low-resolution hyperspectral image
(LR-HSI) and a high-resolution multispectral image (HR-MSI).

GSNO is evaluated under a single-ratio training protocol: the model is trained
at `4x` and the same frozen parameters are evaluated at `4x`, `8x`, `16x`, and
`32x`. The paper result reported as `52.68 dB` on CAVE is produced by
`e3_constrained_elliptical_gaussian`, using the native ADCI path and the CUDA
Gaussian renderer. `model/gsno.py` exposes this same class as `GSNO` without
changing its parameters or checkpoint keys.

> **Status.** This is a pre-submission source package. Dataset files and trained
> weights are not included. The CUDA
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
GSNO/
├── Train_Cave.py                 # shared CAVE/Harvard training entry point
├── Train_Harvard.py              # Harvard convenience entry point
├── datasets/                     # dataset loaders and degradation protocol
├── model/                        # GSNO model and controlled variants
├── extensions/                   # CUDA/Triton acceleration modules
├── tools/                        # metrics and evaluation utilities
├── configs/datasets.yaml         # local dataset path template
├── scripts/                      # reproducible train/evaluation commands
├── checkpoints/README.md         # checkpoint release table
├── requirements.txt              # minimal public dependency list
├── CITATION.cff                  # machine-readable citation
└── third_party/                  # provenance and license boundaries
```

The formal paper model is registered as
`e3_constrained_elliptical_gaussian`. In-repository
experimental variants remain in `model/`; they are not all paper ablations.
Their existing module paths are retained for checkpoint compatibility. Official baseline
implementations are intentionally omitted; comparison methods should be obtained
from their official releases.

The paper's CAVE main result uses this model with `dim=80`, `seed=1`,
`ep_total=1000`, and `sf=4`. The selected checkpoint is at epoch `555` and
reports `52.6838439 dB` at `4x` (rounded to `52.68 dB` in the paper).

## Installation

The paper model uses PyTorch and a compiled CUDA rasterizer. Linux with an
NVIDIA GPU, a compatible CUDA toolkit (including `nvcc`), and a C++ compiler
is recommended. Triton is used only by the optional accelerated variants.
Install a CUDA-enabled PyTorch build matching your toolkit before proceeding:

```bash
git clone https://github.com/QY471/GSNO.git
cd GSNO

conda create -n gsno python=3.10 -y
conda activate gsno
pip install --upgrade pip
pip install -r requirements.txt

cd extensions/adaptive3_rasterizer
python setup.py build_ext --inplace
cd ../..
python -c "from model.gsno import GSNO; GSNO(dim=80, num_bands=31, num_msi=3, adci_layers=3)"
```

Build the extension in place: the model imports the bundled source directory.
The extension is not compiled automatically. CPU-only execution supports the
data utilities and release tests, not the paper model. The original trained
weights are not included, so this package alone does not verify the reported PSNR.

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

For CAVE, each split contains `HSI/<scene>.mat` (key `hsi`, shape
`512 x 512 x 31`) and `RGB/<scene>.mat` (key `rgb`, shape `512 x 512 x 3`).
`Train/Train.txt` and `Test/Test.txt` list scene names without extensions,
one per line. The loader expects 20 training and 12 test scenes.

For Harvard, `Train/1.mat` through `Train/67.mat` and `Test/1.mat` through
`Test/10.mat` contain `HS` (`1040 x 1392 x 31`) and `HRMS`
(`1040 x 1392 x 3`). The training loader samples from the top-left
`1024 x 1024` region. Supply the prepared paired MAT files in the original
experiment order; this repository does not include dataset conversion or split files.

Pass paths using `--data_path` and `--test_data_path`, or set `CAVE_ROOT` /
`HARVARD_ROOT`. `configs/datasets.yaml` is a path template only; the training
scripts do not read it automatically.

The released degradation protocol applies a Gaussian blur with standard
deviation `2.0` followed by phase-aligned downsampling. Training patches are
sampled from the HR reference image; evaluation uses the complete test scenes
or the documented evaluation crop.

## Training

The CAVE configuration associated with the reported result is:

```bash
python Train_Cave.py \
  --dataset cave \
  --data_path /path/to/Cave/Train \
  --test_data_path /path/to/Cave/Test \
  --model e3_constrained_elliptical_gaussian \
  --sf 4 --dim 80 --ep_total 1000 --e_every 5 \
  --checkpoint_root Checkpoint_CAVE
```

The same shared trainer can run on Harvard (example configuration, not a
verified reproduction of the Harvard table):

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
TensorBoard events are written to `run/<run_name>/`.
The training script selects `best_model.pth` using the PSNR on
`--test_data_path` every `--e_every` epochs. This is the existing experiment
protocol; it does not use a separate validation split.

## Evaluation

To evaluate a frozen CAVE checkpoint at several ratios, use the public
multiscale evaluator:

```bash
python tools/evaluate_dynamic_model_multiscale.py \
  --module model.gsno \
  --checkpoint /path/to/best_model.pth \
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

`tools/evaluate_harvard_multiscale.py` is a separate diagnostic using a
top-left `512 x 512` crop and a required dataset manifest. It does not use the
training loader's default `1024 x 1024` evaluation crop. The matching manifest
and checkpoint have not been supplied with this package; the Harvard table is
not yet independently reproducible from the repository alone.

## Checkpoints and Results

Trained weights have not been released. See
[`checkpoints/README.md`](checkpoints/README.md) for availability. The CAVE
PSNR above is a recorded experiment result, not a fresh reproduction from this
package. Do not substitute the metadata of that run for a newly trained model.

## Tests

```bash
python -m unittest discover -s tests -v
python tools/check_public_release.py
```

These tests cover imports, data degradation, documentation links, and model
aliases. They do not validate CUDA rendering or the reported paper metrics.
The safety check scans tracked working-tree files only, not Git history or
third-party distribution rights.

## Citation

```bibtex
@misc{shi2026gsno,
  title     = {Gaussian Spatial-Spectral Neural Operator},
  author    = {Shi, Qinyi and Zhu, Junwei and Zhang, Mouyi and Xu, Honghui and Zheng, Jianwei},
  year      = {2026},
  note      = {Unpublished manuscript},
  url       = {https://github.com/QY471/GSNO}
}
```

## License

The original GSNO code does not yet have an author-approved distribution
license. Bundled rasterizers retain the Inria/MPII research-only license.
AFNO-derived components and EDSR provenance also require author confirmation.
See [`third_party/README.md`](third_party/README.md) before public redistribution.

## Contact

For reproduction questions, open an issue in the release repository or contact
the corresponding author listed in the paper.
