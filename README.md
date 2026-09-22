<div align="center">

# Gaussian Spatial-Spectral Neural Operator

### Cross-Scale Spatial-Spectral Fusion with Gaussian Operators

Qinyi Shi, Junwei Zhu, Mouyi Zhang, Honghui Xu, and Jianwei Zheng

Zhejiang University of Technology

<sub>* Corresponding author</sub>

</div>

<p align="center">
  <img src="assets/gsno_framework.png" alt="GSNO framework" width="95%">
  <br>
  <em>Overall architecture of GSNO.</em>
</p>

> This manuscript is currently under review. Dataset files and trained weights
> are not included in the current source package.

## Highlights

- GSNO learns a spatial-spectral fusion mapping from LR-HSI and HR-MSI.
- Local Kernel Interaction (LKI) models neighborhood feature relations on the
  observation grid.
- Gaussian Spatial Integral Operator (GSIO) performs density-normalized local
  integration on the HR reference grid.
- One model trained at `4x` is evaluated at `4x`, `8x`, `16x`, and `32x`.

## Repository Layout

```text
GSNO/
├── model/gsno.py                  # GSNO model and reconstruction loss
├── datasets/                      # CAVE and Harvard loaders
├── extensions/                    # CUDA Gaussian rasterizer
├── tools/                         # Metrics and multiscale evaluation
├── scripts/                       # Training and evaluation launchers
├── checkpoints/                   # Checkpoint availability
├── tests/                         # CPU regression tests
├── Train_Cave.py                 # Shared training entry point
├── Train_Harvard.py              # Harvard training wrapper
├── requirements.txt              # Python dependencies
├── CITATION.cff                  # Citation metadata
└── third_party/                  # Provenance and license information
```

The package contains one model entry point, `model.gsno.GSNO`. Experimental
architectures and comparison implementations are not included.

## Installation

The paper model uses PyTorch and a compiled CUDA rasterizer. A compatible
CUDA toolkit with `nvcc`, an NVIDIA GPU, and a C++ compiler are required for
model training and inference.

```bash
git clone https://github.com/QY471/GSNO.git
cd GSNO

conda create -n gsno python=3.10 -y
conda activate gsno
pip install -r requirements.txt

cd extensions/adaptive3_rasterizer
python setup.py build_ext --inplace
cd ../..
```

The extension is compiled in place. CPU-only execution supports the data
utilities and release tests, but not the paper model.

## Data Preparation

The repository does not redistribute CAVE or Harvard. Arrange the datasets as:

```text
<DATA_ROOT>/Cave/
├── Train/
└── Test/

<DATA_ROOT>/Harvard/
├── Train/
└── Test/
```

CAVE uses `HSI/<scene>.mat` with key `hsi` and `RGB/<scene>.mat` with key
`rgb`. The loader expects 20 training scenes and 12 test scenes.

Harvard uses numbered MAT files containing `HS` and `HRMS`. The training
protocol loads 67 training scenes and 10 test scenes and samples the top-left
`1024 x 1024` region.

Set `CAVE_ROOT` or `HARVARD_ROOT`, or pass `--data_path` and
`--test_data_path` to the training command.

## Model Zoo

Weights are not yet released. The recorded CAVE checkpoint reached
`52.6838439 dB` at epoch `555`; this is historical run metadata, not a fresh
reproduction from the source package. Released files and checksums will be
listed in [checkpoints/README.md](checkpoints/README.md).

## Evaluation

Evaluate a frozen CAVE checkpoint across fusion ratios:

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

The evaluator reports PSNR, SAM, ERGAS, SSIM, and per-image results. The same
frozen parameters are used for every requested ratio.

## Training

The recorded CAVE configuration is:

| Setting | Value |
|---|---:|
| Dataset | CAVE |
| Training ratio | `4x` |
| Feature dimension | `80` |
| Seed | `1` |
| Epochs | `1000` |
| Evaluation interval | `5` epochs |

```bash
python Train_Cave.py \
  --dataset cave \
  --data_path /path/to/Cave/Train \
  --test_data_path /path/to/Cave/Test \
  --model gsno \
  --sf 4 --dim 80 --ep_total 1000 --e_every 5 \
  --checkpoint_root Checkpoint_CAVE
```

For Harvard, use `Train_Harvard.py` with the corresponding data root. The
training script selects `best_model.pth` using PSNR on `--test_data_path` every
`--e_every` epochs, following the recorded experiment protocol.

## Useful Commands

```bash
python -m unittest discover -s tests -v
python tools/check_public_release.py
```

These commands check imports, degradation, loader shapes, documentation links,
model aliases, and tracked-file safety. They do not reproduce the paper PSNR.

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
license. The bundled Gaussian rasterizer retains its upstream non-commercial
research license. See [third_party/README.md](third_party/README.md) before
redistribution.
