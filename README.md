<div align="center">

# Gaussian Spatial-Spectral Neural Operator

### Cross-Scale Spatial-Spectral Fusion with Gaussian Operators

Qinyi Shi, Junwei Zhu, Mouyi Zhang, Honghui Xu, and Jianwei Zheng<sup>†</sup>

Zhejiang University of Technology

<sub>† Corresponding author: [zjw@zjut.edu.cn](mailto:zjw@zjut.edu.cn)</sub>

</div>

<p align="center">
  <img src="assets/gsno_framework.png" alt="Architecture of GSNO" width="95%">
  <br>
  <em>Architecture of GSNO.</em>
</p>

GSNO reconstructs a high-resolution hyperspectral image from a low-resolution
hyperspectral image and a high-resolution multispectral image. Local Kernel
Interaction (LKI) learns neighborhood weights from feature relations, while the
Gaussian Spatial Integral Operator (GSIO) integrates spatial information with
adaptive anisotropic Gaussian kernels. The paper evaluates a model trained at
`4x` on fusion ratios from `4x` to `32x` without retraining.

This repository provides the model, training code, and CAVE/Harvard evaluation
scripts. Prepared datasets and pretrained checkpoints are not included; train a
checkpoint before running evaluation.

## Installation

Training and inference require an NVIDIA GPU, a compatible CUDA-enabled PyTorch
installation, `nvcc`, and a C++ compiler. The following commands compile the
bundled Gaussian rasterizer in place:

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

Install a PyTorch build compatible with your CUDA toolkit if the build supplied
by `pip install -r requirements.txt` does not match your system.

## Data

Place the prepared CAVE and Harvard files under a common root:

```text
<DATA_ROOT>/
├── Cave/
│   ├── Train/
│   │   ├── Train.txt
│   │   ├── HSI/<scene>.mat
│   │   └── RGB/<scene>.mat
│   └── Test/
│       ├── Test.txt
│       ├── HSI/<scene>.mat
│       └── RGB/<scene>.mat
└── Harvard/
    ├── Train/1.mat ... 67.mat
    └── Test/1.mat ... 10.mat
```

For CAVE, each text file lists scene names without the `.mat` suffix. The MAT
keys are `hsi` (31 bands) and `rgb` (3 bands); the paper uses 20 training and
12 test scenes. Harvard MAT files contain `HS` (`1040 x 1392 x 31`) and `HRMS`
(`1040 x 1392 x 3`). Harvard evaluation uses the top-left `1024 x 1024` crop
of each test image.

The loaders expect aligned HR-HSI and HR-MSI arrays scaled to `[0, 1]`. They
generate LR-HSI online with Gaussian blur (standard deviation 2) and
downsampling. HR-MSI preparation with the Nikon D700 spectral response is not
part of the released loaders. The exact CAVE scene lists and Harvard
raw-scene-to-file mapping used for the paper are also not distributed here.
The original HSI datasets are available from
[CAVE](https://cave.cs.columbia.edu/repository/Multispectral) and
[Harvard](https://vision.seas.harvard.edu/hyperspec/download.html); these raw
downloads require preparation for the layout above. The
[SFNO repository](https://github.com/weili419/SFNO) also links to prepared
CAVE data; check its split and MAT layout before using it with GSNO.

The Harvard loader preloads the 67 training scenes into host memory, requiring
approximately 12.3 GiB for its float32 HSI and MSI arrays.

## Training

Set `DATA_ROOT` to the directory containing `Cave/` and `Harvard/`, then run
the script for the dataset and training ratio you need:

```bash
DATA_ROOT=/path/to/data bash scripts/run_train_cave_x4.sh
DATA_ROOT=/path/to/data bash scripts/run_train_harvard_x4.sh
DATA_ROOT=/path/to/data bash scripts/run_train_cave_x8.sh
DATA_ROOT=/path/to/data bash scripts/run_train_harvard_x8.sh
```

Each script trains a separate model. The defaults are 64-pixel HR patches,
batch size 32, latent width 80, seed 1, and 1,000 epochs. The `4x` CAVE run,
for example, writes its best weights and log to
`Checkpoint_CAVE/GSNO_CAVE_x4/`. The training code selects `best_model.pth` by
PSNR on the path passed as `--test_data_path`; these scripts use the `Test`
folder for both model selection and reported evaluation.

The shell scripts accept `GPU`, `EPOCHS`, and `CHECKPOINT_ROOT` environment
variables. The Python entry points expose the remaining options through
`python Train_Cave.py --help` and `python Train_Harvard.py --help`.

## Evaluation

The evaluator loads one frozen checkpoint and measures PSNR, SAM, ERGAS, and
SSIM at the requested ratios. It writes aggregate results to JSON and
per-image results to CSV. To evaluate a newly trained CAVE `4x` model, read
the best epoch and PSNR from the final line of
`Checkpoint_CAVE/GSNO_CAVE_x4/training.log`, then set:

```bash
export DATA_ROOT=/path/to/data
export CHECKPOINT=Checkpoint_CAVE/GSNO_CAVE_x4/best_model.pth
export BEST_EPOCH=YOUR_BEST_EPOCH
export BEST_PSNR=YOUR_BEST_PSNR
bash scripts/run_eval_cave_cross_scale.sh
```

Use `scripts/run_eval_harvard_cross_scale.sh` and the corresponding Harvard
checkpoint for Harvard. Both scripts evaluate `4x`, `8x`, `16x`, and `32x` by
default. To evaluate a separately trained `8x` model at its training ratio,
set `SELECTION_SCALE=8` and `SCALES="8"`. The recorded best PSNR must belong
to the checkpoint being evaluated; the script checks it against the measured
PSNR at the selection ratio. `BEST_EPOCH` is recorded as metadata because the
weight file itself does not encode the epoch.

For consistency with the paper, the reported ERGAS reference factor is fixed
at 4 across evaluation ratios. The JSON report also includes ERGAS calculated
with the actual ratio. See [checkpoints/README.md](checkpoints/README.md) for
checkpoint availability.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `model/gsno.py` | GSNO model and reconstruction loss |
| `datasets/` | CAVE and Harvard loaders |
| `extensions/adaptive3_rasterizer/` | CUDA Gaussian rasterizer |
| `Train_Cave.py`, `Train_Harvard.py` | Training entry points |
| `tools/evaluate_dynamic_model_multiscale.py` | Frozen-checkpoint evaluation |
| `scripts/` | Dataset and ratio-specific commands |

The model class is `model.gsno.GSNO`; `GSFusion` is an equivalent alias kept
for checkpoint compatibility. Its `adci_layers` constructor argument denotes
the paper's LKI stages. The evaluator loads model weights with `strict=True`.

## Checks

With the dependencies installed, run the CPU release tests and the tracked-file
check from the repository root:

```bash
python -m unittest discover -s tests -v
python tools/check_public_release.py
```

These checks cover loader behavior, command-line arguments, local documentation
links, and tracked-file safety. They do not train GSNO or reproduce the paper's
reported results.

## Citation

```bibtex
@misc{shi2026gsno,
  title  = {Gaussian Spatial-Spectral Neural Operator},
  author = {Shi, Qinyi and Zhu, Junwei and Zhang, Mouyi and Xu, Honghui and Zheng, Jianwei},
  year   = {2026},
  note   = {Unpublished manuscript},
  url    = {https://github.com/QY471/GSNO}
}
```

## License and Attribution

No license has been specified for the original GSNO code. The bundled
Gaussian rasterizer retains its upstream non-commercial research license.
Component provenance and license details are documented in
[third_party/README.md](third_party/README.md).
