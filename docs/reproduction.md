# Reproduction Notes

## CAVE

`model.gsno.GSNO` contains the model recorded as
`e3_constrained_elliptical_gaussian` in the original CAVE run. It uses the native
ADCI implementation and the adaptive CUDA Gaussian rasterizer. The source
package contains no alternative model versions.

| Setting | Recorded value |
|---|---|
| Training ratio | 4x |
| Feature dimension | 80 |
| Seed | 1 |
| Training budget | 1000 epochs |
| Selected epoch | 555 |
| Selected CAVE 4x PSNR | 52.6838439 dB |

The training script selects the checkpoint by test PSNR every five epochs.
It does not use a separate validation split. Cross-scale evaluation freezes
this checkpoint and evaluates it at 4x, 8x, 16x, and 32x.

The original weights are not included. The values above identify a recorded
experiment, not a fresh reproduction. A newly trained checkpoint must be
evaluated with its own selection metadata.

## Harvard

The training loader expects 67 training and 10 test MAT files in the original
experiment order. Its default spatial support is the top-left 1024 x 1024
region. The repository does not supply dataset conversion or split files.

Training-time evaluation uses the same 1024 x 1024 support. The original
Harvard checkpoint and its cross-scale evaluation configuration are not
included, so the Harvard paper table is not yet reproducible from this
package alone.

## Verification

The CPU tests cover command-line imports, PSF-to-OTF conversion, degradation,
data-loader shapes, model aliases, and documentation links. They do not test
CUDA rendering, full training, or the reported PSNR.

Building the paper model requires a compatible CUDA toolkit with `nvcc` and a
C++ compiler. Installing CUDA-enabled PyTorch alone does not build the bundled
extension. Follow the in-place build command in the main README.
