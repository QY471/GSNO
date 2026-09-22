# Models

Use `from model.gsno import GSNO` for the model associated with the recorded
CAVE 52.68 dB result. `GSNO` is an alias of the existing class, not a new wrapper
network, and does not add prefixes to checkpoint keys.

The shared trainer accepts `--model gsno` or
`--model e3_constrained_elliptical_gaussian` for this model. The direct class
named `GSFusion` inside `GSFusion_GSNO.py` is an earlier architecture; that file
also supplies the ADCI layers and losses used by the paper model.

## Main Implementation

| File | Role |
|---|---|
| `gsno.py` | Public model and loss exports |
| `GSFusion_E3_ConstrainedEllipticalGaussian.py` | Elliptical kernels and paper model |
| `GSFusion_HRFused_Circular_PrimitiveEmbedding.py` | Shared feature extraction and primitive embedding |
| `GSFusion_GSNO.py` | ADCI layers and reconstruction losses |

The remaining files contain experimental architectures and controls.
In particular, `ADCICUDAExactContinuous` variants are not interchangeable with
the native-ADCI paper model. Do not use them to evaluate its checkpoint simply
because their tensor shapes match. Existing module names are retained to avoid
breaking experiment configurations and imports.
