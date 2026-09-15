# GSFusion public-release staging

This directory is an internal, non-public staging copy for a future GSFusion release. It is intentionally separate from the research workspace and does not contain the original datasets, checkpoints, training logs, private experiment reports, or third-party baseline repositories.

## Scope

The staging tree contains the reusable implementation needed to inspect and reproduce the core fusion pipeline:

- `Train_Cave.py` and `Train_Harvard.py`: training entry points;
- `model/`: model implementations and controls;
- `datasets/`: dataset loaders and degradation code only;
- `ops/`, `extensions/`, and `submodules/`: renderer/CUDA support that still requires provenance and license review;
- `tools/`: selected smoke, audit, evaluation, and metric utilities;
- `requirements.txt`: the current server environment lock, retained for reference rather than presented as a final public installer.

## Current formal model

The paper-facing model is registered as:

```text
e3_constrained_elliptical_gaussian_adci_cuda_exact_continuous
```

It uses the constrained elliptical Gaussian, ADCI Exact CUDA, density normalization, adaptive 3-sigma support, and continuous output. The exact checkpoint and its metrics remain in the private research archive; no weights are included in this staging tree.

## Before publication

This tree is not yet a public release. Before publishing, we must:

1. choose and add a license;
2. replace server-specific paths and environment assumptions;
3. split a minimal public dependency file from the server lock;
4. document CAVE/Harvard data preparation without redistributing datasets;
5. audit every CUDA/submodule and third-party dependency license;
6. add a clean train/evaluate smoke test on a small synthetic fixture;
7. decide which trained weights, if any, can be released separately;
8. run a clean-machine reproduction test from this tree.

See `THIRD_PARTY_PROVENANCE_PENDING.md`, `RELEASE_EXCLUSIONS_ZH.md`, and `docs/` for the release audit notes.

## Private source of truth

The complete research workspace, checkpoints, logs, figures, reports, and data are preserved separately in the retirement backup. This staging tree must not be treated as a replacement for that archive until the release audit is complete.
