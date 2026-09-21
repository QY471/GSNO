# Third-party code and license boundaries

The public GSNO code contains CUDA rasterization components derived from the
Inria/MPII Gaussian-splatting rasterizer. Those files retain their upstream
copyright notices and are restricted to non-commercial research and evaluation
under the upstream license. They must not be relicensed as MIT or included in
a commercial distribution without permission from the upstream licensors.

The Triton implementation under `extensions/adci_exact_triton/` is part of the
GSNO implementation and depends on the Triton package. The Python dependency
licenses are governed by their respective upstream projects.

`tools/Utils.py` uses PyPHER for the PSF-to-OTF helper. The release falls back
to the PyPI package when the historical local copy is absent; PyPHER is a
three-clause BSD project and must retain its upstream attribution.

Before publishing a release archive, record for every bundled external
component:

1. upstream repository and commit;
2. license text and attribution notice;
3. local modifications;
4. whether source redistribution is permitted.

The repository intentionally excludes external baseline implementations. They
must be installed from their own official repositories when reproducing a
comparison.
