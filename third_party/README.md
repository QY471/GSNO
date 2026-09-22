# Third-Party Components

## Gaussian Rasterizer

The rasterizer headers credit Inria / GRAPHDECO and point to
[diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization).
The Python wrapper cites upstream commit
`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` as an implementation reference.
This is a reference recorded in the code, not a verified base commit for every file.

The unmodified license from that commit is included as
[LICENSE_GAUSSIAN_SPLATTING.md](LICENSE_GAUSSIAN_SPLATTING.md) and copied into
the bundled rasterizer directory, `extensions/adaptive3_rasterizer/`.

Local changes include spectral-channel rendering and adaptive support windows.
The Gaussian-Splatting license limits use to non-commercial research and
evaluation. Retain the full license and upstream copyright notices when
redistributing these sources. These components are not MIT-licensed.

## Fusion Components and Data Protocol

The ADCI local-interaction implementation and the Harvard data protocol were
adapted from the AFNO code supplied to this project. Renaming the public entry
points does not change this provenance. The exact upstream revision and
redistribution terms for those supplied sources still need author confirmation.

## Dependencies

The PSF-to-OTF conversion uses the installed
[PyPHER](https://github.com/aboucaud/pypher) package (BSD-3-Clause).
No PyPHER source is bundled here. PyTorch and the other installed
dependencies retain their respective licenses.

## Project License

The authors have not selected a license for the original GSNO code. This source
package must not be represented as an MIT release or as fully license-cleared
until the project license and the outstanding source attributions are confirmed.
