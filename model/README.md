# Model

`gsno.py` contains the GSNO network used by the recorded CAVE 1000-epoch run.
There are no alternative model versions in this package.

```python
from model.gsno import GSNO

model = GSNO(dim=80, num_bands=31, num_msi=3, adci_layers=3)
```

The constructor argument `adci_layers` is the code-level compatibility name
for the paper's Local Kernel Interaction (LKI) stages; it is retained to keep
the recorded checkpoint parameter layout unchanged.

The implementation includes local feature interaction, a fusion backbone,
elliptical Gaussian integration, and the reconstruction loss. These are
components of one network, not separate experiment configurations.

The CUDA extension in `extensions/adaptive3_rasterizer/` is required.
`GSFusion` is an alias for `GSNO`; both names construct the same class.
Parameter names and initialization order are preserved from the recorded model.
Load trained weights as a `state_dict` with `strict=True`.
