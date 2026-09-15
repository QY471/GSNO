# Important model support

These files are not standalone experiment entry points. They remain active dependencies of the important models kept in `model/`:

- circular density-normalized Gaussian renderer shared by the retained EDSR models;
- primitive/routing common code shared by E6 and LR-primitive transport;
- legacy ScatterGS implementation required by retained ScatterGS and ScaleConsistent models.

Do not archive these files without updating and testing their callers.
