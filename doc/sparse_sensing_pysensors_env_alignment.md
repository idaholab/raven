# SparseSensing New PySensors Environment Alignment

Date: 2026-04-19
Branch: `3_Jimmy_sparse_sensing_newPysensors_alignment`

## Summary

The RAVEN `SparseSensing` code on this branch has already been aligned to a much broader
`pysensors` feature surface:

- `CCQR`
- native reconstruction metrics
- `std`
- `reconstruction_error`
- `GQR`
- `TPGR`
- `TPGR` one-point and two-point energy landscapes

The remaining blocker is the package stack, not the RAVEN-side interface.

## What Was Verified

Current QA-compatible environment:

- env: `raven_libraries`
- `python-sensors 0.4.1`
- `numpy 1.26.4`
- `scipy 1.12.0`
- `matplotlib 3.6.3`
- `scikit-learn 1.1.3`
- `pandas 2.3.3`

Isolated clone created for upgrade testing:

- env: `raven_spsl_043`
- cloned from `raven_libraries`

Upgrade attempted in the isolated env:

```bash
/Users/abdomg/miniconda3/envs/raven_spsl_043/bin/python -m pip install --upgrade python-sensors==0.4.3
```

Observed upgraded packages:

- `python-sensors 0.4.3`
- `numpy 2.4.4`
- `scipy 1.17.1`
- `matplotlib 3.10.8`
- `scikit-learn` remained `1.1.3`

Immediate failure after upgrade:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

This occurred when importing `sklearn`, which RAVEN reaches during normal module import.

Additional dependency conflicts reported by pip during the upgrade:

- `numba 0.61.2` requires `numpy < 2.3`
- `statsforecast 2.0.3` requires `scipy < 1.16.0`
- `tensorflow-macos 2.14.1` requires `numpy < 2.0.0`

## Conclusion

Adopting `python-sensors 0.4.3` in the current RAVEN QA environment is not a
single-package change. It requires a coordinated refresh of at least:

- `scikit-learn` to a build compatible with NumPy 2.x
- `numba`
- `statsforecast`
- `tensorflow-macos` or any TensorFlow-dependent workflow

## Immediate Repo Action Taken

`dependencies.xml` was pinned to:

```xml
<python-sensors source="pip">0.4.1</python-sensors>
```

This prevents accidental drift of the QA environment while the coordinated stack
refresh is planned separately.

## Recommended Next Step

Create a dedicated stack-refresh branch or environment update effort that:

1. chooses a coherent NumPy 2.x-compatible scientific stack,
2. rebuilds or upgrades `scikit-learn` accordingly,
3. resolves `numba` / `statsforecast` / `tensorflow-macos` compatibility,
4. reruns the `SparseSensing` branch tests in that refreshed environment.
