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

The package stack is now largely aligned for the refreshed SparseSensing path.
The remaining broad-suite blockers on this macOS machine are:

- Tk/matplotlib plotting initialization failures in GUI-backed tests
- any additional NumPy 2 compatibility issues that only appear outside the
  currently focused validation slices

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

## Candidate SparseSensing Refresh Env

The isolated `raven_spsl_043` environment was then repaired into a usable
SparseSensing validation target with:

- `python-sensors 0.4.3`
- `numpy 2.2.6`
- `scipy 1.15.3`
- `matplotlib 3.10.8`
- `scikit-learn 1.7.2`
- `statsmodels 0.14.5`

This package set is enough for the current SparseSensing import path and test
surface.

Validated in that environment:

- `tests/framework/unit_tests/PostProcessors/testSparseSensingAdapters.py`
- `tests/framework/unit_tests/TSA/testWavelet.py`
- `tests/framework/unit_tests/Distributions/TestDistributions.py`
- focused SparseSensing XML regressions with
  `RAVEN_IGNORE_VERSIONS=1 ./run_tests --skip-load-env --re='...'`

Observed result:

- 9/10 focused XML regressions passed unchanged
- the remaining case, `testSPSLOptiTwistReconstructionError`, differed only in
  machine-precision reconstruction-error residuals (`~1e-13` to `~1e-24`)
  under the newer numerical stack

Repo-side follow-up:

- `tests/framework/PostProcessors/SparseSensing/tests` now sets
  `zero_threshold = 1e-10` for
  `testSPSLOptiTwistReconstructionError`, so exact-reconstruction residual
  noise is treated as numerical zero during CSV comparison while preserving the
  scientific-notation outputs themselves.

Follow-on dependency work in the same env:

- upgraded `protobuf` into the TensorFlow 2.20-compatible range
- upgraded `opentelemetry-proto` so the new protobuf range no longer conflicts
- removed the stale `tensorflow-macos 2.14.1` / `tensorflow-estimator 2.14.0`
  leftovers from the cloned env
- upgraded `PyWavelets` to `1.8.0` so the TSA Wavelet unit test is compatible
  with the NumPy 2 stack
- upgraded `netCDF4` to `1.7.2` so the NetCDF database tests are compatible
  with the NumPy 2 stack
- `pip check` is now clean in `raven_spsl_043`

Targeted numerical/test follow-up:

- `tests/framework/unit_tests/Distributions/TestDistributions.py` now uses the
  exact computed Poisson CDF values when checking the inverse `ppf(cdf(k))`
  identity, avoiding a brittle jump-discontinuity failure caused by rounding a
  discrete CDF value
- `tests/framework/unit_tests/utils/testCachedNDArray.py` now checks the
  numeric content of the array repr instead of exact NumPy whitespace padding,
  which changed under NumPy 2

TensorFlow/Keras status:

- the clean compatibility path is:
  - `tensorflow 2.21.0`
  - `tf_keras 2.21.0`
  - `TF_USE_LEGACY_KERAS=1` before TensorFlow import
- with that setup, `tf.keras` resolves to `tf_keras.api._v2.keras`, which
  matches the current RAVEN Keras ROM implementation expectations
- validated TensorFlow ROM coverage in `raven_spsl_043`:
  - `tests/framework/ROM/tensorflow_keras/tf_cnn1d`
  - `tests/framework/ROM/tensorflow_keras/tf_mlpc`
  - `tests/framework/ROM/tensorflow_keras/tf_mlpr`
  - `tests/framework/ROM/tensorflow_keras/tf_lstm`
  - `tests/framework/ROM/tensorflow_keras/tf_lstm_regression`

Repo implication:

- the scientific stack and `python-sensors` can move forward now
- TensorFlow can move forward too, but only through `tf_keras` legacy mode for
  now
- after updating `dependencies.xml`, the focused SparseSensing regression set
  also passed in `raven_spsl_043` without `RAVEN_IGNORE_VERSIONS`

## Immediate Repo Action Taken

`dependencies.xml` was initially pinned back to:

```xml
<python-sensors source="pip">0.4.1</python-sensors>
```

This prevented accidental drift while the refresh was being validated.

After the follow-on refresh work, `dependencies.xml` was updated to the
validated default scientific stack:

```xml
<numpy source="pip">2.2</numpy>
<scipy source="pip">1.15</scipy>
<scikit-learn source="pip">1.7</scikit-learn>
<matplotlib source="pip">3.10</matplotlib>
<statsmodels source="pip">0.14</statsmodels>
<protobuf source="pip">6.33</protobuf>
<opentelemetry-proto source="pip">1.41</opentelemetry-proto>
<python-sensors source="pip">0.4.3</python-sensors>
```

TensorFlow was intentionally removed from the default dependency set for now.

That temporary removal has now been replaced with the validated legacy-Keras
compatibility path:

```xml
<tensorflow source="pip">2.21</tensorflow>
<tf-keras source="pip">2.21</tf-keras>
```

RAVEN sets `TF_USE_LEGACY_KERAS=1` automatically before lazy-importing
TensorFlow whenever `tf_keras` is available.

## Recommended Next Step

Create a later TensorFlow/Keras modernization effort that:

1. removes the dependency on `tf_keras` legacy mode,
2. updates the RAVEN Keras ROM interface to run directly on the modern
   TensorFlow/Keras packaging,
3. reruns `tests/framework/ROM/tensorflow_keras`,
4. simplifies the TensorFlow import hook once the legacy bridge is no longer
   needed.
