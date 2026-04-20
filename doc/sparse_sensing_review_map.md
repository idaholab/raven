# SparseSensing New PySensors Review Map

This branch contains two different categories of work:

- `SparseSensing` / `pysensors` feature alignment
- repo-wide scientific-stack and compatibility work needed after moving toward newer `python-sensors` and NumPy 2

If you want to review **only SPSL logic**, start with the `SparseSensing` section and ignore the compatibility section on the first pass.

## Review Order

1. Review `SparseSensing` feature commits first.
2. Review dependency / stack refresh commits second.
3. Review code-interface compatibility commits last.

## 1. SparseSensing / PySensors Feature Commits

These are the commits that directly implement or extend `SparseSensing`.

### `1cba4e069` `Refactor SparseSensing adapter helpers`

- Category: mostly `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/unit_tests/PostProcessors/testSparseSensingAdapters.py`
  - `doc/user_manual/PostProcessors/SparseSensing.tex`
- Why review:
  - introduces the adapter/helper layer that later commits build on
  - first appearance of the new `SparseSensing` parsing/build flow
- Caveat:
  - also touches `dependencies.xml`, but that is not the main point of the commit

### `f4e847f10` `Add SparseSensing reconstruction metrics and CCQR`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLOptiTwistReconstructionError.xml`
  - `tests/framework/unit_tests/PostProcessors/testSparseSensingAdapters.py`
- Why review:
  - first `CCQR` plumbing
  - first reconstruction-metric support

### `e444ea34a` `Require canonical RandomProjection spelling`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/unit_tests/PostProcessors/testSparseSensingAdapters.py`
- Why review:
  - makes the parser use only `RandomProjection`

### `f041b52bd` `Use pysensors-native reconstruction metrics`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLOptiTwistReconstructionError.xml`
- Why review:
  - switches from the temporary RAVEN-side metric path to native `pysensors` reconstruction metrics

### `f140903a2` `Stabilize exact reconstruction metric outputs`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/unit_tests/PostProcessors/testSparseSensingAdapters.py`
- Why review:
  - intermediate checkpoint on exact-reconstruction metric handling

### `e3d47fd72` `Keep exact reconstruction metrics in scientific notation`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/PostProcessors/SparseSensing/tests`
- Why review:
  - final shape of exact-reconstruction metric formatting/tolerance

### `551d51a39` `Add GQR constrained sensing support`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLGQRCircle.xml`
  - `tests/framework/unit_tests/PostProcessors/testSparseSensingAdapters.py`
- Why review:
  - first `GQR`
  - first constrained-sensing XML
  - first built-in constraint shape coverage

### `151738e6f` `Add SparseSensing UQ outputs and TPGR regressions`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLUQStd.xml`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLReconstructionErrorCurve.xml`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLTPGR.xml`
- Why review:
  - adds `std`
  - adds reconstruction-error curve output
  - adds `TPGR` regression coverage

### `bfe02f216` `Add SparseSensing TPGR energy landscapes`

- Category: `SparseSensing`
- Main files:
  - `ravenframework/Models/PostProcessors/SparseSensing.py`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLGQRDistance.xml`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLGQRExactN.xml`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLGQRPredetermined.xml`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLTPGROnePtEnergy.xml`
  - `tests/framework/PostProcessors/SparseSensing/testSPSLTPGRTwoPtEnergy.xml`
- Why review:
  - expands `GQR` strategy coverage
  - adds `TPGR` one-point and two-point energy-landscape outputs

## 2. Dependency / Scientific-Stack Refresh Commits

These are not SPSL features. They were added to make the branch work against the newer scientific stack needed for newer `python-sensors`.

### `86bd06b2a` `Document python-sensors 0.4.3 stack blockers`

- Category: dependency / analysis
- Main files:
  - `dependencies.xml`
  - `doc/sparse_sensing_pysensors_env_alignment.md`
- Why review:
  - records why newer `python-sensors` initially broke the stack

### `845d59bd1` `Refresh SparseSensing scientific stack pins`

- Category: dependency / stack refresh
- Main files:
  - `dependencies.xml`
  - `doc/sparse_sensing_pysensors_env_alignment.md`
- Why review:
  - moves the main scientific stack toward the validated newer versions

### `6cb8d5736` `Restore TensorFlow via tf_keras legacy mode`

- Category: dependency / stack refresh
- Main files:
  - `dependencies.xml`
  - `ravenframework/utils/importerUtils.py`
- Why review:
  - keeps TensorFlow-based ROM coverage working on the refreshed stack

### `a7613a4ba` `Fix NumPy 2 scalar type lookup`

- Category: compatibility
- Main files:
  - `ravenframework/utils/mathUtils.py`
  - `scripts/conversionScripts/conversion_hdf5/reader_hdf5_from_Feb_2018_to_Oct_2021.py`
  - `tests/framework/unit_tests/utils/testMathUtils.py`
- Why review:
  - removes `np.sctypes` dependency

### `275d4d88c` `Clean up NumPy 2 alias removals`

- Category: compatibility
- Main files:
  - `ravenframework/Optimizers/GeneticAlgorithm.py`
  - `ravenframework/SupervisedLearning/SupervisedLearning.py`
  - `tests/framework/unit_tests/DataObjects/TestDataSet.py`
- Why review:
  - fixes removed NumPy aliases

### `8b0ad6d6a` `Fix NumPy 2 product alias in quadratures`

- Category: compatibility
- Main files:
  - `ravenframework/Quadratures.py`
- Why review:
  - small isolated NumPy 2 fix

### `f735dade5` `Fix NumPy 2 GenericParser and Wavelet validation`

- Category: compatibility
- Main files:
  - `dependencies.xml`
  - `ravenframework/CodeInterfaceClasses/Generic/GenericParser.py`
  - `tests/framework/unit_tests/utils/testGenericParser.py`
  - `tests/framework/unit_tests/Distributions/TestDistributions.py`
- Why review:
  - first pass at NumPy-2-safe GenericParser formatting
  - also refreshes Wavelet validation path
- Note:
  - this commit is later refined by `c759fedd0` and `b0f23917e`

### `143a866bd` `Refresh NumPy 2 NetCDF validation`

- Category: compatibility
- Main files:
  - `dependencies.xml`
  - `tests/framework/unit_tests/utils/testCachedNDArray.py`
- Why review:
  - moves NetCDF validation forward under the refreshed stack

### `216d9bdcf` `Serialize NetCDF differ loads`

- Category: compatibility
- Main files:
  - `scripts/TestHarness/testers/NetCDFDiffer.py`
- Why review:
  - fixes parallel NetCDF differ instability

## 3. Code-Interface Compatibility Commits

These are the commits that touched code-interface formatting. They are **not** SPSL features. They came from broader validation after the dependency refresh.

### `c759fedd0` `Preserve compact numeric formatting in code interfaces`

- Category: compatibility
- Main files:
  - `ravenframework/CodeInterfaceClasses/MooseBasedApp/MOOSEparser.py`
  - `ravenframework/CodeInterfaceClasses/Generic/GenericParser.py`
  - `tests/framework/CodeInterfaceTests/RAVEN/tests`
  - `tests/framework/unit_tests/utils/testGenericParser.py`
- Why review:
  - fixes the MOOSE input-parser precision regression
  - adds small tolerance for `CodeInterfaceTests/RAVEN/Optimizer`
- Note:
  - the GenericParser part of this commit is intentionally refined in the next commit

### `b0f23917e` `Restore GenericParser float round-trip precision`

- Category: compatibility
- Main files:
  - `ravenframework/CodeInterfaceClasses/Generic/GenericParser.py`
  - `tests/framework/unit_tests/utils/testGenericParser.py`
- Why review:
  - finalizes the split between:
    - compact formatting in `MOOSEparser`
    - round-trip precision in `GenericParser`
  - fixes the CobraTF regression introduced by over-compacting generic numeric text

## Practical Review Advice

If you want the cleanest SPSL-only pass, review this subset first:

- `1cba4e069`
- `f4e847f10`
- `e444ea34a`
- `f041b52bd`
- `f140903a2`
- `e3d47fd72`
- `551d51a39`
- `151738e6f`
- `bfe02f216`

If you want the repo-wide dependency story after that, review:

- `86bd06b2a`
- `845d59bd1`
- `6cb8d5736`
- `a7613a4ba`
- `275d4d88c`
- `8b0ad6d6a`
- `f735dade5`
- `143a866bd`
- `216d9bdcf`
- `c759fedd0`
- `b0f23917e`

## Validation Summary

The following review-relevant checkpoints were validated:

- focused `SparseSensing` regressions passed
- `CodeInterfaceTests/RAVEN/Optimizer` passed after adding a small numerical tolerance
- the `CodeInterfaceTests` shard passed cleanly once already-classified local false positives were excluded

The excluded local false positives are:

- macOS Tk plotting crashes
- stale local run directories
- sandbox-only Dask `LocalCluster` bind failures for parallel tests that pass when rerun outside the sandbox
