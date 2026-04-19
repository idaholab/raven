# SparseSensing PR Handoff

This note captures the current PR split for the 2026 SparseSensing extension so
the follow-up does not depend on personal memory or GitHub comment history.

## PR Split

### PR #1
- Upstream PR: `idaholab/raven#2574`
- Branch: `1_Jimmy_extending_SPSL_transient_and_HOSVD`
- Scope:
  - `pivotParameter` support
  - `reshape=snapshot`
  - transient regression
  - parameter+time snapshot regression
  - initial unit-test bootstrap/helpers
  - minimal user-manual update for transient snapshot usage

### PR #2
- Stacked PR: `Jimmy-INL/raven#15`
- Branch: `2_Jimmy_sparse_sensing_spatiotemporal`
- Current base branch: `1_Jimmy_extending_SPSL_transient_and_HOSVD`
- Scope:
  - `reshape=spatiotemporal`
  - `HOSVDBasis` in `SparseSensingBases.py`
  - HOSVD basis wiring in `SparseSensing.py`
  - spatiotemporal regression
  - HOSVD regression
  - expanded unit coverage for new branches and error paths
  - follow-up manual updates for `spatiotemporal` and `HOSVD`

## Retarget Plan

PR #2 is intentionally stacked so its diff only shows work beyond PR #1.

After PR #1 merges:
- rebase `2_Jimmy_sparse_sensing_spatiotemporal` onto `devel`
- force-push the rebased branch
- retarget the PR from `Jimmy-INL:1_Jimmy_extending_SPSL_transient_and_HOSVD`
  to `idaholab/raven:devel`

## Commands

Focused regressions:

```bash
./run_tests --re='HOSVD|Spatiotemporal|ParamTime|Transient'
```

Focused unit tests:

```bash
/Users/abdomg/miniconda3/envs/raven_libraries/bin/python -m pytest \
  tests/framework/unit_tests/PostProcessors -v
```

Coverage fallback used in this environment:

```bash
/Users/abdomg/miniconda3/envs/raven_libraries/bin/python -m coverage run -m pytest \
  tests/framework/unit_tests/PostProcessors/test_sparse_sensing_helpers.py -v
/Users/abdomg/miniconda3/envs/raven_libraries/bin/python -m coverage report -m \
  --include='ravenframework/Models/PostProcessors/SparseSensing.py,ravenframework/Models/PostProcessors/SparseSensingBases.py'
```

## Notes

- `pytest-cov` was not available in the active `raven_libraries` environment, so
  plain `coverage.py` was used instead.
- `doc/user_manual` build was attempted with the RAVEN environment Python on
  `PATH`; the build still exits nonzero because of broader pre-existing LaTeX
  issues elsewhere in the manual, not because of the SparseSensing changes.
