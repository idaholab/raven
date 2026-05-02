# SparseSensing Tutorials

Executable notebooks for the staged SparseSensing tutorial set. The first three
notebooks cover the transient and snapshot-reshape work from PR #1; notebooks
04 and 05 cover the PR #2 additions for `spatiotemporal` scheduling and the
current `HOSVDBasis` implementation.

| # | Notebook | Data | Focus |
|---|----------|------|-------|
| 01 | `01_TwistPrototype.ipynb` | OPTI-TWIST thermal snapshots | Steady-state reconstruction anchor |
| 02 | `02_Transient_singleTrajectory.ipynb` | Synthetic separable transient | Single-trajectory time dependence |
| 03 | `03_ParameterAndTime_snapshot.ipynb` | Synthetic parameter-and-time tensor | Snapshot reshape and rank sweep |
| 04 | `04_Spatiotemporal_schedule.ipynb` | Synthetic parameter-and-time tensor | Spatiotemporal reshape and schedule decoding |
| 05 | `05_HOSVD_vs_SVD.ipynb` | Constructed separable tensor | Current HOSVD basis implementation compared to flat snapshot SVD |

The notebooks are intended to be committed with executed outputs so they render on GitHub.

To re-run locally from this directory:

```bash
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 01_TwistPrototype.ipynb
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 02_Transient_singleTrajectory.ipynb
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 03_ParameterAndTime_snapshot.ipynb
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 04_Spatiotemporal_schedule.ipynb
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 05_HOSVD_vs_SVD.ipynb
```

Notebook 05 is intentionally conservative in its claims: the current
`HOSVDBasis` extracts the spatial factor from the tensor's spatial unfolding, so
on the snapshot-stacked matrix used by `SparseSensing` it serves as a
consistency check against flat SVD rather than a claim of universal superiority.
