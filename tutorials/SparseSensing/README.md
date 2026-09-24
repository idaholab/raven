# SparseSensing Tutorials

Executable notebooks for the first SparseSensing tutorial PR:

| # | Notebook | Data | Focus |
|---|----------|------|-------|
| 01 | `01_TwistPrototype.ipynb` | OPTI-TWIST thermal snapshots | Steady-state reconstruction anchor |
| 02 | `02_Transient_singleTrajectory.ipynb` | Synthetic separable transient | Single-trajectory time dependence |
| 03 | `03_ParameterAndTime_snapshot.ipynb` | Synthetic parameter-and-time tensor | Snapshot reshape and rank sweep |

The notebooks are intended to be committed with executed outputs so they render on GitHub.

To re-run locally from this directory:

```bash
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 01_TwistPrototype.ipynb
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 02_Transient_singleTrajectory.ipynb
/Users/abdomg/miniconda3/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=spsl_vibrant 03_ParameterAndTime_snapshot.ipynb
```
