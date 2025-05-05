#!/bin/bash
# file_path=/projects/MFIX/Run_newVarPerturb/run_4varPerb_constTemp_CGP3_RAVEN-InterfaceTest/MFiX_Sampling_Submission_Postprocess/Run/$2/MFIX_RAVEN_Temp.mfx
file_path=/home/kimj5/projects/bitterroot/raven/tests/framework/CodeInterfaceTests/MFiX/MFiX_Sampling_Submission_Postprocess/Run/$2/MFIX_RAVEN_Temp.mfx

module load openmpi
module load mfix

# mpirun --bind-to none --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np $SLURM_CPUS_PER_TASK /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $file_path
mpirun --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np $SLURM_CPUS_PER_TASK /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $file_path