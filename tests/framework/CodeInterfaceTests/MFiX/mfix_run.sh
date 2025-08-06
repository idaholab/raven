#!/bin/bash
# file_path=$RAVEN_FRAMEWORK_DIR/../tests/framework/CodeInterfaceTests/MFiX/MFiX_Sampling_Submission_Postprocess/Run/$2/MFIX_RAVEN_Temp.mfx

module load openmpi
module load mfix

# echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
# echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
# echo "1: $1"
# echo "2: $2"
# TotalThreads=$((SLURM_JOB_NUM_NODES*SLURM_CPUS_PER_TASK))
# echo "TotalThreads: $TotalThreads"

# This works for $SLURM_CPUS_PER_TASK
mpirun --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np $SLURM_CPUS_PER_TASK /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $1

# mpirun --bind-to none --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np $SLURM_CPUS_PER_TASK /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $1

# This is a trial for $TotalThreads
# mpirun --bind-to none --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np $TotalThreads /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $1