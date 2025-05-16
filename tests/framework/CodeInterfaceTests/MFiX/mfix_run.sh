#!/bin/bash
file_path=$RAVEN_FRAMEWORK_DIR/../tests/framework/CodeInterfaceTests/MFiX/MFiX_Sampling_Submission_Postprocess/Run/$2/MFIX_RAVEN_Temp.mfx

module load openmpi
module load mfix

mpirun --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np $SLURM_CPUS_PER_TASK /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $file_path