# file_path=/scratch/kimj5/run_4varPerb_constTemp_CGP3_RAVEN-InterfaceTest/MFiX_Sampling_Submission_Postprocess_Debugging/Run/$2/MFIX_RAVEN_Temp.mfx
# file_path=/projects/MFIX/Run_newVarPerturb/run_4varPerb_constTemp_CGP3_RAVEN-InterfaceTest/MFiX_Sampling_Submission_Postprocess_Debugging/Run/$2/MFIX_RAVEN_Temp.mfx
# file_path=$(python3 -c "import config; print(config.file_path.format('$2'))")


module load openmpi
module load mfix

echo "1: $1"
echo "2: $2"
# echo "3: $SLURM_CPUS_PER_TASK"

# NOTE: For debugging, $SLURM_CPUS_PER_TASK cannot be used, since you are already in the interactive node with submitted CPU cores.
# mpirun --bind-to none --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np 28 /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $file_path
mpirun --bind-to none --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np 28 /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $1
