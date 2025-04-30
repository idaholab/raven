#!/bin/bash
#run this with sbatch bitterroot_slurm.sh -d /projects/MFIX/run2 -b 40
#SBATCH --time=8:00:00 # walltime
#SBATCH --ntasks=200 # total number of processor cores to use

##SBATCH -p hbm
#SBATCH --wckey=ne_gen # Project Code
#SBATCH -J "ml_training" # job name
#SBATCH -e slurm-%j.err # %j will be replaced by the SLURM_JOB_ID
#SBATCH -o slurm-%j.out # %j will be replaced by the SLURM_JOB_ID
export DASK_CONFIG="dask_config.yaml"
export DASK_LOGGING__DISTRIBUTED=warning
echo number of nodes $SLURM_NNODES
echo number of cores $SLURM_NTASKS
echo Number of cores/node $SLURM_CPUS_ON_NODE
echo Number of cores per task. $SLURM_CPUS_PER_TASK
echo "Node allocation successful"

module load openmpi

# There is a local venv file i made with python --version Python 3.10.14 which i ran python -m venv .venv to make
source /projects/MFIX/.venv/bin/activate

# Launch the Dask cluster using MPI
dask scheduler --scheduler-file /projects/MFIX/scheduler.json --interface ib0 --dashboard-address :8787 &

sleep 5

mpirun -np 190 dask-mpi --scheduler-file /projects/MFIX/scheduler.json --worker-class distributed.Worker \
 --interface ib0 --no-scheduler --memory-limit 8GB &
# mpirun -np 290 dask-mpi --scheduler-file /projects/MFIX/congds_postprocessing/scheduler.json --worker-class distributed.Nanny --interface ib0 --local-directory /tmp &

# srun --mpi=pmix_v5 --ntasks 290 --scheduler-file /projects/MFIX/congds_postprocessing/scheduler.txt --worker-class distributed.Nanny  &

DASK_MPI_PID=$!
echo $DASK_MPI_PID
# Give the Dask cluster some time to start up (optional, adjust timing if needed)
sleep 5
echo "Running MFiX_Postprocessing_dask.py on node $(hostname)"
# Run the Python script
/projects/MFIX/.venv/bin/python MFiX_Postprocessing_dask.py "$@"

rm -f /projects/MFIX/scheduler.json