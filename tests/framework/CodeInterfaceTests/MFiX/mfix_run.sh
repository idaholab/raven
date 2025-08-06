#!/bin/bash

module load openmpi
module load mfix

mpirun --oversubscribe -mca mpi_warn_on_fork 0 -mca mca_base_component_show_load_errors 0 -np $SLURM_CPUS_PER_TASK /apps/local/miniforge/23.3.1/envs/mfix-24.1.1/bin/mfixsolver_dmp -s -f $1