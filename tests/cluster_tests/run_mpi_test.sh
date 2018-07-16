#!/bin/bash

echo 'DEBUGG chdir'
cd $PBS_O_WORKDIR/

echo 'DEBUGG module load 1'
module load raven-devel
echo 'DEBUGG module load 2'
module load MVAPICH2/2.0.1-GCC-4.9.2
echo 'DEBUGG library load'
source ../scripts/establish_conda_env.sh --load

echo 'DEBUGG run'
python  ../../framework/Driver.py test_mpi.xml
echo 'DEBUGG done'
