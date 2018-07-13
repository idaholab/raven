#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

# conda definitions should be set up in raven/.ravenrc after installing with raven/scripts/establish_conda_env.sh
module load raven-devel
module load raven-devel-gcc
#module load mpi4py/1.3.1-gmvolf-5.5.4-Python-2.7.9
##  also the name of the raven libraries conda environment
source activate raven_libraries

echo `conda env list`
echo DEBUGG HERE IN RQC
conda list

which python
which mpiexec
echo $COMMAND
$COMMAND
