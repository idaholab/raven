#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

# conda definitions should be set up in raven/.ravenrc after installing with raven/scripts/establish_conda_env.sh
module load raven-devel
module load MVAPICH2/2.0.1-GCC-4.9.2
##  also the name of the raven libraries conda environment
source activate raven_libraries

echo `conda env list`
echo DEBUGG HERE IN RQC
conda list

which python
which mpiexec
echo $COMMAND
$COMMAND
