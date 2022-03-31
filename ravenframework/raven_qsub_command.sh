#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    echo Moving to working dir: ${PBS_O_WORKDIR}
    cd $PBS_O_WORKDIR
fi

# conda definitions should be set up in raven/.ravenrc after installing with raven/scripts/establish_conda_env.sh
module load raven-devel
module load MVAPICH2/2.0.1-GCC-4.9.2
module load imkl/10.3.6.233
##  also the name of the raven libraries conda environment
source activate raven_libraries

which python
which mpiexec
echo ''
echo $COMMAND
$COMMAND
