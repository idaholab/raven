#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    echo Moving to working dir: ${PBS_O_WORKDIR}
    cd $PBS_O_WORKDIR
fi

module load pbs mvapich2/2.3.3-gcc-5.4.0 git conda use.moose
source /apps/local/anaconda/3.7/etc/profile.d/conda.sh
conda activate raven_libraries

# conda definitions should be set up in raven/.ravenrc after installing with raven/scripts/establish_conda_env.sh
# module load raven-devel
# module load MVAPICH2/2.0.1-GCC-4.9.2
##  also the name of the raven libraries conda environment
conda activate raven_libraries

which python
which mpiexec
echo ''
echo $COMMAND
$COMMAND
