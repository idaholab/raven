#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    echo Moving to working dir: ${PBS_O_WORKDIR}
    cd $PBS_O_WORKDIR
fi

# job tracking
#export LMOD_DISABLE_SAME_NAME_AUTOSWAP="no"
#source /etc/profile.d/modules.sh
#module load use.projects utils

# conda definitions should be set up in raven/.ravenrc after installing with raven/scripts/establish_conda_env.sh
module load raven-devel
##  also the name of the raven libraries conda environment
source activate raven_libraries

which python
which mpiexec
which job_tracker.py
echo ''
echo $COMMAND
#python ~/scripts/job_tracker.py "$COMMAND"
$COMMAND
