#!/bin/bash

date

if test -n "$PBS_O_WORKDIR"; then
    echo Moving to working dir: ${PBS_O_WORKDIR}
    cd $PBS_O_WORKDIR
fi

if test -f /etc/profile.d/modules.sh;
then
    source /etc/profile.d/modules.sh
elif test -f /apps/local/lmod/modules.sh;
then
    source /apps/local/lmod/modules.sh;
fi
echo RAVEN_FRAMEWORK_DIR $RAVEN_FRAMEWORK_DIR

if test -e $RAVEN_FRAMEWORK_DIR/../scripts/establish_conda_env.sh; then
    source $RAVEN_FRAMEWORK_DIR/../scripts/establish_conda_env.sh --load
else
    echo RAVEN_FRAMEWORK_DIR ERROR
    echo FILE $RAVEN_FRAMEWORK_DIR/../scripts/establish_conda_env.sh
    echo NOT FOUND
fi
module load openmpi

which python
which mpiexec
echo ''
echo $COMMAND
$COMMAND
