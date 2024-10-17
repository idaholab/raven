#!/bin/bash

if test -n "$SLURM_SUBMIT_DIR"; then
    echo Moving to working dir: ${SLURM_SUBMIT_DIR}
    cd $SLURM_SUBMIT_DIR
fi

source /etc/profile.d/modules.sh
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
