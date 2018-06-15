#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

# set up conda definitions location (for the establish_conda_env.sh script)
export CONDA_DEFS="/apps/local/miniconda2/4.5.4/etc/profile.d/conda.sh"

which python
which mpiexec
echo $COMMAND
$COMMAND
