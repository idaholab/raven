#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

# set up conda definitions location (for the establish_conda_env.sh script)
export CONDA_DEFS="/apps/local/miniconda2/4.5.4/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH="~/.conda/falcon/envs"
# note that $RAVEN_LIBS_NAME should be set here as well ...
# TODO does this run qsubbed commands with the provided env name correctly?

which python
which mpiexec
echo $COMMAND
$COMMAND
