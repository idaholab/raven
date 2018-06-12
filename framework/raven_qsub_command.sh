#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    echo Moving to working dir: ${PBS_O_WORKDIR}
    cd $PBS_O_WORKDIR
fi

source activate raven_libraries

echo 'python:'
which python
echo ''
echo 'mpiexec:'
which mpiexec
echo ''
echo $COMMAND
$COMMAND
