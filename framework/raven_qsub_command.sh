#!/bin/bash

if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

which python
which mpiexec
echo $COMMAND
$COMMAND
