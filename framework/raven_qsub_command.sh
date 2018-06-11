#!/bin/bash

## FIXME why change to raven_framework?  Stay in the working dir.
#if test -n "$PBS_O_WORKDIR"; then
#    cd $PBS_O_WORKDIR
#fi

which python
which mpiexec
echo $COMMAND
$COMMAND
