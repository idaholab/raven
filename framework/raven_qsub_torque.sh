#!/bin/bash


module load mpich

BASE_DIR=/cm/shared/apps/ncsu/projects/MOOSE/raven_libs
INSTALL_DIR="$BASE_DIR/install"
VE_DIR="$BASE_DIR/ve"
export PATH="$INSTALL_DIR/bin:$PATH"
source "$VE_DIR/bin/activate"

echo $PYTHONPATH

if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

which python
which mpiexec
echo $COMMAND
$COMMAND
