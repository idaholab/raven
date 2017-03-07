#!/bin/bash

cd $PBS_O_WORKDIR/

BASE_DIR=/cm/shared/apps/ncsu/projects/MOOSE/raven_libs
INSTALL_DIR="$BASE_DIR/install"
VE_DIR="$BASE_DIR/ve"
export PATH="$INSTALL_DIR/bin:$PATH"
source "$VE_DIR/bin/activate"

module load mpich

python  ../../framework/Driver.py test_mpi.xml
