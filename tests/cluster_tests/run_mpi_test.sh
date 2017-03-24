#!/bin/bash

cd $PBS_O_WORKDIR/

module load raven-devel-gcc

python  ../../framework/Driver.py test_mpi.xml
