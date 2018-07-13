#!/bin/bash

cd $PBS_O_WORKDIR/

module load raven-devel
module load MVAPICH2/2.0.1-GCC-4.9.2
source activate raven_libraries

python  ../../framework/Driver.py test_mpi.xml
