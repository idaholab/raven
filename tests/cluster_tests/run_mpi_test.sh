#!/bin/bash

cd $PBS_O_WORKDIR/

module load raven-devel
module load MVAPICH2/2.0.1-GCC-4.9.2

../../raven_framework test_mpi.xml
