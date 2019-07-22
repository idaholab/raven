#!/bin/bash

cd $PBS_O_WORKDIR/

module load raven-devel

../../raven_framework test_mpi.xml
