#!/bin/bash

rm -Rf ../inputs/mpi_driver_test/FirstMRun/


JOB=`python Driver.py ../inputs/mpi_driver_test/test_mpiqsub.xml | tail -n 1`

JOB_NUMBER=`echo $JOB|cut -d '.' -f 1`
echo $JOB_NUMBER

qsub -W depend=afterany:$JOB_NUMBER -W block=true script

ls ../inputs/mpi_driver_test/FirstMRun/*eqn.csv


