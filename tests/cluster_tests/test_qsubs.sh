#!/bin/bash

num_fails=0

rm -Rf FirstMQRun/

python ../../framework/Driver.py test_mpiqsub_local.xml

sleep 1 #Wait for disk to propagate.
lines=`ls FirstMQRun/*eqn.csv | wc -l`

if test $lines -eq 6; then
    echo PASS mpiqsub
else
    echo FAIL mpiqsub
    num_fails=$(($num_fails+1))
fi

rm -Rf FirstMRun/

qsub -l select=6:ncpus=4:mpiprocs=1 -l walltime=10:00:00 -l place=free -W block=true ./run_mpi_test.sh

sleep 1 #Wait for disk to propagate.
mlines=`ls FirstMRun/*eqn.csv | wc -l`

if test $mlines -eq 6; then
    echo PASS mpi
else
    echo FAIL mpi
    num_fails=$(($num_fails+1))
fi

rm -Rf FirstPRun/

python ../../framework/Driver.py test_pbs.xml

sleep 1 #Wait for disk to propagate.
plines=`ls FirstPRun/*eqn.csv | wc -l`

if test $plines -eq 6; then
    echo PASS pbsdsh
else
    echo FAIL pbsdsh
    num_fails=$(($num_fails+1))
fi


if test $num_fails -eq 0; then
    echo ALL PASSED
else
    echo FAILED: $num_fails
fi
exit $num_fails
