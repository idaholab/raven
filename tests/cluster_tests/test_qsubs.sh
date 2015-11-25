#!/bin/bash

num_fails=0

wait_lines ()
{
    LS_DATA="$1"
    COUNT="$2"
    NAME="$3"
    sleep 2
    lines=`ls $LS_DATA | wc -l`
    if test $lines -ne $COUNT; then
        echo Lines not here yet, waiting longer.
        sleep 20 #Sleep in case this is just disk propagation
        lines=`ls $LS_DATA | wc -l`
    fi
    if test $lines -eq $COUNT; then
        echo PASS $NAME
    else
        echo FAIL $NAME
        num_fails=$(($num_fails+1))
    fi

}

rm -Rf FirstMQRun/

python ../../framework/Driver.py test_mpiqsub_local.xml

wait_lines 'FirstMQRun/*eqn.csv' 6 mpiqsub

rm -Rf FirstMRun/

qsub -l select=6:ncpus=4:mpiprocs=1 -l walltime=10:00:00 -l place=free -W block=true ./run_mpi_test.sh

wait_lines 'FirstMRun/*eqn.csv' 6 mpi

rm -Rf FirstPRun/

python ../../framework/Driver.py test_pbs.xml

wait_lines 'FirstPRun/*eqn.csv' 6 pbsdsh

######################################
# test parallel for internal Objects #
######################################
# first stes (external model in parallel)
cd InternalParallel/
rm -Rf InternalParallelExtModel/*.csv

python ../../../framework/Driver.py test_internal_parallel_extModel.xml

wait_lines 'InternalParallelExtModel/*.csv' 28 paralExtModel

cd ..

# second test (ROM in parallel)
cd InternalParallel/
rm -Rf InternalParallelScikit/*.csv

python ../../../framework/Driver.py test_internal_parallel_ROM_scikit.xml

wait_lines 'InternalParallelScikit/*.csv' 2 paralROM

cd ..

# third test (PostProcessor in parallel)
cd InternalParallel/
rm -Rf InternalParallelPostProcessorLS/*.csv

python ../../../framework/Driver.py test_internal_parallel_PP_LS.xml

wait_lines 'InternalParallelPostProcessorLS/*.csv' 6 paralROM

cd ..

cd InternalParallel/
rm -Rf InternalParallelMSR/*.csv

python ../../../framework/Driver.py test_internal_MSR.xml

wait_lines 'InternalParallelMSR/*.csv' 1 parallelMSR

cd ..


############################################
# test parallel for internal Objects ENDED #
############################################

################################
# other parallel objects tests #
################################

cd AdaptiveSobol/
rm -Rf workdir/*

python ../../../framework/Driver.py test_adapt_sobol_parallel.xml

wait_lines 'workdir/*.csv' 1 adaptiveSobol

cd ..


if test $num_fails -eq 0; then
    echo ALL PASSED
else
    echo FAILED: $num_fails
fi
exit $num_fails
