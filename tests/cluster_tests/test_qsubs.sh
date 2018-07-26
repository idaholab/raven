#!/bin/bash

#This script is used on the INL cluster machine Falcon to test
# the cluster interface.

num_fails=0
fails=''

pushd ../../framework
RAVEN_FRAMEWORK_DIR=$(pwd)
source ../scripts/establish_conda_env.sh --load
popd

wait_lines ()
{
    echo Return code: $?
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
    if test $lines -ne $COUNT; then
        echo Lines not here yet, waiting even longer.
        sleep 60 #Sleep in case this is just disk propagation
        lines=`ls $LS_DATA | wc -l`
    fi
    if test $lines -eq $COUNT; then
        echo PASS $NAME
    else
        echo FAIL $NAME
        fails=$fails', '$NAME
        num_fails=$(($num_fails+1))
        printf '\n\nStandard Error:\n'
        cat $RAVEN_FRAMEWORK_DIR/test_qsub.e* || echo No *.e* file found! Continuing ...
        printf '\n\nStandard Output:\n'
        cat $RAVEN_FRAMEWORK_DIR/test_qsub.o* || echo No *.o* file found! Continuing ...
    fi
    rm $RAVEN_FRAMEWORK_DIR/test_qsub.[eo]* || echo Trying to remove *.o*, *.e* files but not found. Continuing ...
    echo ''

}

echo Current directory: `pwd`

echo Removing old databases...
rm -Rf DatabaseStorage/

rm -Rf FirstMQRun/
#REQUIREMENT_TEST R-IS-7
../../raven_framework test_mpiqsub_local.xml pbspro_mpi.xml cluster_runinfo_legacy.xml
wait_lines 'FirstMQRun/[1-6]/*test.csv' 6 mpiqsub
echo ''

rm -Rf FirstMNRun/
../../raven_framework test_mpiqsub_nosplit.xml cluster_runinfo_legacy.xml
wait_lines 'FirstMNRun/[1-6]/*.csv' 6 mpiqsub_nosplit
echo ''

rm -Rf FirstMLRun/
../../raven_framework test_mpiqsub_limitnode.xml cluster_runinfo_legacy.xml
wait_lines 'FirstMLRun/[1-6]/*.csv' 6 mpiqsub_limitnode
echo ''

rm -Rf FirstMRun/

echo ''
echo 'Running interactive MPI test ...'
qsub -P moose -l select=6:ncpus=4:mpiprocs=1 -l walltime=10:00:00 -l place=free -W block=true ./run_mpi_test.sh
wait_lines 'FirstMRun/[1-6]/*test.csv' 6 mpi
echo ''

rm -Rf FirstPRun/
../../raven_framework test_pbs.xml cluster_runinfo_legacy.xml
wait_lines 'FirstPRun/[1-6]/*test.csv' 6 pbsdsh
echo ''

rm -Rf FirstMFRun/
../../raven_framework test_mpiqsub_flex.xml cluster_runinfo_legacy.xml
wait_lines 'FirstMFRun/[1-6]/*.csv' 6 mpiqsub_flex
echo ''

rm -Rf FirstMForcedRun/
../../raven_framework test_mpiqsub_forced.xml cluster_runinfo_legacy.xml
wait_lines 'FirstMForcedRun/[1-6]/*.csv' 6 mpiqsub_forced
echo ''

######################################
# test parallel for internal Objects #
######################################
# first stes (external model in parallel)
cd InternalParallel/
rm -Rf InternalParallelExtModel/*.csv
#REQUIREMENT_TEST R-IS-8
../../../raven_framework test_internal_parallel_extModel.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'InternalParallelExtModel/*.csv' 28 paralExtModel
cd ..
echo ''

# second test (ROM in parallel)
cd InternalParallel/
rm -Rf InternalParallelScikit/*.csv
#REQUIREMENT_TEST R-IS-9
../../../raven_framework test_internal_parallel_ROM_scikit.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'InternalParallelScikit/*.csv' 2 paralROM
cd ..
echo ''

# third test (PostProcessor in parallel)
cd InternalParallel/
rm -Rf InternalParallelPostProcessorLS/*.csv
../../../raven_framework test_internal_parallel_PP_LS.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'InternalParallelPostProcessorLS/*.csv' 4 parallelPP
cd ..
echo ''

# forth test (Topology Picard in parallel)
cd InternalParallel/
rm -Rf InternalParallelMSR/*.csv
../../../raven_framework test_internal_MSR.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'InternalParallelMSR/*.csv' 1 parallelMSR
cd ..
echo ''

# fifth test (Ensamble Model Picard in parallel)
cd InternalParallel/
rm -Rf metaModelNonLinearParallel/*.png
../../../raven_framework test_ensemble_model_picard_parallel.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'metaModelNonLinearParallel/*.png' 3 parallelEnsemblePicard
cd ..
echo ''

# sixth test (Ensamble Model Picard in parallel)
cd InternalParallel/
rm -Rf metaModelLinearParallel/*.png
../../../raven_framework test_ensemble_model_linear_internal_parallel.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'metaModelLinearParallel/*.png' 2 parallelEnsembleLinear
cd ..
echo ''

# seven test (HybridModel Code in parallel)
cd InternalParallel/
rm -Rf hybridModelCode/*.csv
../../../raven_framework test_hybrid_model_code.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'hybridModelCode/*.csv' 1 parallelHybridModelCode
cd ..
echo ''

# eighth test (HybridModel External Model in parallel)
cd InternalParallel/
rm -Rf hybridModelExternal/*.csv
../../../raven_framework test_hybrid_model_external.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'hybridModelExternal/*.csv' 1 parallelHybridModelExternal
cd ..
echo ''

############################################
# test parallel for internal Objects ENDED #
############################################

################################
# other parallel objects tests #
################################

# Adaptive Sobol
cd AdaptiveSobol/
rm -Rf workdir/*
../../../raven_framework test_adapt_sobol_parallel.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'workdir/*.csv' 1 adaptiveSobol
cd ..
echo ''

# Raven-Running-Raven (RAVEN code interface)
cd RavenRunsRaven/raven_running_raven_internal_models/
rm -Rf FirstMRun DatabaseStorage *csv testPointSet_dump.xml
cd ..
../../../raven_framework test_raven_running_raven_int_models.xml ../pbspro_mpi.xml ../cluster_runinfo_legacy.xml
wait_lines 'raven_running_raven_internal_models/testP*.csv' 17 ravenRunningRaven
cd ..
echo ''


if test $num_fails -eq 0; then
    echo ALL PASSED
else
    echo FAILED: $num_fails $fails
fi
exit $num_fails
