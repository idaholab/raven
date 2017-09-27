#!/bin/bash

#This is a generic test of the cluster features.  Depending on the cluster
# it is used on, $HOME/.raven/environments/raven_libs_profile
# and $HOME/.raven/default_runinfo.xml may need to be created or modified
# to setup the correct environment and provide required cluster parameters.

num_fails=0
fails=''

pushd ../../framework
RAVEN_FRAMEWORK_DIR=$(pwd)
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
        cat $RAVEN_FRAMEWORK_DIR/test_qsub.e*
        printf '\n\nStandard Output:\n'
        cat $RAVEN_FRAMEWORK_DIR/test_qsub.o*
    fi
    rm $RAVEN_FRAMEWORK_DIR/test_qsub.[eo]*

}

rm -Rf FirstMQRun/

#REQUIREMENT_TEST R-IS-7
python ../../framework/Driver.py test_mpiqsub_local.xml pbspro_mpi.xml cluster_runinfo.xml

wait_lines 'FirstMQRun/[1-6]/*test.csv' 6 mpiqsub

rm -Rf FirstMNRun/

python ../../framework/Driver.py test_mpiqsub_nosplit.xml cluster_runinfo.xml

wait_lines 'FirstMNRun/[1-6]/*.csv' 6 mpiqsub_nosplit

rm -Rf FirstMLRun/

python ../../framework/Driver.py test_mpiqsub_limitnode.xml cluster_runinfo.xml

wait_lines 'FirstMLRun/[1-6]/*.csv' 6 mpiqsub_limitnode

rm -Rf FirstMRun/

qsub -P moose -l select=6:ncpus=4:mpiprocs=1 -l walltime=10:00:00 -l place=free -W block=true ./run_mpi_test.sh

wait_lines 'FirstMRun/[1-6]/*test.csv' 6 mpi

rm -Rf FirstPRun/

python ../../framework/Driver.py test_pbs.xml cluster_runinfo.xml

wait_lines 'FirstPRun/[1-6]/*test.csv' 6 pbsdsh

rm -Rf FirstMFRun/

python ../../framework/Driver.py test_mpiqsub_flex.xml cluster_runinfo.xml

wait_lines 'FirstMFRun/[1-6]/*.csv' 6 mpiqsub_flex

rm -Rf FirstMForcedRun/

python ../../framework/Driver.py test_mpiqsub_forced.xml cluster_runinfo.xml

wait_lines 'FirstMForcedRun/[1-6]/*.csv' 6 mpiqsub_forced

######################################
# test parallel for internal Objects #
######################################
# first stes (external model in parallel)
cd InternalParallel/
rm -Rf InternalParallelExtModel/*.csv

#REQUIREMENT_TEST R-IS-8
python ../../../framework/Driver.py test_internal_parallel_extModel.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'InternalParallelExtModel/*.csv' 28 paralExtModel

cd ..

# second test (ROM in parallel)
cd InternalParallel/
rm -Rf InternalParallelScikit/*.csv

#REQUIREMENT_TEST R-IS-9
python ../../../framework/Driver.py test_internal_parallel_ROM_scikit.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'InternalParallelScikit/*.csv' 2 paralROM

cd ..

# third test (PostProcessor in parallel)
cd InternalParallel/
rm -Rf InternalParallelPostProcessorLS/*.csv

python ../../../framework/Driver.py test_internal_parallel_PP_LS.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'InternalParallelPostProcessorLS/*.csv' 6 parallelPP

cd ..

# forth test (Topology Picard in parallel)

cd InternalParallel/
rm -Rf InternalParallelMSR/*.csv

python ../../../framework/Driver.py test_internal_MSR.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'InternalParallelMSR/*.csv' 1 parallelMSR

cd ..

# fifth test (Ensamble Model Picard in parallel)
cd InternalParallel/
rm -Rf metaModelNonLinearParallel/*.png

python ../../../framework/Driver.py test_ensemble_model_picard_parallel.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'metaModelNonLinearParallel/*.png' 3 parallelEnsemblePicard

cd ..

# sixth test (Ensamble Model Linear Picard in parallel)
cd InternalParallel/
rm -Rf metaModelLinearParallel/*.png

python ../../../framework/Driver.py test_ensemble_model_linear_internal_parallel.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'metaModelLinearParallel/*.png' 2 parallelEnsembleLinear

cd ..

# seven test (HybridModel Code in parallel)
cd InternalParallel/
rm -Rf hybridModelCode/*.csv

python ../../../framework/Driver.py test_hybrid_model_code.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'hybridModelCode/*.csv' 1 parallelHybridModelCode

cd ..

# eighth test (HybridModel External Model in parallel)
cd InternalParallel/
rm -Rf hybridModelExternal/*.csv

python ../../../framework/Driver.py test_hybrid_model_external.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'hybridModelExternal/*.csv' 1 parallelHybridModelExternal

cd ..

############################################
# test parallel for internal Objects ENDED #
############################################

################################
# other parallel objects tests #
################################

cd AdaptiveSobol/
rm -Rf workdir/*

python ../../../framework/Driver.py test_adapt_sobol_parallel.xml ../pbspro_mpi.xml ../cluster_runinfo.xml

wait_lines 'workdir/*.csv' 1 adaptiveSobol

cd ..


if test $num_fails -eq 0; then
    echo ALL PASSED
else
    echo FAILED: $num_fails $fails
fi
exit $num_fails
