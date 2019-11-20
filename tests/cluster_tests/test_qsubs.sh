#!/bin/bash

#This script is used on the INL cluster machine Falcon to test
# the cluster interface.

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
        cat $RAVEN_FRAMEWORK_DIR/test_qsub.e* || echo No *.e* file found! Continuing ...
        printf '\n\nStandard Output:\n'
        cat $RAVEN_FRAMEWORK_DIR/test_qsub.o* || echo No *.o* file found! Continuing ...
    fi
    rm $RAVEN_FRAMEWORK_DIR/test_qsub.[eo]* || echo Trying to remove *.o*, *.e* files but not found. Continuing ...
    echo ''

}

echo Current directory: `pwd`

cd ${RAVEN_FRAMEWORK_DIR}/..
./run_tests --only-run-types="qsub" --re=cluster_tests
EXIT=$?

## Find output files and print them
#for FILE in `find . -name 'test_qsub.*' -print; find . -name 'testHybrid-*' -print; find . -name '*_server_out.log' -print`; do
#    echo FILE $FILE
#    cat $FILE | tr -d '\0'
#    echo
#done

exit $EXIT
