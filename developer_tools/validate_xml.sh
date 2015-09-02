#!/bin/bash
#BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
#INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}
SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`

ORIGPYTHONPATH="$PYTHONPATH"

maxlen=$(($(tput cols) < 100 ? $(tput cols) : 100))

TEST_DIRS="${SCRIPT_DIR}/../tests/framework ${SCRIPT_DIR}/../tests/cluster_tests"

failed_tests=0
passed_tests=0

# convert the external xml input files to normal raven input files and then validate
CONVERSION_SCRIPT_DIR="${SCRIPT_DIR}/../scripts/conversionScripts"
$PYTHON_CMD $CONVERSION_SCRIPT_DIR/externalXMLNode.py ${SCRIPT_DIR}/../tests/framework/test_External_XML.xml
echo -e "\033[1;32m the following script: test_External_XML.xml has been converted for xsd validate"

for testdir in ${TEST_DIRS}
do
    echo -e "\033[1;32mValidating $testdir "
    cd $testdir
    for I in $(python ${SCRIPT_DIR}/get_coverage_tests.py)
    do
        echo -en "\033[0mValidating $I"
        startlen=$((11+${#I}))
        VALOUT=$(xmllint --noout --schema  ${SCRIPT_DIR}/XSDSchemas/raven.xsd $I 2>&1)
        if test $? -eq 0;
        then
            periodlength=$((maxlen - startlen - 11))
            printf '%0.s.' $(seq 1 $periodlength)
            echo -e "\033[1;32m validated!"
            passed_tests=$(($passed_tests + 1))
        else
            periodlength=$((maxlen - startlen - 8))
            printf '%0.s.' $(seq 1 $periodlength)
            echo -e "\033[1;31m FAILED!"
            echo -e "$VALOUT"
            failed_tests=$(($failed_tests + 1))
        fi
    done
    echo -e "\033[0m--------------------------------------------------"
done
echo -e "\033[1;32mPassed $passed_tests \033[1;31mFailed $failed_tests\033[0m"

mv ${SCRIPT_DIR}/../tests/framework/test_External_XML.xml.bak ${SCRIPT_DIR}/../tests/framework/test_External_XML.xml
echo -e "\033[1;32m test_External_XML.xml has been converted back!"
exit $failed_tests
