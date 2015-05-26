#!/bin/bash
#BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
#INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}

SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
ORIGPYTHONPATH="$PYTHONPATH"


cd tests/framework

failed_tests=0
passed_tests=0

for I in $(python ${SCRIPT_DIR}/developer_tools/get_coverage_tests.py)
do
    echo Validating $I
    xmllint --noout --schema ../../raven.xsd $I
    if test $? -eq 0;
    then
        passed_tests=$(($passed_tests + 1))
    else
        failed_tests=$(($failed_tests + 1))
    fi
done
echo Passed $passed_tests Failed $failed_tests
exit $failed_tests


