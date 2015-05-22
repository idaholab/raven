#!/bin/bash
#BUILD_DIR=${BUILD_DIR:=$HOME/raven_libs/build}
#INSTALL_DIR=${INSTALL_DIR:=$HOME/raven_libs/pylibs}
PYTHON_CMD=${PYTHON_CMD:=python}
JOBS=${JOBS:=1}

SCRIPT_DIRNAME=`dirname $0`
SCRIPT_DIR=`(cd $SCRIPT_DIRNAME; pwd)`
ORIGPYTHONPATH="$PYTHONPATH"

#update_python_path ()
#{
#    if ls -d $INSTALL_DIR/*/python*/site-packages/
#    then
#        export PYTHONPATH=`ls -d $INSTALL_DIR/*/python*/site-packages/`:"$ORIGPYTHONPATH"
#    fi
#}

#update_python_path
#PATH=$INSTALL_DIR/bin:$PATH


cd tests/framework

for I in $(python ${SCRIPT_DIR}/developer_tools/get_coverage_tests.py)
do
    echo Validating $I
    xmllint --noout --schema ../../raven.xsd $I
done

