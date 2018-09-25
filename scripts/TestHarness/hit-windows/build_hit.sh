#!/bin/bash

echo $MOOSE_DIR

if [ -d "../../../moose/framework/contrib" ]; then
  HIT_DIR=../../../moose/framework/contrib/hit
  MOOSE_PYTHON_DIR=../../../moose/python
else
  HIT_DIR=../../../../moose/framework/contrib/hit
  MOOSE_PYTHON_DIR=../../../../moose/python
fi

#HIT_DIR=../../../moose/framework/contrib/hit
#MOOSE_PYTHON_DIR=../../../moose/python

if [[ $VS90COMNTOOLS ]]
then
    cp setup.py $HIT_DIR
    pushd $HIT_DIR
    python setup.py build_ext --inplace --compiler=msvc
    EXIT_CODE=$?
    popd
    cp ${HIT_DIR}/hit.pyd $MOOSE_PYTHON_DIR
    touch ${MOOSE_PYTHON_DIR}/hit.so
else
    make -C $HIT_DIR bindings
    EXIT_CODE=$?
    cp ${HIT_DIR}/hit.so $MOOSE_PYTHON_DIR
fi
echo ls -l $MOOSE_PYTHON_DIR
ls -l $MOOSE_PYTHON_DIR
exit $EXIT_CODE
