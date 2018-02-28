#!/bin/bash

HIT_DIR=../../../moose/framework/contrib/hit
MOOSE_PYTHON_DIR=../../../moose/python
if [[ $VSVARSALL ]]
then
    cp setup.py $HIT_DIR
    pushd $HIT_DIR
    python setup.py build_ext --inplace --compiler=msvc
    popd
    cp ${HIT_DIR}/hit.pyd $MOOSE_PYTHON_DIR
    touch ${MOOSE_PYTHON_DIR}/hit.so
else
    pushd $HIT_DIR
    make bindings
    popd
    cp ${HIT_DIR}/hit.so $MOOSE_PYTHON_DIR
fi
