#!/bin/bash

HIT_DIR=../../../moose/framework/contrib/hit
MOOSE_PYTHON_DIR=../../../moose/python
if [[ $VSVARSALL ]]
then
    SETUPTOOLS_COMPILER_FLAG='--compiler=msvc'
fi
cp setup.py $HIT_DIR
pushd $HIT_DIR
python setup.py build_ext --inplace $SETUPTOOLS_COMPILER_FLAG
popd
if [[ -f ${HIT_DIR}/hit.so ]]
then
    cp ${HIT_DIR}/hit.so $MOOSE_PYTHON_DIR
elif [[ -f ${HIT_DIR}/hit.pyd ]]
then
    cp ${HIT_DIR}/hit.pyd $MOOSE_PYTHON_DIR
    touch ${MOOSE_PYTHON_DIR}/hit.so
fi
