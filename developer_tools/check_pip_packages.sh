#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_DIR=`dirname $SCRIPT_DIR`

source $RAVEN_DIR/scripts/read_ravenrc.sh
CONDA_DEFS=$(read_ravenrc "CONDA_DEFS")
source ${CONDA_DEFS}

cd $RAVEN_DIR

ls -l dist

#The following works to install from the dist directory, but also can install
# from pypi so doesn't fail if not created in dist, so can't be used
#python -m pip install -f file://${RAVEN_DIR}/dist raven_framework || exit -1

echo
echo Checking Python 3.7

conda activate python37_pip
python -m pip uninstall -y raven_framework || echo not installed
python -m pip install dist/raven_framework*cp37*.whl || exit -1


echo
echo Checking Python 3.8

conda activate python38_pip
python -m pip uninstall -y raven_framework || echo not installed
python -m pip install dist/raven_framework*cp38*.whl || exit -1
