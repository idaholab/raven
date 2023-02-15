#!/bin/bash

# This script tests that the 3.9 and 3.8 wheels built in directory dist
# are installable. It will not work if there is more than one
# version of the wheel for a python version (including for other os's)
# It is designed to be run after (or similar):
# rm -Rf dist; ./developer_tools/make_pip_packages.sh
# It will install raven_framework in the python37_pip and python38_pip
# conda environments
# It requires that .ravenrc has a working CONDA_DEFS statement.
# To run from the raven directory:
# ./developer_tools/check_pip_packages.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_DIR=`dirname $SCRIPT_DIR`

source $RAVEN_DIR/scripts/read_ravenrc.sh
CONDA_DEFS=$(read_ravenrc "CONDA_DEFS")
source ${CONDA_DEFS}

cd $RAVEN_DIR

ls -l dist

#The following works to install from the dist directory, but also can install
# from pypi so doesn't fail if wheels not created in dist, so can't be used
# for automated testing
#python -m pip install -f file://${RAVEN_DIR}/dist raven_framework || exit -1

echo
echo Checking Python 3.8

conda activate python38_pip
python -m pip uninstall -y raven_framework || echo not installed
python -m pip install dist/raven_framework*cp38*.whl || exit -1


echo
echo Checking Python 3.9

conda activate python39_pip
python -m pip uninstall -y raven_framework || echo not installed
python -m pip install dist/raven_framework*cp39*.whl || exit -1
