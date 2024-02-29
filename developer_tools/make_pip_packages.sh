#!/bin/bash

# This script builds 3.9 and 3.8 wheels and puts them in the dist directory
# It will create python39_pip and python38_pip conda environments
# and use them for building pip packages.
# It requires that .ravenrc has a working CONDA_DEFS statement.
# To run from the raven directory:
# ./developer_tools/make_pip_packages.sh
# this can be run on windows, mac and linux, and then the wheels for
# each operating system can be collected, tested and uploaded to pypi
# The main pip build instructions are in the setup.py file.


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_DIR=`dirname $SCRIPT_DIR`

source $RAVEN_DIR/scripts/read_ravenrc.sh
CONDA_DEFS=$(read_ravenrc "CONDA_DEFS")
source ${CONDA_DEFS}

conda env remove --name python39_pip
conda create -y --name python39_pip python=3.9 swig

conda env remove --name python310_pip
conda create -y --name python310_pip python=3.10 swig

cd $RAVEN_DIR

rm -f setup.cfg
python ./scripts/library_handler.py pip --action=setup.cfg > setup.cfg

conda activate python39_pip
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build

conda activate python310_pip
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build

