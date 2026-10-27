#!/bin/bash

# This script builds 3.12, 3.11 and 3.10 wheels and puts them in the dist directory
# It will create python312_pip, python311_pip and python310_pip conda environments
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

conda env remove -y --name python310_pip || echo python310_pip not removed
conda create -y --name python310_pip python=3.10 swig

conda env remove -y --name python311_pip || echo python311_pip not removed
conda create -y --name python311_pip python=3.11 swig

conda env remove -y --name python312_pip || echo python312_pip not removed
conda create -y --name python312_pip python=3.12 swig

cd $RAVEN_DIR

rm -f setup.cfg
python ./scripts/library_handler.py pip --action=setup.cfg > setup.cfg

conda activate python310_pip
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build

conda activate python311_pip
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build

conda activate python312_pip
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build
