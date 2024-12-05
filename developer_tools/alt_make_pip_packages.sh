#!/bin/bash

# This script builds 3.9 and 3.10 wheels and puts them in the dist directory
# It will create pip39_venv and pip310_venv conda environments
# and use them for building pip packages.
# It requires that there are working swig, python3.9 and python3.10 commands.
# if brew is installed, on mac this can be done by:
# brew install swig python@3.9 python-tk@3.9 python@3.10 python-tk@3.10
# To run from the raven directory:
# ./developer_tools/alt_make_pip_packages.sh
# this has been tested on mac and also probably works on linux
# The main pip build instructions are in the setup.py file.


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_DIR=`dirname $SCRIPT_DIR`

cd $RAVEN_DIR

rm -f setup.cfg
python3.9 ./scripts/library_handler.py pip --action=setup.cfg > setup.cfg

rm -Rf pip39_venv/
python3.9 -m venv pip39_venv
source pip39_venv/bin/activate
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build
deactivate

rm -Rf pip310_venv/
python3.10 -m venv pip310_venv
source pip310_venv/bin/activate
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build
deactivate
