#!/bin/bash

# This script builds 3.12, 3.11 and 3.10 wheels and puts them in the dist directory
# It will create pip312_venv, pip311_venv and pip310_venv virtual environments
# and use them for building pip packages.
# It requires that there are working swig, python3.12, python3.11 and python3.10 commands.
# if brew is installed, on mac this can be done by:
# brew install swig python@3.12 python-tk@3.12 python@3.11 python-tk@3.11 python@3.10 python-tk@3.10
# To run from the raven directory:
# ./developer_tools/alt_make_pip_packages.sh
# this has been tested on mac and also probably works on linux
# The main pip build instructions are in the setup.py file.


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_DIR=`dirname $SCRIPT_DIR`

cd $RAVEN_DIR

rm -f setup.cfg
python3.10 ./scripts/library_handler.py pip --action=setup.cfg > setup.cfg

rm -Rf pip310_venv/
python3.10 -m venv pip310_venv
source pip310_venv/bin/activate
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build
deactivate

rm -Rf pip311_venv/
python3.11 -m venv pip311_venv
source pip311_venv/bin/activate
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build
deactivate

rm -Rf pip312_venv/
python3.12 -m venv pip312_venv
source pip312_venv/bin/activate
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build
deactivate
