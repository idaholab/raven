#!/bin/bash

# This script builds 3.13, 3.12 and 3.11 wheels and puts them in the dist directory
# It will create pip313_venv, pip312_venv and pip311_venv virtual environments
# and use them for building pip packages.
# It requires that there are working swig, python3.13, python3.12 and python3.11 commands.
# if brew is installed, on mac this can be done by:
# brew install swig python@3.13 python-tk@3.13 python@3.12 python-tk@3.12 python@3.11 python-tk@3.11
# To run from the raven directory:
# ./developer_tools/alt_make_pip_packages.sh
# this has been tested on mac and also probably works on linux
# The main pip build instructions are in the setup.py file.


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_DIR=`dirname $SCRIPT_DIR`

cd $RAVEN_DIR

rm -f setup.cfg
python3.11 ./scripts/library_handler.py pip --action=setup.cfg > setup.cfg

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

rm -Rf pip313_venv/
python3.13 -m venv pip313_venv
source pip313_venv/bin/activate
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build
deactivate
