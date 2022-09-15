#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_DIR=`dirname $SCRIPT_DIR`

source $RAVEN_DIR/scripts/read_ravenrc.sh
CONDA_DEFS=$(read_ravenrc "CONDA_DEFS")
source ${CONDA_DEFS}

conda env remove --name python37_pip
conda create -y --name python37_pip python=3.7 swig

conda env remove --name python38_pip
conda create -y --name python38_pip python=3.8 swig

cd $RAVEN_DIR

rm -f setup.cfg
python ./scripts/library_handler.py pip --action=setup.cfg > setup.cfg

conda activate python37_pip
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build

conda activate python38_pip
command -v python
python -m ensurepip
python -m pip install --upgrade build
python -m build

