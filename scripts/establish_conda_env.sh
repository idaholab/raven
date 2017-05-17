#!/bin/bash

#Similar to setup_raven_libs.sh, but installs libraries, not just opens environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

conda_install_or_create ()
{
  if conda env list | grep -q raven_libraries;
  then
    echo ... RAVEN environment located, checking packages ...
    `python $SCRIPT_DIR/TestHarness/testers/RavenUtils.py --conda-install`
  else
    echo ... No RAVEN environment located, creating it ...
    try_using_raven_conda
    `python $SCRIPT_DIR/TestHarness/testers/RavenUtils.py --conda-create`
  fi
}

try_using_raven_conda ()
{
  if test -e /opt/raven_libs/bin/conda;
  then
    export PATH="/opt/raven_libs/bin:$PATH"
    source activate raven_libraries
  fi
}

if which conda 2> /dev/null;
then
  echo conda located, checking environments ...
  conda_install_or_create
else
  echo conda not initially located, checking for RAVEN conda ...
  try_using_raven_conda
  if which conda 2> /dev/null;
  then
    echo conda located, checking environments ...
    conda_install_or_create
  else
    echo No conda found!  Unable to ensure RAVEN environment and libraries exist.
  fi
fi
