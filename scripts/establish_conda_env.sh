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
    try_using_raven_environment
    `python $SCRIPT_DIR/TestHarness/testers/RavenUtils.py --conda-create`
  fi
}

try_using_raven_environment ()
{
  #First check home directory for a script, then /opt/raven_libs
  if test -e $HOME/.raven/environments/raven_libs_profile;
    then
  source $HOME/.raven/environments/raven_libs_profile
    else
  if test -e /opt/raven_libs/environments/raven_libs_profile;
  then
      source /opt/raven_libs/environments/raven_libs_profile
  fi
    fi
}

if [ "${#CONDA_EXE}" -gt 0 ];
then
  echo sourcing conda function definitions ...
  . "$(dirname $(dirname "${CONDA_EXE}"))/etc/profile.d/conda.sh"
fi

if command -v conda 2> /dev/null;
then
  echo conda located, checking version ...
  echo `conda -V`
  echo ... checking environments ...
  conda_install_or_create
else
  echo conda not initially located, checking for RAVEN conda ...
  try_using_raven_environment
  if which conda 2> /dev/null;
  then
    echo conda located, checking environments ...
    conda_install_or_create
  else
    echo No conda found!  Unable to ensure RAVEN environment and libraries exist.
  fi
fi
