#!/bin/bash

echo Establishing RAVEN conda environment ...

#Similar to setup_raven_libs.sh, but installs libraries, not just opens environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONDA_DEFS="$HOME/miniconda2/etc/profile.d/conda.sh"

# if runner desires optional test libraries (like PIL) to be installed,
#   they can request it by passing --optional to this script
if [ "$1" == "--optional" ];
then
  echo ... including optional testing libraries ...
  INSTALL_OPTIONAL="--optional"
else
  echo ... only including required libraries ...
  INSTALL_OPTIONAL=" "
fi

conda_install_or_create ()
{
  if conda env list | grep -q raven_libraries;
  then
    echo ... RAVEN environment located, checking packages ...
    `python $SCRIPT_DIR/TestHarness/testers/RavenUtils.py --conda-install ${INSTALL_OPTIONAL}`
  else
    echo ... No RAVEN environment located, creating it ...
    try_using_raven_environment
    `python $SCRIPT_DIR/TestHarness/testers/RavenUtils.py --conda-create ${INSTALL_OPTIONAL}`
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

# try activating conda base once so that $CONDA_EXE gets filled in.
## See discussion https://github.com/conda/conda/issues/7126
## attempt to make sure conda is around and understood.
### THIS decision tree is relevant to conda 4.5.4; supposedly this will all be fixed in conda 4.6.
# if "conda" is a recognized command ...
echo Attempting to locate \"conda\" ...
if command -v conda 2> /dev/null;
then
  # if the executable environment already set up ...
  if  [ "${#CONDA_EXE}" -gt 0 ];
  then
    echo ... conda located at ${CONDA_EXE}
  # if not, then this can be helped out by activating any environment
  else
    echo ... conda available but exec variable not set, so initializing base library ...
    conda activate base
    echo ... conda established.
  fi
else
  # conda is not found, so try locating it in the default place
  echo \"conda\" has not been found, checking for definitions in ${CONDA_DEFS} ...
  if test -e ${CONDA_DEFS};
  then
    echo ... found!  Sourcing definitions ...
    source ${CONDA_DEFS}
    echo ... activating base environment ...
    conda activate base
    echo ... conda established.
  else
    echo Unable to locate \"conda\"!
    exit 404
  fi
fi

# if conda version 4.4+, they use function definitions instead of path inclusion.
## however, they also define the CONDA_EXE variable, which is available.
## once loaded, command -v conda works for both versions of conda.
if [ "${#CONDA_EXE}" -gt 0 ];
then
  echo sourcing conda function definitions ...
  source "$(dirname $(dirname "${CONDA_EXE}"))/etc/profile.d/conda.sh"
fi


if command -v conda 2> /dev/null;
then
  echo conda located, checking version ...
  echo `conda -V`
  echo ... checking environments ...
  conda_install_or_create
else
  echo conda not initially located, checking for definitions at ${CONDA_DEFS} ...
  if test -e ${CONDA_DEFS}
  then
    source ${CONDA_DEFS}
  else
    echo conda not initially located, checking for RAVEN conda ...
    try_using_raven_environment
  fi
  if which conda 2> /dev/null;
  then
    echo conda located, checking environments ...
    conda_install_or_create
  else
    echo No conda found!  Unable to ensure RAVEN environment and libraries exist.
  fi
fi
