#!/bin/bash

echo Establishing RAVEN conda environment ...

#Similar to setup_raven_libs.sh, but installs libraries, not just opens environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# determine operating system
case $OSTYPE in
  "linux-gnu")
    OSOPTION="--linux"
    ;;
  "darwin"*)
    OSOPTION="--mac"
    ;;
  "msys"*)
    OSOPTION="--windows"
    ;;
  "cygwin"*)
    OSOPTION="--windows"
    ;;
  *)
    OSOPTION=""
    ;;
esac

# default location of conda definitions, windows is unsurprisingly an exception
if [[ "$OSOPTION" = "--windows" ]];
then
  CONDA_DEFS="/c/ProgramData/Miniconda2/etc/profile.d/conda.sh";
else
  CONDA_DEFS="$HOME/miniconda2/etc/profile.d/conda.sh";
fi


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
  # set up library environment
  if conda env list | grep raven_libraries;
  then
    echo ... RAVEN environment located, checking packages ...
    COMMAND=`echo $(python $SCRIPT_DIR/TestHarness/testers/RavenUtils.py --conda-install ${INSTALL_OPTIONAL} ${OSOPTION})`
  else
    echo ... No RAVEN environment located, creating it ...
    try_using_raven_environment
    COMMAND=`echo $(python $SCRIPT_DIR/TestHarness/testers/RavenUtils.py --conda-create ${INSTALL_OPTIONAL} ${OSOPTION})`
  fi
  echo ... ... flags: ${INSTALL_OPTIONS} ${OSOPTION}
  echo ... ... conda command: ${COMMAND}
  $COMMAND
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

# try loading conda definitions at default location
## TODO if CONDA_DEFS already set, don't re-set it!  Allows users to define location
echo Attempting to locate \"conda\" ...
if test -e ${CONDA_DEFS};
then
  echo Sourcing conda definitions at ${CONDA_DEFS}
  source ${CONDA_DEFS}
fi

# if "conda" is a recognized command ...
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
  echo conda not initially located, checking for RAVEN conda ...
  try_using_raven_environment
  if command -v conda 2> /dev/null;
  then
    echo ... conda located at ${CONDA_EXE}
  else
    echo Unable to locate \"conda\"!
    echo You can provide the location of \"conda.sh\" after conda is installed by setting a bash environment variable: CONDA_DEFS
    echo Example for Default Location:
    echo   export CONDA_DEFS=${CONA_DEFS}

    echo If conda is installed but the version is older than 4.4, it needs to be updated to run this script.
    exit 404
  fi
fi

# we already know conda is available, so check environments
#if command -v conda 2> /dev/null;
#then
echo conda located, checking version ...
echo `conda -V`
echo ... checking environments ...
conda_install_or_create
