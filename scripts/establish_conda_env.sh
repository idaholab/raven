#!/bin/bash

# USES:
# --load - just finds and sources the conda environment (default)
# --install - find (create if not found) and update the environment, then load it
# OTHER OPTIONS:
# --optional - if updating, install optional libraries as well as base ones

# ENVIRONMENT VARIABLES
# location of conda definitions: CONDA_DEFS (defaults if not set based on OS)
# name for raven libraries: RAVEN_LIBS_NAME (defaults to raven_libraries if not set)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_UTILS=${SCRIPT_DIR}/TestHarness/testers/RavenUtils.py

function establish_OS ()
{
	case $OSTYPE in
		"linux")
			OSOPTION="--linux"
			;;
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
			echo Unknown OS: $OSTYPE
			OSOPTION=""
			;;
	esac
}

function find_conda_defs ()
{
	if [ -z ${CONDA_DEFS} ];
	then
		# default location of conda definitions, windows is unsurprisingly an exception
		if [[ "$OSOPTION" = "--windows" ]];
		then
			export CONDA_DEFS="/c/ProgramData/Miniconda2/etc/profile.d/conda.sh";
		else
			export CONDA_DEFS="$HOME/miniconda2/etc/profile.d/conda.sh";
		fi
	fi
}

function install_libraries()
{
  if [[ $ECE_VERBOSE ]]; then echo Installing libraries ...; fi
  # XXX need to set environment name when sent to RAVEN_UTILS !!
  COMMAND=`echo $(python ${RAVEN_UTILS} --conda-install ${INSTALL_OPTIONAL} ${OSOPTION})`
  echo conda command: ${COMMAND}
  ${COMMAND}
}

function create_libraries()
{
  if [[ $ECE_VERBOSE ]]; then echo Installing libraries ...; fi
  # XXX need to set environment name when sent to RAVEN_UTILS !!
  COMMAND=`echo $(python ${RAVEN_UTILS} --conda-create ${INSTALL_OPTIONAL} ${OSOPTION})`
  echo conda command: ${COMMAND}
  ${COMMAND}
}

function display_usage()
{
	echo ''
	echo '  ------------------------------------------'
	echo '  Default usage:'
	echo '    establish_conda_env.sh'
	echo ''
	echo '  Description:'
	echo '      This loads the RAVEN conda environment specified in \$RAVEN_LIBS_NAME \(default raven_libraries\).'
	echo '      This script is also used for installing these libraries\; see options below.'
	echo '  ------------------------------------------'
	echo ''
	echo '  Options:'
	echo '    --conda-defs'
	echo '      Defines location of conda definitions (often miniconda2/etc/profile.d/conda.sh). If not provided, guesses based on OS.'
	echo ''
	echo '    --help'
	echo '      Displays this text and exits'
	echo ''
	echo '    --install'
	echo '      Installs current python library versions for this release of RAVEN using conda'
	echo ''
	echo '    --load'
	echo '      Attempts to activate RAVEN conda environment without installation'
	echo ''
	echo '    --no-clean'
	echo '      Prevents deletion of old environment before installation.  Requires --install.'
	echo ''
	echo '    --optional'
	echo '      Additionally installs optional libraries used in some RAVEN workflows.  Requires --install.'
	echo ''
	echo '    --quiet'
	echo '      Runs script with minimal output'
	echo ''
}

# main

# set default operation
ECE_MODE=1 # 1 for loading, 2 for install, 0 for help
INSTALL_OPTIONAL="" # --optional if installing optional, otherwise blank
ECE_VERBOSE=0 # 0 for printing, anything else for no printing
ECE_CLEAN=0 # 0 for yes (remove raven libs env before installing), 1 for don't remove it

# parse command-line arguments
while test $# -gt 0
do
  case "$1" in
    --help)
      display_usage
      exit 0
      ;;
    --load)
      ECE_MODE=1
      ;;
    --install)
      ECE_MODE=2
      ;;
    --optional)
      echo Including optional libraries ...
      INSTALL_OPTIONAL="--optional"
      ;;
    --quiet)
      ECE_VERBOSE=1
      ;;
    --conda-defs)
      shift
      export CONDA_DEFS=$1
      ;;
    --no-clean)
      ECE_CLEAN=1
      ;;
  esac
  shift
done

if [[ $ECE_VERBOSE == 0 ]];
then
  echo Run Options:
  echo "  Mode: $ECE_MODE"
  echo "  Verbosity: $ECE_VERBOSE"
  echo "  Clean: $ECE_CLEAN"
  echo "  Conda Defs: $CONDA_DEFS"
  if [[ $ECE_MODE == 1 ]];
  then
    echo 'Loading RAVEN libraries ...'
  elif [[ $ECE_MODE == 2 ]];
  then
    echo 'Installing RAVEN libraries ...'
  fi
fi

# determine operating system
establish_OS

# set raven libraries environment name, if not set
if [ -z $RAVEN_LIBS_NAME ];
then
  export RAVEN_LIBS_NAME=raven_libraries
fi
if [[ $ECE_VERBOSE == 0 ]]; then echo RAVEN conda environment is named \"${RAVEN_LIBS_NAME}\"; fi

# establish conda function definitions (conda 4.4+ ONLY)
find_conda_defs
if test -e ${CONDA_DEFS};
then
	if [[ $ECE_VERBOSE == 0 ]]; then echo Found conda definitions at ${CONDA_DEFS}; fi
  source ${CONDA_DEFS}
else
  echo Conda definitions not found at \"${CONDA_DEFS}\"!
  echo Specify the location of miniconda2/etc/profile.d/conda.sh through the --conda-defs option.
  exit 1
fi

# debug output conda version
if [[ $ECE_VERBOSE == 0 ]]; then echo `conda -V`; fi

# find RAVEN libraries environment
if conda env list | grep ${RAVEN_LIBS_NAME} 2> /dev/null;
then
  if [[ $ECE_VERBOSE == 0 ]]; then echo Found library environment ...; fi
  LIBS_EXIST=0
else
  if [[ $ECE_VERBOSE == 0 ]]; then echo Did not find library environment ...; fi
  LIBS_EXIST=1
fi

# take action depending on mode
## load only
if [[ $ECE_MODE == 1 ]];
then
  if [[ ! $LIBS_EXIST == 0 ]];
  then
    echo conda environment ${RAVEN_LIBS_NAME} not found!
    echo Please run "raven/establish_conda_env.sh" with argument "--install".
    exit 1
  fi
fi

## install mode
if [[ $ECE_MODE == 2 ]];
then
  # if libraries already exist, depends on if in "clean" mode or not
  if [[ $LIBS_EXIST == 0 ]];
  then
    # if libs exist and clean mode, scrub them
    if [[ $ECE_CLEAN == 0 ]];
    then
      if [[ $ECE_VERBOSE == 0 ]]; then echo Removing old environment ...; fi
      conda deactivate
      conda remove -n ${RAVEN_LIBS_NAME} --all -y
      create_libraries
    # if libs exist, but not clean mode, install
    else
      install_libraries
    fi
  # if libraries don't exist, create them
  else
    create_libraries
  fi
fi

# by here, libraries exist and have been created, so activate them
if [[ $ECE_VERBOSE == 0 ]]; then echo Activating environment ...; fi
conda activate ${RAVEN_LIBS_NAME}
if [[ $ECE_VERBOSE == 0 ]]; then echo  ... done!; fi


