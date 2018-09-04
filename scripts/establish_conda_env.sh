#!/bin/bash

# USES:
# --load - just finds and sources the conda environment (default)
# --install - find (create if not found) and update the environment, then load it
# OTHER OPTIONS:
# --optional - if updating, install optional libraries as well as base ones

# ENVIRONMENT VARIABLES
# location of conda definitions: CONDA_DEFS (defaults if not set based on OS)
# name for raven libraries: RAVEN_LIBS_NAME (defaults to raven_libraries if not set)

ECE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RAVEN_UTILS=${ECE_SCRIPT_DIR}/TestHarness/testers/RavenUtils.py

# fail if ANYTHING this script fails (mostly, there are exceptions)
set -e

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
			echo Unknown OS: $OSTYPE\; ignoring.
			OSOPTION=""
			;;
	esac
}

function read_ravenrc ()
{
  # $1 should be the keyword we're looking for
  # returns keyword argument through echo
  ## note that "| xargs" trims leading and trailing whitespace
  local TARGET=`echo $1 | xargs`
  # location of the RC file
  local RCNAME="${ECE_SCRIPT_DIR}/../.ravenrc"
  # if the RC file exists, loop through it and read keyword arguments split by "="
  if [ -f "$RCNAME" ]; then
    while IFS='=' read -r KEY ARG || [[ -n "$keyarg" ]]; do
      # trim whitespace
      KEY=`echo $KEY | xargs`
      ARG=`echo $ARG | xargs`
      # check for key match
      if [ "$KEY" = "$TARGET" ]; then
        echo "$ARG"
        return 0
      fi
    done < ${RCNAME}
  fi
  # if not found, return empty
  echo ''
}

function find_conda_defs ()
{
	if [ -z ${CONDA_DEFS} ];
	then
    # first check the RAVEN RC file for the key
    CONDA_DEFS=$(read_ravenrc "CONDA_DEFS")
    # if not set in RC, then will be empty string; next try defaults
    if [[ ${#CONDA_DEFS} == 0 ]];
    then
      # default location of conda definitions, windows is unsurprisingly an exception
      if [[ "$OSOPTION" = "--windows" ]];
      then
        CONDA_DEFS="/c/ProgramData/Miniconda2/etc/profile.d/conda.sh";
      else
        CONDA_DEFS="$HOME/miniconda2/etc/profile.d/conda.sh";
      fi
    # if found in RC, just use that.
    else
      if [[ $ECE_VERBOSE == 0 ]];
      then
        echo ... found conda path in ravenrc: ${CONDA_DEFS}
        echo ... \>\> If this is not the desirable path, rerun with argument --conda-defs \[path\] or remove the entry from raven/.ravenrc file.
      fi
    fi
	fi

  # fix Windows backslashes to be forward, compatible with all *nix including mingw
  CONDA_DEFS="${CONDA_DEFS//\\//}"
}

function install_libraries()
{
  if [[ $ECE_VERBOSE == 0 ]]; then echo Installing libraries ...; fi
  local COMMAND=`echo $(python ${RAVEN_UTILS} --conda-install ${INSTALL_OPTIONAL} ${OSOPTION})`
  echo ... conda command: ${COMMAND}
  ${COMMAND}
  # conda-forge
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from conda-forge ...; fi
  local COMMAND=`echo $(python ${RAVEN_UTILS} --conda-forge --conda-install ${OSOPTION})`
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... conda-forge command: ${COMMAND}; fi
  ${COMMAND}
}

function create_libraries()
{
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries ...; fi
  local COMMAND=`echo $(python ${RAVEN_UTILS} --conda-create ${INSTALL_OPTIONAL} ${OSOPTION})`
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... conda command: ${COMMAND}; fi
  ${COMMAND}
  # conda-forge
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from conda-forge ...; fi
  local COMMAND=`echo $(python ${RAVEN_UTILS} --conda-forge --conda-install ${OSOPTION})`
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... conda-forge command: ${COMMAND}; fi
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

function activate_env()
{
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Activating environment ...; fi
  conda activate ${RAVEN_LIBS_NAME}
}

function set_install_settings()
{
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Setting install variables ...; fi
  local COMMAND="python $ECE_SCRIPT_DIR/update_install_data.py --write --conda-defs ${CONDA_DEFS} --RAVEN_LIBS_NAME ${RAVEN_LIBS_NAME}"
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... ${COMMAND}; fi
  ${COMMAND}
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
      return
      ;;
    --load)
      ECE_MODE=1
      ;;
    --install)
      ECE_MODE=2
      ;;
    --optional)
      echo ... Including optional libraries ...
      INSTALL_OPTIONAL="--optional"
      ;;
    --quiet)
      ECE_VERBOSE=1
      ;;
    --conda-defs)
      shift
      CONDA_DEFS=$1
      ;;
    --no-clean)
      ECE_CLEAN=1
      ;;
  esac
  shift
done

if [[ $ECE_VERBOSE == 0 ]];
then
  echo ... Run Options:
  echo ...    Mode: $ECE_MODE
  echo ...   Verbosity: $ECE_VERBOSE
  echo ...   Clean: $ECE_CLEAN
  echo ...   Conda Defs: $CONDA_DEFS
  if [[ $ECE_MODE == 1 ]];
  then
    echo ... Loading RAVEN libraries ...
  elif [[ $ECE_MODE == 2 ]];
  then
    echo ... Installing RAVEN libraries ...
  fi
fi

# determine operating system
establish_OS
if [[ $ECE_VERBOSE == 0 ]]; then echo ... Detected OS as ${OSOPTION} ...; fi

# set raven libraries environment name, if not set
if [ -z $RAVEN_LIBS_NAME ];
then
  # check the RC file first
  RAVEN_LIBS_NAME=$(read_ravenrc "RAVEN_LIBS_NAME")
  # if not found through the RC file, will be empty string, so default to raven_libraries
  if [[ ${#RAVEN_LIBS_NAME} == 0 ]];
  then
    RAVEN_LIBS_NAME=raven_libraries
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... \"${RAVEN_LIBS_NAME}\" not found in global variables or ravenrc, defaulting to ${RAVEN_LIBS_NAME}; fi
  # verbosity to print library name findings in RC file
  else
    if [[ $ECE_VERBOSE == 0 ]];
    then
      echo ... \$RAVEN_LIBS_NAME set through raven/.ravenrc to ${RAVEN_LIBS_NAME}
      echo ... \>\> If this is not desired, then remove it from the ravenrc file before running.
    fi
  fi
else
  if [[ $ECE_VERBOSE == 0 ]];
  then
    echo ... RAVEN library name set through \$RAVEN_LIBS_NAME global variable.
    echo ... \>\> If this is not desired, then unset the variable before running.
  fi
fi
if [[ $ECE_VERBOSE == 0 ]]; then echo ... \>\> RAVEN conda environment is named \"${RAVEN_LIBS_NAME}\"; fi

# establish conda function definitions (conda 4.4+ ONLY, 4.3 and before not supported)
find_conda_defs
if test -e ${CONDA_DEFS};
then
	if [[ $ECE_VERBOSE == 0 ]]; then echo ... Found conda definitions at ${CONDA_DEFS}; fi
  source ${CONDA_DEFS}
else
  echo ... Conda definitions not found at \"${CONDA_DEFS}\"!
  echo ... \>\> Specify the location of miniconda2/etc/profile.d/conda.sh through the --conda-defs option.
  return 1
fi

# debug output conda version
if [[ $ECE_VERBOSE == 0 ]]; then echo `conda -V`; fi

# find RAVEN libraries environment
if conda env list | grep ${RAVEN_LIBS_NAME};
then
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Found library environment ...; fi
  LIBS_EXIST=0
else
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Did not find library environment ...; fi
  LIBS_EXIST=1
fi

# take action depending on mode
## load only
if [[ $ECE_MODE == 1 ]];
then
  # as long as library env exists, activate it
  if [[ $LIBS_EXIST == 0 ]];
  then
    activate_env
  # if it doesn't exist, make some noise.
  else
    echo conda environment ${RAVEN_LIBS_NAME} not found!
    echo Please run "raven/establish_conda_env.sh" with argument "--install".
    return 1
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
      if [[ $ECE_VERBOSE == 0 ]]; then echo ... Removing old environment ...; fi
      conda deactivate
      conda remove -n ${RAVEN_LIBS_NAME} --all -y
      create_libraries
    # if libs exist, but not clean mode, install;
    else
      install_libraries
    fi
  # if libraries don't exist, create them
  else
    create_libraries
  fi
  # since installation successful, write changed settings
  ## store information about this creation in raven/.ravenrc text file
  if [[ $ECE_VERBOSE == 0 ]]; then echo  ... writing settings to raven/.ravenrc ...; fi
  set_install_settings
fi

# activate environment and write settings if successful
activate_env

if [[ $ECE_VERBOSE == 0 ]]; then echo  ... done!; fi
