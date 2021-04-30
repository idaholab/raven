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
RAVEN_LIB_HANDLER=${ECE_SCRIPT_DIR}/library_handler.py

# fail if ANYTHING this script fails (mostly, there are exceptions)
set -e

function establish_OS ()
{
	case $OSTYPE in
		"linux")
			OSOPTION="--os linux"
			;;
		"linux-gnu")
			OSOPTION="--os linux"
			;;
		"darwin"*)
			OSOPTION="--os mac"
			;;
		"msys"*)
			OSOPTION="--os windows"
			;;
		"cygwin"*)
			OSOPTION="--os windows"
			;;
		*)
			echo Unknown OS: $OSTYPE\; ignoring.
			OSOPTION=""
			;;
	esac
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
        CONDA_DEFS="/c/ProgramData/Miniconda3/etc/profile.d/conda.sh";
      elif test -e "$HOME/miniconda3/etc/profile.d/conda.sh";
      then
        CONDA_DEFS="$HOME/miniconda3/etc/profile.d/conda.sh";
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
  if [[ "$INSTALL_MANAGER" == "CONDA" ]];
  then
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER} ${INSTALL_OPTIONAL} ${OSOPTION} conda --action install)`
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... conda command: \"${COMMAND}\"; fi
    ${COMMAND}
    # conda-forge
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from conda-forge ...; fi
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER} ${INSTALL_OPTIONAL} ${OSOPTION} conda --action install --subset forge)`
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... conda-forge command: ${COMMAND}; fi
    ${COMMAND}
    # pip only
    activate_env
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from PIP-ONLY ...; fi
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER}  ${INSTALL_OPTIONAL} ${OSOPTION} conda --action install --subset pip)`
    if [[ "$PROXY_COMM" != "" ]]; then COMMAND=`echo $COMMAND --proxy $PROXY_COMM`; fi
    if [[ $ECE_VERBOSE == 0 ]]; then echo ...pip-only command: ${COMMAND}; fi
    ${COMMAND}
  else
    # activate the enviroment
    activate_env
    # pip install
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from pip ...; fi
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER}  ${INSTALL_OPTIONAL} ${OSOPTION} pip --action install)`
    if [[ "$PROXY_COMM" != "" ]]; then COMMAND=`echo $COMMAND --proxy $PROXY_COMM`; fi
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... pip command: ${COMMAND}; fi
    ${COMMAND}
  fi
}

function create_libraries()
{
  # TODO there's a lot of redundancy here with install_libraries; maybe this can be consolidated?
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries ...; fi
  if [[ "$INSTALL_MANAGER" == "CONDA" ]];
  then
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER} ${INSTALL_OPTIONAL} ${OSOPTION} conda --action create)`
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... conda command: ${COMMAND}; fi
    ${COMMAND}
    # conda-forge
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from conda-forge ...; fi
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER} ${INSTALL_OPTIONAL} ${OSOPTION} conda --action install --subset forge)`
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... conda-forge command: ${COMMAND}; fi
    ${COMMAND}
    # pip only
    activate_env
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from PIP-ONLY ...; fi
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER}  ${INSTALL_OPTIONAL} ${OSOPTION} conda --action install --subset pip)`
    if [[ "$PROXY_COMM" != "" ]]; then COMMAND=`echo $COMMAND --proxy $PROXY_COMM`; fi
    if [[ $ECE_VERBOSE == 0 ]]; then echo ...pip-only command: ${COMMAND}; fi
    ${COMMAND}
  else
    #pip create virtual enviroment
    local COMMAND=`echo virtualenv $PIP_ENV_LOCATION --python=python`
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... virtual enviroment command: ${COMMAND}; fi
    ${COMMAND}
    # activate the enviroment
    activate_env
    # pip install
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Installing libraries from pip ...; fi
    local COMMAND=`echo $($PYTHON_COMMAND ${RAVEN_LIB_HANDLER}  ${INSTALL_OPTIONAL} ${OSOPTION} pip --action install)`
    if [[ "$PROXY_COMM" != "" ]]; then COMMAND=`echo $COMMAND --proxy $PROXY_COMM`; fi
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... pip command: ${COMMAND}; fi
    ${COMMAND}
  fi
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
	echo '      Defines location of conda definitions (often miniconda3/etc/profile.d/conda.sh). If not provided, guesses based on OS.'
	echo ''
	echo '    --help'
	echo '      Displays this text and exits'
	echo ''
	echo '    --installation-manager'
	echo '      Package installation manager. (CONDA, PIP). If not provided, default to CONDA'
	echo ''
	echo '    --proxy <proxy>'
	echo '      Specify a proxy to be used in the form [user:passwd@]proxy.server:port.'
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
	echo '    --py3'
	echo '    When installing, make raven_libraries use Python 3'
	echo ''
    echo ''
    echo '    --py2'
    echo '    DEPRECATED: When installing, make raven_libraries use Python 2'
    echo ''
    echo ''
	echo '    --quiet'
	echo '      Runs script with minimal output'
	echo ''
}

function activate_env()
{
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Activating environment ...; fi
  if [[ "$INSTALL_MANAGER" == "CONDA" ]];
  then
    conda activate ${RAVEN_LIBS_NAME}
  else
    source ${PIP_ENV_LOCATION}/bin/activate
  fi
}

function set_install_settings()
{
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... Setting install variables ...; fi
  if [[ "$INSTALL_MANAGER" == "CONDA" ]];
  then
    local COMMAND="$PYTHON_COMMAND $ECE_SCRIPT_DIR/update_install_data.py --write --conda-defs ${CONDA_DEFS} --RAVEN_LIBS_NAME ${RAVEN_LIBS_NAME} --python-command ${PYTHON_COMMAND} --installation-manager ${INSTALL_MANAGER}"
  else
    local COMMAND="$PYTHON_COMMAND $ECE_SCRIPT_DIR/update_install_data.py --write --RAVEN_LIBS_NAME ${RAVEN_LIBS_NAME} --python-command ${PYTHON_COMMAND} --installation-manager ${INSTALL_MANAGER}"
  fi
  if [[ $ECE_VERBOSE == 0 ]]; then echo ... ${COMMAND}; fi
  ${COMMAND}
}


# main
# source read ravenrc script
RAVEN_RC_SCRIPT=$ECE_SCRIPT_DIR/read_ravenrc.sh
RAVEN_RC_SCRIPT="${RAVEN_RC_SCRIPT//\\//}"
source $RAVEN_RC_SCRIPT
# set default operation
ECE_MODE=1 # 1 for loading, 2 for install, 0 for help
INSTALL_OPTIONAL="" # --optional if installing optional, otherwise blank
ECE_VERBOSE=0 # 0 for printing, anything else for no printing
ECE_CLEAN=0 # 0 for yes (remove raven libs env before installing), 1 for don't remove it
INSTALL_MANAGER="CONDA" # CONDA (default) or PIP
PROXY_COMM="" # proxy is none

# parse command-line arguments
while test $# -gt 0
do
  case "$1" in
    --help)
      display_usage
      return
      ;;
    --installation-manager)
      shift
      INSTALL_MANAGER=$1
      ;;
    --proxy)
      shift
      PROXY_COMM=$1
      ;;
    --load)
      ECE_MODE=1
      ;;
    --install)
      ECE_MODE=2
      ;;
    --optional)
      echo ... Including optional libraries ...
      INSTALL_OPTIONAL="--optional $INSTALL_OPTIONAL"
      ;;
    --py3)
      echo ... --py3 option detected. --pyX option DEPRECATED. Creating Python 3 libraries ...
      INSTALL_OPTIONAL="$INSTALL_OPTIONAL"
      ;;
    --py2)
      echo ... --py2 option detected. Python 2 is DEPRECATED. Continue creating Python 3 libraries ...
      INSTALL_OPTIONAL="$INSTALL_OPTIONAL"
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

# get installation manager
INSTALL_MANAGER="$(echo $INSTALL_MANAGER | tr '[a-z]' '[A-Z]')"
echo $INSTALL_MANAGER
if [[ "$INSTALL_MANAGER" != "CONDA" && "$INSTALL_MANAGER" != "PIP" ]];
then
  echo ... ERROR: installation-manager $INSTALL_MANAGER UNKNOWN. Available are [CONDA, PIP] !
  exit
fi

if [[ $ECE_VERBOSE == 0 ]];
then
  echo ... Run Options:
  echo ...    Mode: $ECE_MODE
  echo ...   Verbosity: $ECE_VERBOSE
  echo ...   Clean: $ECE_CLEAN
  echo ...    Mode: $INSTALL_MANAGER
  if [[ "$INSTALL_MANAGER" == "CONDA" ]];
  then
    echo ...   Conda Defs: $CONDA_DEFS
  fi
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

if [ -z $PYTHON_COMMAND ];
then
    # check the RC file first
    PYTHON_COMMAND=$(read_ravenrc "PYTHON_COMMAND")
    local_py_command=python
    #If not found through the RC file, will be empty string, so default python
    PYTHON_COMMAND=${PYTHON_COMMAND:=$local_py_command}
fi
export PYTHON_COMMAND
if [[ $ECE_VERBOSE == 0 ]];
then
    echo ... Using Python command ${PYTHON_COMMAND}
fi

# set raven libraries environment name, if not set
if [ -z $RAVEN_LIBS_NAME ];
then
  # check the RC file first
  export RAVEN_LIBS_NAME=$(read_ravenrc "RAVEN_LIBS_NAME")
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
if [[ $ECE_VERBOSE == 0 ]]; then echo ... \>\> RAVEN environment is named \"${RAVEN_LIBS_NAME}\"; fi

if [[ "$INSTALL_MANAGER" == "CONDA" ]];
  then
  # establish conda function definitions (conda 4.4+ ONLY, 4.3 and before not supported)
  find_conda_defs
  if test -e ${CONDA_DEFS};
  then
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Found conda definitions at ${CONDA_DEFS}; fi
    source ${CONDA_DEFS}
    #The next lines are useful sometimes, but excessivly verbose.
    #if [[ $ECE_VERBOSE == 0 ]]; then
    #  echo ... conda:
    #  which conda || echo no conda
    #  conda info || echo conda info failed
    #fi
  else
    echo ... Conda definitions not found at \"${CONDA_DEFS}\"!
    echo ... \>\> Specify the location of miniconda3/etc/profile.d/conda.sh through the --conda-defs option.
    exit 1
  fi
else
  # check if pip exists
  if ! pip_loc="$(type -p pip3)" || [[ -z $pip_loc ]]; then
    echo ... PIP \(pip3\) command not found !!
    echo ... \>\> Install PIP if you want to use it or CONDA as alternative installation manager!
    exit 1
  else
    # install virtual env and upgrade pip
    if ! ve_loc="$(type -p virtualenv)" || [[ -z $ve_loc ]]; then
      pip3 install virtualenv
    fi
    # set PIP_ENV_LOCATION
    PIP_ENV_LOCATION="$HOME/pip_envs"
  fi
fi

# if conda, find the raven libs
if [[ "$INSTALL_MANAGER" == "CONDA" ]];
  then
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
else
  # debug output pip version
  if [[ $ECE_VERBOSE == 0 ]]; then echo `pip3 -V`; fi
  # find RAVEN libraries environment
  PIP_ENV_LOCATION="$PIP_ENV_LOCATION/${RAVEN_LIBS_NAME}"
  if [ -d "$PIP_ENV_LOCATION" ]
  then
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Found library environment ...; fi
    LIBS_EXIST=0
  else
    if [[ $ECE_VERBOSE == 0 ]]; then echo ... Did not find library environment ...; fi
    LIBS_EXIST=1
  fi
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
    echo ${INSTALL_MANAGER} environment ${RAVEN_LIBS_NAME} not found!
    echo Please run "raven/scripts/establish_conda_env.sh" with argument "--install" "--installation-manager $INSTALL_MANAGER".
    exit 1
  fi
fi

# Right before library installation, install ExamplePlugin
echo Installing ExamplePlugin...
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
${parent_path}/install_plugins.py -s ${parent_path}/../plugins/ExamplePlugin


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
      if [[ "$INSTALL_MANAGER" == "CONDA" ]];
      then
        conda deactivate
        conda remove -n ${RAVEN_LIBS_NAME} --all -y
        #Activate base to get python back
        conda activate
      else
        rm -rf ${PIP_ENV_LOCATION}
      fi
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

if [ -z "$RAVEN_SIGNATURE" ];
then
    RAVEN_SIGNATURE=$(read_ravenrc "RAVEN_SIGNATURE")
fi
if [ ! -z "$RAVEN_SIGNATURE" ];
then
    if [[ $ECE_VERBOSE == 0 ]]; then echo "... Using '$RAVEN_SIGNATURE' for signing ..."; fi
    export RAVEN_SIGNATURE
fi

if [[ $ECE_VERBOSE == 0 ]]; then echo  ... done!; fi
