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

function display_usage()
{
	echo ''
	echo '  ------------------------------------------'
	echo '  Default usage:'
	echo '    update_path_in_remote_servers.sh'
	echo ''
	echo '  Description:'
	echo '      This script is in charge for updating the python path'
	echo '  ------------------------------------------'
	echo ''
	echo '  Options:'
	echo '    --help'
	echo '      Displays this text and exits'
    echo ''
    echo '    --remote-node-address'
    echo '      Remote node address (ssh into)'
    echo ''
	echo '     --python-path'
	echo '      The PYTHONPATH enviroment variable'
	echo ''
	echo '     --working-dir'
	echo '      The workind directory'
	echo ''
 
}

# main
# set control variable
REMOTE_ADDRESS=""
PYTHONPATH=""
WORKINGDIR=""

# parse command-line arguments
while test $# -gt 0
do
  case "$1" in
    --help)
      display_usage
      return
      ;;
    --remote-node-address)
      shift
      REMOTE_ADDRESS=$1
      ;;
    --python-path)
      shift
      PYTHONPATH=$1
      ;;
      --working-dir)
      shift
      WORKINGDIR=$1
      ;;
  esac
  shift
done

echo $REMOTE_ADDRESS
if [[ "$REMOTE_ADDRESS" == "" ]];
then
  echo ... ERROR: --remote-node-address argument must be inputted !
  exit
fi

if [[ "$PYTHONPATH" == "" ]];
then
  echo ... ERROR: --python-path argument must be inputted !
  exit
fi

if [[ "$WORKINGDIR" == "" ]];
then
  echo ... ERROR: --working-dir argument must be inputted !
  exit
fi


# start the script
# ssh in the remote node and run the ray servers
CWD=`pwd`
OUTPUT=$CWD/server_debug_$REMOTE_ADDRESS

ssh $REMOTE_ADDRESS $ECE_SCRIPT_DIR/server_update_path.py ${WORKINGDIR} ${OUTPUT} ${PYTHONPATH}



