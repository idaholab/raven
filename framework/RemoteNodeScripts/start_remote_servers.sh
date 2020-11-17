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

function display_usage()
{
	echo ''
	echo '  ------------------------------------------'
	echo '  Default usage:'
	echo '    start_remote_servers.sh'
	echo ''
	echo '  Description:'
	echo '      This script is in charge for instanciating ray servers in remote nodes'
	echo '  ------------------------------------------'
	echo ''
	echo '  Options:'
	echo '    --help'
	echo '      Displays this text and exits'
    echo ''
    echo '    --remote-node-address'
    echo '      Remote node address (ssh into)'
    echo ''
	echo '    --address'
	echo '      Head node address'
	echo ''
	echo '    --redis-password'
	echo '      Specify the password for redis (head node password)'
	echo ''
	echo '    --num-cpus'
	echo '      Number of cpus available/to use in this node'
	echo ''
	echo '    --num-gpus'
	echo '      Number of gpus available/to use in this node'
	echo ''
	echo '    --remote-bash-profile'
	echo '      The bash profile to source before executing the tunneling commands'
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
HEAD_ADDRESS=""
REDIS_PASS=""
PYTHONPATH=""
WORKINGDIR=""
# set default
NUM_CPUS=1
NUM_GPUS=-1
REMOTE_BASH=""

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
    --address)
      shift
      HEAD_ADDRESS=$1
      ;;
    --redis-password)
      shift
      REDIS_PASS=$1
      ;;
    --num-cpus)
      shift
      NUM_CPUS=$1
      ;;
    --num-gpus)
      shift
      NUM_GPUS=$1
      ;;
    --remote-bash-profile)
      shift
      REMOTE_BASH=$1
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

if [[ "$HEAD_ADDRESS" == "" ]];
then
  echo ... ERROR: --address argument must be inputted !
  exit
fi

if [[ "$REDIS_PASS" == "" ]];
then
  echo ... ERROR: --redis-password argument must be inputted !
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

if [[ "$REMOTE_BASH" == "" ]];
then
  ssh $REMOTE_ADDRESS $ECE_SCRIPT_DIR/server_start.py ${WORKINGDIR} ${OUTPUT} ${PYTHONPATH} "${ECE_SCRIPT_DIR}/start_ray.sh $OUTPUT $HEAD_ADDRESS $REDIS_PASS $NUM_CPUS"
else
  ssh $REMOTE_ADDRESS $ECE_SCRIPT_DIR/server_start.py ${WORKINGDIR} ${OUTPUT} ${PYTHONPATH} "${ECE_SCRIPT_DIR}/start_ray.sh $OUTPUT $HEAD_ADDRESS $REDIS_PASS $NUM_CPUS $REMOTE_BASH"
fi


