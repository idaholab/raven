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
}

# main
# set control variable
REMOTE_ADDRESS=""
HEAD_ADDRESS=""
REDIS_PASS=""
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
  esac
  shift
done


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

# start the script
# ssh in the remote node
ssh $REMOTE_ADDRESS << EOF
  # if REMOTE_BASH, source it
  if [[ "$REMOTE_BASH" != "" ]];
  then
    source $REMOTE_BASH
  fi
  # run ray (the assumption here is that ray is installed in the
  #          remote machine and in the PATH)
  if ! command -v ray &> /dev/null
  then
      echo "ray could not be found in remote node!"
      exit
  fi
  # execute the command
  echo ray start --address=$HEAD_ADDRESS --redis-password=$REDIS_PASS --num-cpus $NUM_CPUS
  # run the command
  ray start --address=$HEAD_ADDRESS --redis-password=$REDIS_PASS --num-cpus $NUM_CPUS
EOF
