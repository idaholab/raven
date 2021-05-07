#!/bin/bash
# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created on Feb 16, 2013
#
# @author: alfoa
#

# DESCRIPTION:
# This script is in charge for instanciating ray servers in remote nodes
# USES:
# --remote-node-address - Remote node address (ssh into)
# --address - Head node address
# --redis-password - Specify the password for redis (head node password)
# --num-cpus - Number of cpus available/to use in this node
# --num-gpus - Number of gpus available/to use in this node
# --remote-bash-profile - The bash profile to source before executing the tunneling commands
# --python-path - The PYTHONPATH enviroment variable
# --working-dir - The workind directory
# --help - Displays the info above and exits

ECE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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


