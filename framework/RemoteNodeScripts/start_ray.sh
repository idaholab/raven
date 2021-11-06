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
### CODE BEGIN ###
# start ray
# OUTPUT FILE FOR LOGGING
OUTFILE=$1
# BASH_PROFILE 
HEAD_ADDRESS=$2
REDIS_PASS=$3
NUM_CPUS=$4
RAVEN_FRAMEWORK_DIR=$5

echo starting >> $OUTFILE

if [ $# -eq 6 ]
  then
  REMOTE_BASH=$6
  source $REMOTE_BASH >> $OUTFILE 2>&1
fi

which ray >> $OUTFILE 2>&1
hostname >> $OUTFILE

echo loaded >> $OUTFILE
command -v ray >> $OUTFILE 2>&1
ray start --address=$HEAD_ADDRESS --redis-password=$REDIS_PASS --num-cpus $NUM_CPUS >> $OUTFILE 2>&1
