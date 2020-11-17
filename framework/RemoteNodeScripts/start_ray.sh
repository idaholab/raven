#!/bin/bash
# start ray
# OUTPUT FILE FOR LOGGING
OUTFILE=$1
# BASH_PROFILE 
HEAD_ADDRESS=$2
REDIS_PASS=$3
NUM_CPUS=$4

echo starting >> $OUTFILE

if [ $# -eq 5 ]
  then
  REMOTE_BASH=$5
  source $REMOTE_BASH >> $OUTFILE 2>&1
fi

which ray >> $OUTFILE 2>&1
hostname >> $OUTFILE

echo loaded >> $OUTFILE
command -v ray >> $OUTFILE 2>&1
ray start --address=$HEAD_ADDRESS --redis-password=$REDIS_PASS --num-cpus $NUM_CPUS >> $OUTFILE 2>&1

