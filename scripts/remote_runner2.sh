#!/bin/bash

echo `pwd`

echo PBS_O_WORKDIR $PBS_O_WORKDIR
hostname

source /etc/profile  
if [ `echo $MODULEPATH | grep -c '/apps/projects/moose/modulefiles'` -eq 0 ]; then   export MODULEPATH=$MODULEPATH:/apps/projects/moose/modulefiles; fi

echo "$MODULEPATH"
module load moose-dev-gcc pbs python/3.2

#cd $PBS_O_WORKDIR

which python3

echo $@

OUTPUT=$1
WORKING_DIR=$2

cd $WORKING_DIR

shift 2
$@ > $OUTPUT 2>&1

