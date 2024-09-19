#!/bin/bash

echo `pwd`

echo PBS_O_WORKDIR $PBS_O_WORKDIR
hostname

source /etc/profile
AVAIL_MODULES=`module avail 2>&1`
#echo AVAIL_MODULES $AVAIL_MODULES
echo $AVAIL_MODULES | grep 'raven-devel-gcc';
if test 0 -eq $?;
then
    echo found raven-devel-gcc
    module load pbs raven-devel-gcc
else
    echo no raven-devel-gcc
     if [ `echo $MODULEPATH | grep -c '/apps/projects/moose/modulefiles'` -eq 0 ]; then   export MODULEPATH=$MODULEPATH:/apps/projects/moose/modulefiles; fi

     echo "$MODULEPATH"
     module load python/3

     which python3
fi

#Disable MVAPICH2 sched_setaffinity
export MV2_ENABLE_AFFINITY=0

echo $@

OUTPUT=$1
WORKING_DIR=$2
BUFFER_SIZE=$3

cd $WORKING_DIR

shift 3
$@ 2>&1 | dd ibs=$BUFFER_SIZE > $OUTPUT
