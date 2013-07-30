#!/bin/bash

#if test -z "$MODULEPATH"; 
#then
#    export MODULEPATH=/apps/local/modules/modulefiles
#fi

#if test -x /apps/local/modules/bin/modulecmd;
#then
#	MODULECMD=/apps/local/modules/bin/modulecmd
#else
#	if test -x /usr/bin/modulecmd;
#	then 
#		MODULECMD=/usr/bin/modulecmd
#       fi
#fi

echo `pwd`

echo PBS_O_WORKDIR $PBS_O_WORKDIR
hostname

#export -p > $PBS_O_WORKDIR/remote_export
#set > $HOME/raven/fission_comp/trunk/raven/new_set
#cat $HOME/raven/fission_comp/trunk/raven/orig_set | set
#ls -l $PBS_O_WORKDIR/orig_export
#eval `cat $HOME/raven/fission_comp/trunk/raven/orig_set3`
#source $PBS_O_WORKDIR/orig_export

#export -p > $PBS_O_WORKDIR/remote_mod_export

source /etc/profile  
if [ `echo $MODULEPATH | grep -c '/apps/projects/moose/modulefiles'` -eq 0 ]; then   export MODULEPATH=$MODULEPATH:/apps/projects/moose/modulefiles; fi

echo "$MODULEPATH"
#export MODULEPATH
#eval `$MODULECMD bash load moose-dev-gcc pbs python/3.2`
module load moose-dev-gcc pbs python/3.2

#cd $PBS_O_WORKDIR

which python3

echo $@

OUTPUT=$1
WORKING_DIR=$2

cd $WORKING_DIR

shift 2
$@ > $OUTPUT 2>&1

