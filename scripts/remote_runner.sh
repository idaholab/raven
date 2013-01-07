#!/bin/bash

#if test -z "$MODULEPATH"; 
#then
#    export MODULEPATH=/apps/local/modules/modulefiles
#fi

echo `pwd`

echo PBS_O_WORKDIR $PBS_O_WORKDIR

export -p > $PBS_O_WORKDIR/remote_export
#set > $HOME/raven/fission_comp/trunk/raven/new_set
#cat $HOME/raven/fission_comp/trunk/raven/orig_set | set
ls -l $PBS_O_WORKDIR/orig_export
#eval `cat $HOME/raven/fission_comp/trunk/raven/orig_set3`
source $PBS_O_WORKDIR/orig_export

export -p > $PBS_O_WORKDIR/remote_mod_export
    
echo "$MODULEPATH"
#export MODULEPATH
eval `/apps/local/modules/bin/modulecmd bash load moose-dev-gcc pbs python/3.2`

cd $PBS_O_WORKDIR

which python3

echo $@

OUTPUT=$1

shift 1
$@ > $OUTPUT 2>&1

