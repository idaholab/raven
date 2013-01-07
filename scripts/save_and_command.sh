#!/bin/bash

eval `/apps/local/modules/bin/modulecmd bash load pbs`

cd $PBS_O_WORKDIR

if [ `echo $MODULEPATH | grep -c '/apps/projects/moose/modulefiles'` -eq 0 ]; then   export MODULEPATH=$MODULEPATH:/apps/projects/moose/modulefiles; fi

export -p | egrep -v '^declare -x PBS_'  > $PBS_O_WORKDIR/orig_export

echo $COMMAND
$COMMAND
