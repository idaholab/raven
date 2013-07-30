#!/bin/bash

eval `/apps/local/modules/bin/modulecmd bash load pbs python/2.7`

cd $PBS_O_WORKDIR

if [ `echo $MODULEPATH | grep -c '/apps/projects/moose/modulefiles'` -eq 0 ]; then   export MODULEPATH=$MODULEPATH:/apps/projects/moose/modulefiles; fi

#module load python/2.7

export PYTHONPATH=$HOME/raven_libs/pylibs/lib/python2.7/site-packages

which python
echo $COMMAND
$COMMAND
