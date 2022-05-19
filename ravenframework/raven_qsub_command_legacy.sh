#!/bin/bash

#eval `/apps/local/modules/bin/modulecmd bash load pbs python/2.7`

#cd $PBS_O_WORKDIR

#if [ `echo $MODULEPATH | grep -c '/apps/projects/moose/modulefiles'` -eq 0 ]; then   export MODULEPATH=$MODULEPATH:/apps/projects/moose/modulefiles; fi

#module load python/2.7
#eval `/apps/local/modules/bin/modulecmd bash load moose-dev-gcc python/3.2`

#export PYTHONPATH=$HOME/raven_libs/pylibs/lib/python2.7/site-packages

module purge
module load pbs_is_loaded raven-devel
source activate raven_libraries

if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

which python
which mpiexec
echo $COMMAND
$COMMAND
