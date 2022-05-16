#!/bin/bash

#Note these also need to be set in the RunInfo:
# <NodeParameter>-machinefile</NodeParameter>
# <MPIExec>mpirun</MPIExec>
# <RemoteRunCommand>raven_qsub_intel.sh</RemoteRunCommand>

module load pbs_is_loaded raven-devel-gcc
module load impi/4.0.3.008
module load imkl/10.3.6.233


if test -n "$PBS_O_WORKDIR"; then
    cd $PBS_O_WORKDIR
fi

which python
which mpiexec
which mpirun
echo $COMMAND
$COMMAND
