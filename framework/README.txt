Framework README

-- Command Replacement Strings --

Inside of JobHandler, external commands can have various strings
replaced.  This is useful for running them on the cluster or to 
vary things depending on the status.  

The JobHandler keeps a list of the running jobs, and the index of this 
can be used.  

%INDEX% - 0 based index in list of running jobs.  Note that this is stable for the life of the job.  
%INDEX1% - 1 based index in the list of running jobs, same as %INDEX%+1
%CURRENT_ID% - 0 based id for the job handler.  This starts as 0, and increases for each job the job handler starts.
%CURRENT_ID1% - 1 based id for the job handler, same as %CURRENT_ID%+1
%SCRIPT_DIR% - Expands to the full path of the script directory (raven/scripts)
%FRAMEWORK_DIR% - Expands to the full path of the framework directory (raven/framework)
%WORKING_DIR% - Expands to the working directory where the input is
%BASE_WORKING_DIR% - Expands to the base working directory given in RunInfo.  This will likely be a parent of WORKING_DIR
%METHOD% - Expands to the environmental variable $METHOD
%NUM_CPUS% - Expands to the number of cpus to use per single batch.  This is ParallelProcNumb in the XML file.

As an example, in pbsdsh mode, the Simulation class add this to the start
of every command:  

pbsdsh -v -n %INDEX1% -- %FRAMEWORK_DIR%/raven_remote.sh out_%CURRENT_ID% %WORKING_DIR%

%INDEX1% is used to choose the node that the command runs on.  
%FRAMEWORK_DIR%/raven_remote.sh is the command that is run, and %FRAMEWORK_DIR% allows the full path to be specified. 
out_%CURRENT_ID% is used to specify a unique output file.  %WORKING_DIR% is used remotely to find the correct working directory.  

Another example:
strace -fo %WORKING_DIR%/strace_%CURRENT_ID% %FRAMEWORK_DIR%/../RAVEN-opt


-- Running on Cluster --

There are two methods of running on a cluster.  The first is to let
the Driver create the qsub command, and the second is to run from an
already submitted interactive qsub command.

--- Changing input ---

In order to run on the cluster, the RunInfo block needs to have the
mode element set to pbsbsh.  If the Driver will be used to create the 
qsub command, then it needs a batchSize parameter to tell how many
nodes to reserve, and an expected time to tell the queuing system
how long the problem should take to run.  The ParallelProcNumb tells
how many threads an individual command should have.

<RunInfo>
...
    <mode>pbsdsh</mode>
    <batchSize>5</batchSize>
    <expectedTime>0:30:00</expectedTime>
    <ParallelProcNumb>1</ParallelProcNumb>
...


--- Running Driver ---

If the Driver is creating the pbs command, then a command like:

module load python/2.7

export PYTHONPATH=$HOME/raven_libs/pylibs/lib/python2.7/site-packages

python Driver.py ../inputs/test_simple5.xml

will work.  The framework will create and submit an qsub command.


--- Running from an interactive node ---

When running from an interactive node, the batchSize is determined from 
the number of nodes that the qsub command used, not from the RunInfo block.
To run interactively, before running the qsub command, switch to the 
framework directory.  Then run the qsub command like:

cd raven/framework

qsub -I -l select=5:ncpus=1 -l walltime=10:00:00 -l place=free

#In side, switch back to the framework directory:

cd $PBS_O_WORKDIR

#Load python 2.7 and pbs

module load python/2.7 pbs


#The Pythonpath needs to include the raven libs:

export PYTHONPATH=$HOME/raven_libs/pylibs/lib/python2.7/site-packages 

#Then the driver can be run

python Driver.py ../inputs/test_simple5.xml

--- Running MPI mode from an interactive node ---

Currently in MPI mode, the framework needs to be run from a node that
supports mpiexec.  

---- Figuring out nodes to run on ----

A file needs to be provided that includes a list of nodes to run on.
This can be specified in the input by changing the mode line to
include a nodefile element:

<mode>mpi<nodefile>/tmp/nodes</nodefile></mode>

Alternatively the name of an environmental variable can be given and
that environmental variable will be used:

<mode>mpi<nodefileenv>NODEFILE</nodefileenv></mode>

If no node file is specified it uses the variable
$PBS_NODEFILE to find a list of nodes that it can run on, otherwise it
will just run on the local node.  Here is an example qsub command and
other things needed to run:

qsub -I -l select=13:ncpus=4:mpiprocs=1 -l walltime=10:00:00 -l place=free

cd raven/framework

export PYTHONPATH=$HOME/raven_libs/pylibs/lib/python2.7/site-packages

module load python/2.7 pbs

if [ `echo $MODULEPATH | grep -c '/apps/projects/moose/modulefiles'` -eq 0 ]; then   export MODULEPATH=$MODULEPATH:/apps/projects/moose/modulefiles; fi
module load moose-dev-gcc python/3.2 


python Driver.py ../inputs/mpi_driver_test/test_mpi.xml


-- Defaults file --

The Driver will look for a file ~/.raven/default_runinfo.xml and will
use it for default values for the data in RunInfo.  These will be
overridden by values in the input file. Example:

<?xml version="1.0" encoding="UTF-8"?>
<Simulation>
<RunInfo>
  <DefaultInputFile>test.xml</DefaultInputFile>
</RunInfo>
</Simulation> 
