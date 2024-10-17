# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Module that contains a SimulationMode for Slurm and mpiexec
"""

import os
import math
import string
from ravenframework import Simulation

#For the mode information
modeName = "slurm"
modeClassName = "SlurmSimulationMode"

class SlurmSimulationMode(Simulation.SimulationMode):
  """
    SlurmSimulationMode is a specialized class of SimulationMode.
    It is aimed to distribute the runs on a Slurm cluster
  """

  def __init__(self, *args):
    """
      Constructor
      @ In, args, list, unused positional arguments
      @ Out, None
    """
    super().__init__(*args)
    #figure out if we are in Slurm
    self.__inSlurm = "SLURM_JOB_ID" in os.environ
    self.__nodeFile = False
    self.__coresNeeded = None #If not none, use this instead of calculating it
    self.__memNeeded = None #If not none, use this for mem=
    self.__partition = None #If not none, use this for partition=
    self.__mpiparams = [] #Paramaters to give to mpi
    self.__createPrecommand = True #If true, do create precommand.
    self.printTag = 'SLURM SIMULATION MODE'

  def modifyInfo(self, runInfoDict):
    """
      This method is aimed to modify the Simulation instance in
      order to distribute the jobs using slurm
      @ In, runInfoDict, dict, the original runInfo
      @ Out, newRunInfo, dict, of modified values
    """
    newRunInfo = {}
    newRunInfo['batchSize'] = runInfoDict['batchSize']
    workingDir = runInfoDict['WorkingDir']
    if self.__nodeFile or self.__inSlurm:
      if not self.__nodeFile:
        nodeFile = os.path.join(workingDir,"slurmNodeFile_"+str(os.getpid()))
        #generate nodeFile
        os.system("srun --overlap -- hostname > "+nodeFile)
      else:
        nodeFile = self.__nodeFile
      self.raiseADebug('Setting up remote nodes based on "{}"'.format(nodeFile))
      lines = open(nodeFile,"r").readlines()
      #XXX This is an undocumented way to pass information back
      newRunInfo['Nodes'] = list(lines)
      numMPI = runInfoDict['NumMPI']
      oldBatchsize = runInfoDict['batchSize']
      #the batchsize is just the number of nodes of which there is one
      # per line in the nodeFile divided by the numMPI (which is per run)
      # and the floor and int and max make sure that the numbers are reasonable
      maxBatchsize = max(int(math.floor(len(lines) / numMPI)), 1)

      if maxBatchsize < oldBatchsize:
        newRunInfo['batchSize'] = maxBatchsize
        self.raiseAWarning("changing batchsize from "+str(oldBatchsize)+" to "+str(maxBatchsize)+" to fit on "+str(len(lines))+" processors")
      newBatchsize = newRunInfo['batchSize']
      self.raiseADebug('Batch size is "{}"'.format(newBatchsize))
      if newBatchsize > 1:
        #need to split node lines so that numMPI nodes are available per run
        workingDir = runInfoDict['WorkingDir']
        for i in range(newBatchsize):
          nodeFile = open(os.path.join(workingDir, f"node_{i}"), "w")
          for line in lines[i*numMPI : (i+1) * numMPI]:
            nodeFile.write(line)
          nodeFile.close()
        #then give each index a separate file.
        nodeCommand = runInfoDict["NodeParameter"]+" %BASE_WORKING_DIR%/node_%INDEX% "
      else:
        #If only one batch just use original node file
        nodeCommand = runInfoDict["NodeParameter"]+" "+nodeFile

    else:
      #Not in PBS, so can't look at PBS_NODEFILE and none supplied in input
      newBatchsize = newRunInfo['batchSize']
      numMPI = runInfoDict['NumMPI']
      #TODO, we don't have a way to know which machines it can run on
      # when not in PBS so just distribute it over the local machine:
      nodeCommand = " "

    if len(self.__mpiparams) > 0:
      mpiParams = " ".join(self.__mpiparams)+" "
    else:
      mpiParams = ""
    # Create the mpiexec pre command
    # Note, with defaults the precommand is "mpiexec -f nodeFile -n numMPI"
    if self.__createPrecommand:
      newRunInfo['precommand'] = runInfoDict["MPIExec"]+" "+mpiParams+nodeCommand+" -n "+str(numMPI)+" "+runInfoDict['precommand']
    else:
      newRunInfo['precommand'] = runInfoDict['precommand']
    if runInfoDict['NumThreads'] > 1:
      newRunInfo['threadParameter'] = runInfoDict['threadParameter']
      #add number of threads to the post command.
      newRunInfo['postcommand'] =" {} {}".format(newRunInfo['threadParameter'],runInfoDict['postcommand'])
    self.raiseAMessage("precommand: "+newRunInfo['precommand']+", postcommand: "+newRunInfo.get('postcommand',runInfoDict['postcommand']))
    return newRunInfo

  def __createAndRunSbatch(self, runInfoDict):
    """
      Generates a SLURM sbatch command to run the simulation
      @ In, runInfoDict, dict, dictionary of run info.
      @ Out, remoteRunCommand, dict, dictionary of command.
    """
    # determine the cores needed for the job
    if self.__coresNeeded is not None:
      coresNeeded = self.__coresNeeded
    else:
      coresNeeded = runInfoDict['batchSize']*runInfoDict['NumMPI']

    # get the requested memory, if any
    if self.__memNeeded is not None:
      memString = " --mem="+self.__memNeeded
    else:
      memString = None

    # raven/framework location
    frameworkDir = runInfoDict["FrameworkDir"]
    # number of "threads"
    ncpus = runInfoDict['NumThreads']
    # job title
    jobName = runInfoDict['JobName'] if 'JobName' in runInfoDict.keys() else 'raven_qsub'
    ## fix up job title
    validChars = set(string.ascii_letters).union(set(string.digits)).union(set('_'))
    if any(char not in validChars for char in jobName):
      raise IOError('JobName can only contain alphanumeric and "_" characters! Received'+jobName)
    #--job-name=
    # Generate the sbatch command needed to run input
    ## raven_framework location
    raven = os.path.abspath(os.path.join(frameworkDir,'..','raven_framework'))
    command_env = {}
    command_env.update(os.environ)
    command_env["COMMAND"] = raven + " " + " ".join(runInfoDict["SimulationFiles"])
    command_env["RAVEN_FRAMEWORK_DIR"] = frameworkDir
    ## generate the command, which will be passed into "args" of subprocess.call
    command = ["sbatch","--job-name="+jobName]+\
              runInfoDict["clusterParameters"]+\
              ["--ntasks="+str(coresNeeded),
               "--cpus-per-task="+str(ncpus)]+\
               ([memString] if memString is not None else [])+\
               (["--partition="+self.__partition] if self.__partition is not None else [])+\
              ["--time="+runInfoDict["expectedTime"],
               '--export=ALL,COMMAND,RAVEN_FRAMEWORK_DIR',
               runInfoDict['RemoteRunCommand']]
    # Set parameters for the run command
    remoteRunCommand = {}
    ## directory to start in, where the input file is
    remoteRunCommand["cwd"] = runInfoDict['InputDir']
    ## command to run in that directory
    remoteRunCommand["args"] = command
    print("remoteRunCommand",remoteRunCommand)
    print("COMMAND", command_env["COMMAND"])
    print("RAVEN_FRAMEWORK_DIR", command_env["RAVEN_FRAMEWORK_DIR"])
    remoteRunCommand["env"] = command_env
    ## print out for debugging
    return remoteRunCommand

  def remoteRunCommand(self, runInfoDict):
    """
      If this returns None, don't do anything.  If it returns a
      dictionary, then run the command in the dictionary.
      @ In, runInfoDict, dict, the run info dictionary
      @ Out, remoteRunCommand, dict, a dictionary with information for running.
    """
    if not self.__runSbatch or self.__inSlurm:
      return None
    assert self.__runSbatch and not self.__inSlurm
    return self.__createAndRunSbatch(runInfoDict)

  def XMLread(self, xmlNode):
    """
      XMLread is called with the mode node, and is used here to
      get extra parameters needed for the simulation mode MPI.
      @ In, xmlNode, xml.etree.ElementTree.Element, the xml node that belongs to this class instance
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == "nodefile":
        self.__nodeFile = child.text.strip()
      elif child.tag == "memory":
        self.__memNeeded = child.text.strip()
      elif child.tag == "coresneeded":
        self.__coresNeeded = int(child.text.strip())
      elif child.tag == "partition":
        self.__partition = child.text.strip()
      elif child.tag.lower() == "runsbatch":
        self.__runSbatch = True
      elif child.tag.lower() == "mpiparam":
        self.__mpiparams.append(child.text.strip())
      elif child.tag.lower() == "noprecommand":
        self.__createPrecommand = False
      else:
        self.raiseADebug("We should do something with child "+str(child))
