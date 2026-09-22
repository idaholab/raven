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
import subprocess
from ravenframework import Simulation
from ravenframework.utils import InputData, InputTypes
from ravenframework.CustomModes import ClusterUtils
from ravenframework.CustomModes.ClusterMode import ClusterSimulationMode

#For the mode information
modeName = "slurm"
modeClassName = "SlurmSimulationMode"

class SlurmSimulationMode(ClusterSimulationMode):
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
    self.__runSbatch = False #If true, submit this run via sbatch when outside Slurm.
    self.__useSrun = False #If true, launch runs with native srun instead of mpiexec+nodefiles.
    self.printTag = 'SLURM SIMULATION MODE'

  def modifyInfo(self, runInfoDict):
    """
      This method is aimed to modify the Simulation instance in
      order to distribute the jobs using slurm
      @ In, runInfoDict, dict, the original runInfo
      @ Out, newRunInfo, dict, of modified values
    """
    workingDir = runInfoDict['WorkingDir']
    nodeFileName = None
    if self.__nodeFile or self.__inSlurm:
      if not self.__nodeFile:
        self.__nodeFile = os.path.join(workingDir,"slurmNodeFile_"+str(os.getpid()))
        #generate nodeFile (checked srun, with scontrol-based fallback)
        self.__generateNodeFile(self.__nodeFile)
      nodeFileName = self.__nodeFile
    if self.__useSrun:
      #srun-native launch: Slurm assigns per-step resources itself, so no
      # per-batch node files are needed; keep node discovery for the Nodes
      # bookkeeping (used e.g. by the internal-parallel cluster bring-up)
      newRunInfo = self._modifyInfoForCluster(runInfoDict, nodeFileName,
                                              mpiParams=None,
                                              createPrecommand=False,
                                              splitNodeFiles=False,
                                              clusterName="this Slurm allocation")
      if self.__createPrecommand:
        newRunInfo['precommand'] = ClusterUtils.buildSrunPrecommand(
          runInfoDict['NumMPI'], self.__mpiparams, runInfoDict['precommand'])
        self.raiseAMessage("srun precommand: "+newRunInfo['precommand'])
      return newRunInfo
    #the batch sizing, node-file splitting and precommand assembly are shared
    # with the other cluster modes (see ClusterMode.ClusterSimulationMode)
    return self._modifyInfoForCluster(runInfoDict, nodeFileName,
                                      mpiParams=self.__mpiparams,
                                      createPrecommand=self.__createPrecommand,
                                      clusterName="this Slurm allocation")

  def __generateNodeFile(self, nodeFileName):
    """
      Generates the node file (one line per available task/processor) for the
      current Slurm allocation. Tries "srun hostname" first and falls back to
      expanding $SLURM_JOB_NODELIST via "scontrol show hostnames".
      @ In, nodeFileName, str, the path of the node file to write
      @ Out, None
    """
    lines = None
    try:
      result = subprocess.run(["srun", "--overlap", "--", "hostname"],
                              capture_output=True, text=True, timeout=300)
      if result.returncode == 0 and result.stdout.strip():
        lines = [line for line in result.stdout.splitlines() if line.strip()]
      else:
        self.raiseAWarning('"srun --overlap -- hostname" failed (return code '
                           f'{result.returncode}): {result.stderr.strip()}')
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
      self.raiseAWarning(f'Unable to run "srun --overlap -- hostname": {exc}')
    if lines is None:
      # fall back to scontrol-based expansion of the allocation node list
      self.raiseADebug('Falling back to "scontrol show hostnames" for node discovery')
      try:
        lines = ClusterUtils.slurmNodeListFromScontrol()
      except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        self.raiseAnError(RuntimeError, 'Could not determine the nodes of this Slurm '
                          f'allocation with either srun or scontrol: {exc}')
    if not lines:
      self.raiseAnError(RuntimeError, 'Slurm node discovery returned no nodes! '
                        'Check that RAVEN is running inside a valid allocation.')
    with open(nodeFileName, "w") as nodeFileObject:
      for line in lines:
        nodeFileObject.write(line.strip() + "\n")

  def __createAndRunSbatch(self, runInfoDict):
    """
      Generates a SLURM sbatch command to run the simulation
      @ In, runInfoDict, dict, dictionary of run info.
      @ Out, remoteRunCommand, dict, dictionary of command.
    """
    # determine the cores needed for the job. Note that these can be distributed
    #  that is they may not be able to share memory.
    if self.__coresNeeded is not None:
      coresNeeded = self.__coresNeeded
    else:
      coresNeeded = runInfoDict['batchSize']*runInfoDict['NumMPI']

    # get the requested memory, if any
    if self.__memNeeded is not None:
      memString = "--mem="+self.__memNeeded
    else:
      memString = None

    # raven/framework location
    frameworkDir = runInfoDict["FrameworkDir"]
    # number of "threads" (unlike cores, these will run on a single computer
    #  and so can share memory)
    ncpus = runInfoDict['NumThreads']
    # job title
    jobName = runInfoDict['JobName'] if 'JobName' in runInfoDict.keys() else 'raven_qsub'
    ## fix up job title (shared validator; alphanumeric, "_" and "-" allowed)
    try:
      jobName = ClusterUtils.sanitizeJobName(jobName)
    except ValueError as err:
      self.raiseAnError(IOError, str(err))
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
    self.raiseAMessage("remoteRunCommand: "+str(remoteRunCommand))
    self.raiseADebug("COMMAND: "+command_env["COMMAND"])
    self.raiseADebug("RAVEN_FRAMEWORK_DIR: "+command_env["RAVEN_FRAMEWORK_DIR"])
    remoteRunCommand["env"] = command_env
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


  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = InputData.parameterInputFactory("mode", ordered=False, contentType=InputTypes.StringType)
    inputSpecification.addSub(InputData.parameterInputFactory("runSbatch"))
    inputSpecification.addSub(InputData.parameterInputFactory("nodefile", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("nodefileenv", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("memory", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("coresneeded", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("partition", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("MPIParam", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("noprecommand"))
    inputSpecification.addSub(InputData.parameterInputFactory("useSrun"))
    return inputSpecification

  def handleInput(self, paramInput):
    """
      Function to handle the slurm mode parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      childName = child.getName().lower()
      if childName == "nodefile":
        self.__nodeFile = child.value.strip()
      elif childName == "nodefileenv":
        envName = child.value.strip()
        if envName not in os.environ:
          self.raiseAnError(IOError, f'<nodefileenv> environment variable "{envName}" '
                            'is not defined in the current environment!')
        self.__nodeFile = os.environ[envName]
      elif childName == "memory":
        self.__memNeeded = child.value.strip()
      elif childName == "coresneeded":
        self.__coresNeeded = child.value
      elif childName == "partition":
        self.__partition = child.value.strip()
      elif childName == "runsbatch":
        self.__runSbatch = True
      elif childName == "mpiparam":
        self.__mpiparams.append(child.value.strip())
      elif childName == "noprecommand":
        self.__createPrecommand = False
      elif childName == "usesrun":
        self.__useSrun = True
      else:
        self.raiseAWarning(f'Unrecognized <mode> option "{child.getName()}" ignored '
                           'by the Slurm simulation mode.')
