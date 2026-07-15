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
    self.__runSbatch = False
    self.__noSplitNode = False #If true, keep all MPI ranks for each batch on one physical node
    self.__maxOnNode = None #Used with __noSplitNode to limit ranks used on one node
    self.__noOverlap = False #Used with __noSplitNode to prevent multiple batches from sharing one node
    self.__restrictScheduler = False #If true, add Slurm placement restrictions for noSplitNode at sbatch time
    self.__exclusive = False #If true with __restrictScheduler, request exclusive nodes
    self.printTag = 'SLURM SIMULATION MODE'

  def __generateSlurmNodeFile(self, nodeFile):
    """
      Generate a node file containing one hostname entry per allocated Slurm task.
      @ In, nodeFile, str, node file path to create
      @ Out, None
    """
    command = ["srun", "--overlap"]
    slurmTasks = os.environ.get("SLURM_NTASKS", os.environ.get("SLURM_NPROCS", None))
    if slurmTasks is not None:
      command.extend(["-n", slurmTasks])
    command.append("hostname")
    with open(nodeFile, "w") as output:
      subprocess.check_call(command, stdout=output)

  def __writeSubNodeFile(self, workingDir, index, group):
    """
      Write one per-batch node file.
      @ In, workingDir, str, base working directory
      @ In, index, int, batch index
      @ In, group, list(str), hostnames to write
      @ Out, None
    """
    with open(os.path.join(workingDir, "node_" + str(index)), "w") as subNodeFile:
      for node in group:
        subNodeFile.write(node.strip() + "\n")

  def __buildNoSplitNodeGroups(self, lines, numMPI):
    """
      Build groups of hostnames with each group entirely on one physical node.
      @ In, lines, list(str), lines from the Slurm node file
      @ In, numMPI, int, number of MPI ranks needed for each RAVEN batch member
      @ Out, groups, list(list(str)), full no-split groups
    """
    nodes = [line.strip() for line in lines if line.strip()]
    nodes.sort()
    groups = []

    def addGroupsForNode(node, slots):
      """
        Add as many full NumMPI groups as possible from a single node.
        @ In, node, str, node name
        @ In, slots, list(str), all available slots for this node
        @ Out, None
      """
      if node is None or len(slots) == 0:
        return
      originalSlots = len(slots)
      if self.__maxOnNode is not None:
        slots = slots[:self.__maxOnNode]
      fullGroups = len(slots) // numMPI
      if self.__noOverlap and fullGroups > 0:
        fullGroups = 1
      for groupIndex in range(fullGroups):
        start = groupIndex * numMPI
        groups.append(slots[start:start+numMPI])
      usedSlots = fullGroups * numMPI
      if fullGroups == 0:
        self.raiseAWarning("not using node " + str(node) + " because it only provides " + str(len(slots)) + " available processor(s), fewer than NumMPI " + str(numMPI))
      elif not self.__noOverlap and len(slots) > usedSlots:
        self.raiseAWarning("not using part of node " + str(node) + " because of partial group: " + str(slots[usedSlots:]))
      elif self.__noOverlap and len(slots) > numMPI:
        self.raiseADebug("not using extra processors on node " + str(node) + " because noOverlap is True")
      if self.__maxOnNode is not None and originalSlots > len(slots):
        self.raiseADebug("not using " + str(originalSlots - len(slots)) + " processor(s) on node " + str(node) + " because maxOnNode is " + str(self.__maxOnNode))

    currentNode = None
    currentSlots = []
    for node in nodes:
      if node != currentNode:
        addGroupsForNode(currentNode, currentSlots)
        currentNode = node
        currentSlots = []
      currentSlots.append(node)
    addGroupsForNode(currentNode, currentSlots)
    return groups

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
        self.__nodeFile = os.path.join(workingDir,"slurmNodeFile_"+str(os.getpid()))
        #generate nodeFile
        self.__generateSlurmNodeFile(self.__nodeFile)
      self.raiseADebug('Setting up remote nodes based on "{}"'.format(self.__nodeFile))
      lines = open(self.__nodeFile,"r").readlines()
      #XXX This is an undocumented way to pass information back
      newRunInfo['Nodes'] = list(lines)
      numMPI = runInfoDict['NumMPI']
      oldBatchsize = runInfoDict['batchSize']
      if self.__noSplitNode:
        groups = self.__buildNoSplitNodeGroups(lines, numMPI)
        maxBatchsize = len(groups)
        if maxBatchsize == 0:
          self.raiseAnError(IOError, "Cannot run with given parameters because no Slurm nodes have NumMPI " + str(numMPI) + " available and noSplitNode is True")
        if maxBatchsize < oldBatchsize:
          newRunInfo['batchSize'] = maxBatchsize
          self.raiseAWarning("changing batchsize from " + str(oldBatchsize) + " to " + str(maxBatchsize) + " because noSplitNode is True and only " + str(maxBatchsize) + " full node group(s) are available")
        newBatchsize = newRunInfo['batchSize']
        self.raiseADebug('Batch size is "{}"'.format(newBatchsize))
        for i, group in enumerate(groups[:newBatchsize]):
          self.__writeSubNodeFile(workingDir, i, group)
        if newBatchsize > 1:
          #then give each index a separate file.
          nodeCommand = runInfoDict["NodeParameter"]+" %BASE_WORKING_DIR%/node_%INDEX% "
        else:
          #For true noSplitNode, even a single batch must use the selected single-node file.
          nodeCommand = runInfoDict["NodeParameter"]+" "+os.path.join(workingDir, "node_0")
      else:
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
            self.__writeSubNodeFile(workingDir, i, lines[i*numMPI : (i+1) * numMPI])
          #then give each index a separate file.
          nodeCommand = runInfoDict["NodeParameter"]+" %BASE_WORKING_DIR%/node_%INDEX% "
        else:
          #If only one batch just use original node file
          nodeCommand = runInfoDict["NodeParameter"]+" "+self.__nodeFile

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

  def __getClusterParameterNames(self, clusterParameters):
    """
      Extract Slurm option names from the user-provided cluster parameters.
      @ In, clusterParameters, list(str), Slurm options supplied by the user
      @ Out, names, set(str), normalized option names
    """
    names = set()
    for param in clusterParameters:
      if not isinstance(param, str):
        continue
      if param.startswith("--"):
        names.add(param.split("=", 1)[0])
      elif param in ("-N", "-n", "-c"):
        names.add(param)
    return names

  def __getNoSplitSchedulerParameters(self, runInfoDict, coresNeeded):
    """
      Build strict Slurm scheduler-side placement parameters for noSplitNode.
      This uses one RAVEN batch member per physical node.
      @ In, runInfoDict, dict, dictionary of run info
      @ In, coresNeeded, int, requested Slurm task count
      @ Out, schedulerParameters, list(str), extra sbatch parameters
    """
    if not (self.__noSplitNode and self.__restrictScheduler):
      return []

    batchSize = runInfoDict['batchSize']
    numMPI = runInfoDict['NumMPI']
    expectedTasks = batchSize * numMPI
    if coresNeeded != expectedTasks:
      self.raiseAnError(IOError, "When noSplitNode restrictScheduler is enabled, coresneeded must be omitted or equal to batchSize*NumMPI. Received coresneeded " + str(coresNeeded) + " but expected " + str(expectedTasks))

    userParameterNames = self.__getClusterParameterNames(runInfoDict["clusterParameters"])
    conflictingNames = sorted(userParameterNames.intersection(set(["--nodes", "-N", "--ntasks-per-node"])))
    if len(conflictingNames) > 0:
      self.raiseAWarning("clusterParameters already contains Slurm placement option(s) " + str(conflictingNames) + "; noSplitNode restrictScheduler will add strict placement options after them")

    schedulerParameters = [
      "--nodes=" + str(batchSize) + "-" + str(batchSize),
      "--ntasks-per-node=" + str(numMPI)
    ]
    if self.__exclusive:
      schedulerParameters.append("--exclusive")
    return schedulerParameters

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
    schedulerParameters = self.__getNoSplitSchedulerParameters(runInfoDict, coresNeeded)
    ## generate the command, which will be passed into "args" of subprocess.call
    command = ["sbatch","--job-name="+jobName]+\
              runInfoDict["clusterParameters"]+\
              schedulerParameters+\
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
    inputSpecification.addSub(InputData.parameterInputFactory("memory", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("coresneeded", contentType=InputTypes.IntegerType))
    inputSpecification.addSub(InputData.parameterInputFactory("partition", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("MPIParam", contentType=InputTypes.StringType))
    for noSplitNodeName in ("noSplitNode", "nosplitnode"):
      noSplitNodeInput = InputData.parameterInputFactory(noSplitNodeName)
      noSplitNodeInput.addParam("maxOnNode", param_type=InputTypes.IntegerType)
      noSplitNodeInput.addParam("noOverlap", param_type=InputTypes.BoolType)
      noSplitNodeInput.addParam("restrictScheduler", param_type=InputTypes.BoolType)
      noSplitNodeInput.addParam("exclusive", param_type=InputTypes.BoolType)
      inputSpecification.addSub(noSplitNodeInput)
    inputSpecification.addSub(InputData.parameterInputFactory("noprecommand"))
    inputSpecification.addSub(InputData.parameterInputFactory("noPrecommand"))
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
      elif childName == "nosplitnode":
        self.__noSplitNode = True
        self.__maxOnNode = child.parameterValues.get("maxOnNode", None)
        self.__noOverlap = child.parameterValues.get("noOverlap", False)
        self.__restrictScheduler = child.parameterValues.get("restrictScheduler", False)
        self.__exclusive = child.parameterValues.get("exclusive", False)
      elif childName == "noprecommand":
        self.__createPrecommand = False
