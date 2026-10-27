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
Module that contains a SimulationMode for PBSPro and mpiexec
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

import os
import math
import string
from ravenframework import Simulation
from ravenframework.CustomModes import ClusterUtils
from ravenframework.CustomModes.ClusterMode import ClusterSimulationMode

#For the mode information
modeName = ["mpi","pbs"]
modeClassName = "PBSSimulationMode"

class PBSSimulationMode(ClusterSimulationMode):
  """
    PBSSimulationMode is a specialized class of SimulationMode.
    It is aimed to distribute the runs using the MPI protocol on PBS
  """
  def __init__(self, *args):
    """
      Constructor
      @ In, args, list, unused positional arguments
      @ Out, None
    """
    super().__init__(*args)
    #Figure out if we are in PBS
    self.__inPbs = "PBS_NODEFILE" in os.environ
    self.__nodefile = False
    self.__runQsub = False
    self.__coresNeeded = None #If not none, use this instead of calculating it
    self.__memNeeded = None #If not none, use this for mem=
    self.__place = "free" #use this for place=
    self.__mpiparams = [] #Paramaters to give to mpi
    self.__createPrecommand = True #If true, do create precommand.
    self.printTag = 'MPI SIMULATION MODE'

  def modifyInfo(self, runInfoDict):
    """
      This method is aimed to modify the Simulation instance in
      order to distribute the jobs using the MPI protocol
      @ In, runInfoDict, dict, the original runInfo
      @ Out, newRunInfo, dict, of modified values
    """
    nodeFileName = None
    if self.__nodefile or self.__inPbs:
      if not self.__nodefile:
        #Figure out number of nodes and use for batchsize
        nodeFileName = os.environ["PBS_NODEFILE"]
      else:
        nodeFileName = self.__nodefile

    #Disable MPI processor affinity, which causes multiple processes
    # to be forced to the same thread.
    os.environ["MV2_ENABLE_AFFINITY"] = "0"

    #the batch sizing, node-file splitting and precommand assembly are shared
    # with the other cluster modes (see ClusterMode.ClusterSimulationMode)
    return self._modifyInfoForCluster(runInfoDict, nodeFileName,
                                      mpiParams=self.__mpiparams,
                                      createPrecommand=self.__createPrecommand,
                                      clusterName="this PBS allocation")

  def __createAndRunQSUB(self, runInfoDict):
    """
      Generates a PBS qsub command to run the simulation
      @ In, runInfoDict, dict, dictionary of run info.
      @ Out, remoteRunCommand, dict, dictionary of command.
    """
    # Check if the simulation has been run in PBS mode and, in case, construct the proper command
    # determine the cores needed for the job
    if self.__coresNeeded is not None:
      coresNeeded = self.__coresNeeded
    else:
      coresNeeded = runInfoDict['batchSize']*runInfoDict['NumMPI']

    # get the requested memory, if any
    if self.__memNeeded is not None:
      memString = ":mem="+self.__memNeeded
    else:
      memString = ""
    # raven/framework location
    frameworkDir = runInfoDict["FrameworkDir"]
    # number of "threads"
    ncpus = runInfoDict['NumThreads']
    # job title
    jobName = runInfoDict['JobName'] if 'JobName' in runInfoDict.keys() else 'raven_qsub'
    ## fix up job title (shared validator; PBS limits names to 15 characters)
    try:
      shortJobName = ClusterUtils.sanitizeJobName(jobName, maxLength=15)
    except ValueError as err:
      self.raiseAnError(IOError, str(err))
    if shortJobName != jobName:
      self.raiseAMessage('JobName is limited to 15 characters; truncating to '+shortJobName)
    jobName = shortJobName
    # Generate the qsub command needed to run input
    ## raven_framework location
    raven = os.path.abspath(os.path.join(frameworkDir,'..','raven_framework'))
    ## generate the command, which will be passed into "args" of subprocess.call
    command = ["qsub","-N",jobName]+\
              runInfoDict["clusterParameters"]+\
              ["-l",
                  "select={}:ncpus={}:mpiprocs=1{}".format(coresNeeded,ncpus,memString),
               "-l","walltime="+runInfoDict["expectedTime"],
               "-l","place="+self.__place,"-v",
               'COMMAND="{} '.format(raven)+
               " ".join(runInfoDict["SimulationFiles"])+'",'+
               'RAVEN_FRAMEWORK_DIR="{}"'.format(frameworkDir),
               runInfoDict['RemoteRunCommand']]
    # Set parameters for the run command
    remoteRunCommand = {}
    ## directory to start in, where the input file is
    remoteRunCommand["cwd"] = runInfoDict['InputDir']
    ## command to run in that directory
    remoteRunCommand["args"] = command
    ## print out for debugging
    print("remoteRunCommand",remoteRunCommand)
    return remoteRunCommand

  def remoteRunCommand(self, runInfoDict):
    """
      If this returns None, don't do anything.  If it returns a
      dictionary, then run the command in the dictionary.
      @ In, runInfoDict, dict, the run info dictionary
      @ Out, remoteRunCommand, dict, a dictionary with information for running.
    """
    if not self.__runQsub or self.__inPbs:
      return None
    assert self.__runQsub and not self.__inPbs
    return self.__createAndRunQSUB(runInfoDict)

  def XMLread(self, xmlNode):
    """
      XMLread is called with the mode node, and is used here to
      get extra parameters needed for the simulation mode MPI.
      @ In, xmlNode, xml.etree.ElementTree.Element, the xml node that belongs to this class instance
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == "nodefileenv":
        envName = child.text.strip()
        if envName not in os.environ:
          self.raiseAnError(IOError, f'<nodefileenv> environment variable "{envName}" '
                            'is not defined in the current environment!')
        self.__nodefile = os.environ[envName]
      elif child.tag == "nodefile":
        self.__nodefile = child.text.strip()
      elif child.tag == "memory":
        self.__memNeeded = child.text.strip()
      elif child.tag == "coresneeded":
        self.__coresNeeded = int(child.text.strip())
      elif child.tag == "place":
        self.__place = child.text.strip()
      elif child.tag.lower() == "runqsub":
        self.__runQsub = True
      elif child.tag.lower() == "mpiparam":
        self.__mpiparams.append(child.text.strip())
      elif child.tag.lower() == "noprecommand":
        self.__createPrecommand = False
      else:
        self.raiseADebug("We should do something with child "+str(child))
