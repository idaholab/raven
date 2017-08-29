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
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

import os
import math
import string
import Simulation

#For the mode information
modeName = "mpi"
modeClassName = "MPISimulationMode"

class MPISimulationMode(Simulation.SimulationMode):
  """
    MPISimulationMode is a specialized class of SimulationMode.
    It is aimed to distribute the runs using the MPI protocol
  """
  def __init__(self,messageHandler):
    """
      Constructor
      @ In, messageHandler, instance, instance of the messageHandler class
      @ Out, None
    """
    Simulation.SimulationMode.__init__(self,messageHandler)
    self.messageHandler = messageHandler
    #Figure out if we are in PBS
    self.__inPbs = "PBS_NODEFILE" in os.environ
    self.__nodefile = False
    self.__runQsub = False
    self.__coresNeeded = None #If not none, use this instead of calculating it
    self.__memNeeded = None #If not none, use this for mem=
    self.__place = "free" #use this for place=
    self.printTag = 'MPI SIMULATION MODE'

  def modifyInfo(self, runInfoDict):
    """
      This method is aimed to modify the Simulation instance in
      order to distribute the jobs using the MPI protocol
      @ In, runInfoDict, dict, the original runInfo
      @ Out, newRunInfo, dict, of modified values
    """
    newRunInfo = {}
    newRunInfo['batchSize'] = runInfoDict['batchSize']
    if self.__nodefile or self.__inPbs:
      if not self.__nodefile:
        #Figure out number of nodes and use for batchsize
        nodefile = os.environ["PBS_NODEFILE"]
      else:
        nodefile = self.__nodefile
      lines = open(nodefile,"r").readlines()
      #XXX This is an undocumented way to pass information back
      newRunInfo['Nodes'] = list(lines)
      numMPI = runInfoDict['NumMPI']
      oldBatchsize = runInfoDict['batchSize']
      #the batchsize is just the number of nodes of which there is one
      # per line in the nodefile divided by the numMPI (which is per run)
      # and the floor and int and max make sure that the numbers are reasonable
      maxBatchsize = max(int(math.floor(len(lines)/numMPI)),1)
      if maxBatchsize < oldBatchsize:
        newRunInfo['batchSize'] = maxBatchsize
        self.raiseAWarning("changing batchsize from "+str(oldBatchsize)+" to "+str(maxBatchsize)+" to fit on "+str(len(lines))+" processors")
      newBatchsize = newRunInfo['batchSize']
      if newBatchsize > 1:
        #need to split node lines so that numMPI nodes are available per run
        workingDir = runInfoDict['WorkingDir']
        for i in range(newBatchsize):
          nodeFile = open(os.path.join(workingDir,"node_"+str(i)),"w")
          for line in lines[i*numMPI:(i+1)*numMPI]:
            nodeFile.write(line)
          nodeFile.close()

        #then give each index a separate file.
        nodeCommand = runInfoDict["NodeParameter"]+" %BASE_WORKING_DIR%/node_%INDEX% "
      else:
        #If only one batch just use original node file
        nodeCommand = runInfoDict["NodeParameter"]+" "+nodefile
    else:
      #Not in PBS, so can't look at PBS_NODEFILE and none supplied in input
      newBatchsize = newRunInfo['batchSize']
      numMPI = runInfoDict['NumMPI']
      #TODO, we don't have a way to know which machines it can run on
      # when not in PBS so just distribute it over the local machine:
      nodeCommand = " "

    #Disable MPI processor affinity, which causes multiple processes
    # to be forced to the same thread.
    os.environ["MV2_ENABLE_AFFINITY"] = "0"

    # Create the mpiexec pre command
    # Note, with defaults the precommand is "mpiexec -f nodeFile -n numMPI"
    newRunInfo['precommand'] = runInfoDict["MPIExec"]+" "+nodeCommand+" -n "+str(numMPI)+" "+runInfoDict['precommand']
    if(runInfoDict['NumThreads'] > 1):
      #add number of threads to the post command.
      newRunInfo['postcommand'] = " --n-threads=%NUM_CPUS% "+runInfoDict['postcommand']
    self.raiseAMessage("precommand: "+newRunInfo['precommand']+", postcommand: "+newRunInfo.get('postcommand',runInfoDict['postcommand']))
    return newRunInfo

  def __createAndRunQSUB(self, runInfoDict):
    """
      Generates a PBS qsub command to run the simulation
      @ In, runInfoDict, dict, dictionary of run info.
      @ Out, remoteRunCommand, dict, dictionary of command.
    """
    # Check if the simulation has been run in PBS mode and, in case, construct the proper command
    #while true, this is not the number that we want to select
    if self.__coresNeeded is not None:
      coresNeeded = self.__coresNeeded
    else:
      coresNeeded = runInfoDict['batchSize']*runInfoDict['NumMPI']
    if self.__memNeeded is not None:
      memString = ":mem="+self.__memNeeded
    else:
      memString = ""
    #batchSize = runInfoDict['batchSize']
    frameworkDir = runInfoDict["FrameworkDir"]
    ncpus = runInfoDict['NumThreads']
    jobName = runInfoDict['JobName'] if 'JobName' in runInfoDict.keys() else 'raven_qsub'
    #check invalid characters
    validChars = set(string.ascii_letters).union(set(string.digits)).union(set('-_'))
    if any(char not in validChars for char in jobName):
      raise IOError('JobName can only contain alphanumeric and "_", "-" characters! Received'+jobName)
    #check jobName for length
    if len(jobName) > 15:
      jobName = jobName[:10]+'-'+jobName[-4:]
      print('JobName is limited to 15 characters; truncating to '+jobName)
    #Generate the qsub command needed to run input
    command = ["qsub","-N",jobName]+\
              runInfoDict["clusterParameters"]+\
              ["-l",
               "select="+str(coresNeeded)+":ncpus="+str(ncpus)+":mpiprocs=1"+memString,
               "-l","walltime="+runInfoDict["expectedTime"],
               "-l","place="+self.__place,"-v",
               'COMMAND="../raven_framework '+
               " ".join(runInfoDict["SimulationFiles"])+'"',
               runInfoDict['RemoteRunCommand']]
    #Change to frameworkDir so we find raven_qsub_command.sh
    remoteRunCommand = {}
    remoteRunCommand["cwd"] = frameworkDir
    remoteRunCommand["args"] = command
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
        self.__nodefile = os.environ[child.text.strip()]
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
      else:
        self.raiseADebug("We should do something with child "+str(child))
