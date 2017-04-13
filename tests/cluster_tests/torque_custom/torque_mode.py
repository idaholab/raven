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
from __future__ import division, print_function, unicode_literals, absolute_import

import os
import string
import subprocess
import Simulation
import math

def createAndRunQSUB(simulation):
  """
    Generates a PBS qsub command to run the simulation
    @ In, simulation, instance, instance of the simulation class
    @ Out, None
  """
  # Check if the simulation has been run in PBS mode and, in case, construct the proper command
  #while true, this is not the number that we want to select
  coresNeeded = simulation.runInfoDict['batchSize']*simulation.runInfoDict['NumMPI']
  #batchSize = simulation.runInfoDict['batchSize']
  frameworkDir = simulation.runInfoDict["FrameworkDir"]
  ncpus = simulation.runInfoDict['NumThreads']
  jobName = simulation.runInfoDict['JobName'] if 'JobName' in simulation.runInfoDict.keys() else 'raven_qsub'
  #check invalid characters
  validChars = set(string.ascii_letters).union(set(string.digits)).union(set('-_'))
  if any(char not in validChars for char in jobName):
    simulation.raiseAnError(IOError,'JobName can only contain alphanumeric and "_", "-" characters! Received'+jobName)
  #check jobName for length
  if len(jobName) > 15:
    jobName = jobName[:10]+'-'+jobName[-4:]
    simulation.raiseAMessage('JobName is limited to 15 characters; truncating to '+jobName)
  #Generate the qsub command needed to run input
  command = ["qsub","-N",jobName]+\
            simulation.runInfoDict["clusterParameters"]+\
            ["-l",
             "nodes="+str(coresNeeded)+":ppn="+str(ncpus),
             "-l","walltime="+simulation.runInfoDict["expectedTime"],
             "-v",
             'COMMAND=python Driver.py '+
             " ".join(simulation.runInfoDict["SimulationFiles"]),
             simulation.runInfoDict['RemoteRunCommand']]
  #Change to frameworkDir so we find raven_qsub_command.sh
  os.chdir(frameworkDir)
  simulation.raiseAMessage(os.getcwd()+' '+str(command))
  subprocess.call(command)


class TorqueSimulationMode(Simulation.SimulationMode):
  """Implements SimulationMode to add the new mode torque
  In this mode, torque is used to run new commands.
  """

  def __init__(self,simulation):
    """
      Create a new TorqueSimulationMode instance.
      @ In, simulation, Simulation.simulation object, the base simulation object that this thing will build off of
      @ Out, None
    """
    Simulation.SimulationMode.__init__(self,simulation)
    self.__simulation = simulation
    #Check if in pbs by seeing if environmental variable exists
    self.__inPbs = "PBS_NODEFILE" in os.environ
    self.__nodefile = False
    self.__runQsub = False
    self.__noSplitNode = False #If true, don't split mpi processes across nodes
    self.__limitNode = False #If true, fiddle with max on Node
    self.__maxOnNode = None #Used with __noSplitNode and __limitNode to limit number on a node
    self.__noOverlap = False #Used with __limitNode to prevent multiple batches from being on one node
    #self.printTag = returnPrintTag('PBSDSH SIMULATION MODE')

  def doOverrideRun(self):
    """
      Check if the simulation has been run in PBS mode and
      if not the run needs to be overridden so qsub can be called.
      @ In, None
      @ Out, None
    """
    doOverrRun = (not self.__inPbs) and self.__runQsub
    return doOverrRun

  def runOverride(self):
    """
      If not in pbs mode, qsub needs to be called.
      @ In, None
      @ Out, None
    """
    #Check and see if this is being accidently run
    assert not self.__inPbs
    createAndRunQSUB(self.__simulation)

  def modifySimulation(self):
    """
      This method is aimed to modify the Simulation instance in
      order to distribute the jobs using the MPI protocol
      @ In, None
      @ Out, None
    """
    if self.__nodefile or self.__inPbs:
      if not self.__nodefile:
        #Figure out number of nodes and use for batchsize
        nodefile = os.environ["PBS_NODEFILE"]
      else:
        nodefile = self.__nodefile
      lines = open(nodefile,"r").readlines()
      self.__simulation.runInfoDict['Nodes'] = list(lines)
      numMPI = self.__simulation.runInfoDict['NumMPI']
      oldBatchsize = self.__simulation.runInfoDict['batchSize']
      #the batchsize is just the number of nodes of which there is one
      # per line in the nodefile divided by the numMPI (which is per run)
      # and the floor and int and max make sure that the numbers are reasonable
      maxBatchsize = max(int(math.floor(len(lines)/numMPI)),1)
      if maxBatchsize < oldBatchsize:
        self.__simulation.runInfoDict['batchSize'] = maxBatchsize
        self.raiseAWarning("changing batchsize from "+str(oldBatchsize)+" to "+str(maxBatchsize)+" to fit on "+str(len(lines))+" processors")
      newBatchsize = self.__simulation.runInfoDict['batchSize']
      if newBatchsize > 1:
        #need to split node lines so that numMPI nodes are available per run
        workingDir = self.__simulation.runInfoDict['WorkingDir']
        if not (self.__noSplitNode or self.__limitNode):
          for i in range(newBatchsize):
            nodeFile = open(os.path.join(workingDir,"node_"+str(i)),"w")
            for line in lines[i*numMPI:
              (i+1)*numMPI]:
              nodeFile.write(line)
            nodeFile.close()
        else:
          #self.__noSplitNode == True or self.__limitNode == True
          #XXX This may be much more complicated than needed.
          # The needed functionality probably needs to be discussed.
          nodes = []
          for line in lines:
            nodes.append(line.strip())

          nodes.sort()

          currentNode = ""
          countOnNode = 0
          nodeUsed = False

          if self.__noSplitNode:
            groups = []
          else:
            groups = [[]]

          for i in range(len(nodes)):
            node = nodes[i]
            if node != currentNode:
              currentNode = node
              countOnNode = 0
              nodeUsed = False
              if self.__noSplitNode:
                #When switching node, make new group
                groups.append([])
            if self.__maxOnNode is None or countOnNode < self.__maxOnNode:
              countOnNode += 1
              if len(groups[-1]) >= numMPI:
                groups.append([])
                nodeUsed = True
              if not self.__noOverlap or not nodeUsed:
                groups[-1].append(node)

          fullGroupCount = 0
          for group in groups:
            if len(group) < numMPI:
              self.raiseAWarning("not using part of node because of partial group: "+str(group))
            else:
              nodeFile = open(os.path.join(workingDir,"node_"+str(fullGroupCount)),"w")
              for node in group:
                print(node,file=nodeFile)
              nodeFile.close()
              fullGroupCount += 1
          if fullGroupCount == 0:
            self.raiseAnError(IOError, "Cannot run with given parameters because no nodes have numMPI "+str(numMPI)+" available and NoSplitNode is "+str(self.__noSplitNode)+" and LimitNode is "+str(self.__limitNode))
          if fullGroupCount != self.__simulation.runInfoDict['batchSize']:
            self.raiseAWarning("changing batchsize to "+str(fullGroupCount)+" because NoSplitNode is "+str(self.__noSplitNode)+" and LimitNode is "+str(self.__limitNode)+" and some nodes could not be used.")
            self.__simulation.runInfoDict['batchSize'] = fullGroupCount

        #then give each index a separate file.
        nodeCommand = self.__simulation.runInfoDict["NodeParameter"]+" %BASE_WORKING_DIR%/node_%INDEX% "
      else:
        #If only one batch just use original node file
        nodeCommand = self.__simulation.runInfoDict["NodeParameter"]+" "+nodefile
    else:
      #Not in PBS, so can't look at PBS_NODEFILE and none supplied in input
      newBatchsize = self.__simulation.runInfoDict['batchSize']
      numMPI = self.__simulation.runInfoDict['NumMPI']
      #TODO, we don't have a way to know which machines it can run on
      # when not in PBS so just distribute it over the local machine:
      nodeCommand = " "

    #Disable MPI processor affinity, which causes multiple processes
    # to be forced to the same thread.
    os.environ["MV2_ENABLE_AFFINITY"] = "0"

    # Create the mpiexec pre command
    # Note, with defaults the precommand is "mpiexec -f nodeFile -n numMPI"
    self.__simulation.runInfoDict['precommand'] = self.__simulation.runInfoDict["MPIExec"]+" "+nodeCommand+" -n "+str(numMPI)+" "+self.__simulation.runInfoDict['precommand']
    if(self.__simulation.runInfoDict['NumThreads'] > 1):
      #add number of threads to the post command.
      self.__simulation.runInfoDict['postcommand'] = " --n-threads=%NUM_CPUS% "+self.__simulation.runInfoDict['postcommand']
    self.raiseAMessage("precommand: "+self.__simulation.runInfoDict['precommand']+", postcommand: "+self.__simulation.runInfoDict['postcommand'])

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
      elif child.tag.lower() == "runqsub":
        self.__runQsub = True
      elif child.tag.lower() == "nosplitnode":
        self.__noSplitNode = True
        self.__maxOnNode = child.attrib.get("maxOnNode",None)
        if self.__maxOnNode is not None:
          self.__maxOnNode = int(self.__maxOnNode)
        if "noOverlap" in child.attrib:
          self.__noOverlap = True
      elif child.tag.lower() == "limitnode":
        self.__limitNode = True
        self.__maxOnNode = child.attrib.get("maxOnNode",None)
        if self.__maxOnNode is not None:
          self.__maxOnNode = int(self.__maxOnNode)
        else:
          self.raiseAnError(IOError, "maxOnNode must be specified with LimitNode")
        if "noOverlap" in child.attrib and child.attrib["noOverlap"].lower() in utils.stringsThatMeanTrue():
          self.__noOverlap = True
      else:
        self.raiseAWarning("We should do something with child "+str(child))
