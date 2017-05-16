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

import Simulation

class MPISimulationMode(Simulation.SimulationMode):
  """
    MPISimulationMode is a specialized class of SimulationMode.
    It is aimed to distribute the runs using the MPI protocol
  """
  def __init__(self,messageHandler):
    """
      Constructor
      @ In, simulation, instance, instance of the simulation class
      @ Out, None
    """
    Simulation.SimulationMode.__init__(self,messageHandler)
    self.messageHandler = messageHandler
    #Figure out if we are in PBS
    self.__inPbs = "PBS_NODEFILE" in os.environ
    self.__nodefile = False
    self.__runQsub = False
    self.__noSplitNode = False #If true, don't split mpi processes across nodes
    self.__limitNode = False #If true, fiddle with max on Node
    self.__maxOnNode = None #Used with __noSplitNode and __limitNode to limit number on a node
    self.__noOverlap = False #Used with __limitNode to prevent multiple batches from being on one node
    # If (__noSplitNode or __limitNode) and  __maxOnNode is not None,
    # don't put more than that on on single shared memory node
    self.printTag = 'MPI SIMULATION MODE'

  def modifySimulation(self, runInfoDict):
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
      #XXX
      runInfoDict['Nodes'] = list(lines)
      numMPI = runInfoDict['NumMPI']
      oldBatchsize = runInfoDict['batchSize']
      #the batchsize is just the number of nodes of which there is one
      # per line in the nodefile divided by the numMPI (which is per run)
      # and the floor and int and max make sure that the numbers are reasonable
      maxBatchsize = max(int(math.floor(len(lines)/numMPI)),1)
      if maxBatchsize < oldBatchsize:
        #XXX
        runInfoDict['batchSize'] = maxBatchsize
        self.raiseAWarning("changing batchsize from "+str(oldBatchsize)+" to "+str(maxBatchsize)+" to fit on "+str(len(lines))+" processors")
      newBatchsize = runInfoDict['batchSize']
      if newBatchsize > 1:
        #need to split node lines so that numMPI nodes are available per run
        workingDir = runInfoDict['WorkingDir']
        if not (self.__noSplitNode or self.__limitNode):
          for i in range(newBatchsize):
            nodeFile = open(os.path.join(workingDir,"node_"+str(i)),"w")
            for line in lines[i*numMPI:
              (i+1)*numMPI]:
              nodeFile.write(line)
            nodeFile.close()
        else:
          #self.__noSplitNode == True or self.__limitNode == True
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
          if fullGroupCount != runInfoDict['batchSize']:
            self.raiseAWarning("changing batchsize to "+str(fullGroupCount)+" because NoSplitNode is "+str(self.__noSplitNode)+" and LimitNode is "+str(self.__limitNode)+" and some nodes could not be used.")
            #XXX
            runInfoDict['batchSize'] = fullGroupCount

        #then give each index a separate file.
        nodeCommand = runInfoDict["NodeParameter"]+" %BASE_WORKING_DIR%/node_%INDEX% "
      else:
        #If only one batch just use original node file
        nodeCommand = runInfoDict["NodeParameter"]+" "+nodefile
    else:
      #Not in PBS, so can't look at PBS_NODEFILE and none supplied in input
      newBatchsize = runInfoDict['batchSize']
      numMPI = runInfoDict['NumMPI']
      #TODO, we don't have a way to know which machines it can run on
      # when not in PBS so just distribute it over the local machine:
      nodeCommand = " "

    #Disable MPI processor affinity, which causes multiple processes
    # to be forced to the same thread.
    os.environ["MV2_ENABLE_AFFINITY"] = "0"

    # Create the mpiexec pre command
    # Note, with defaults the precommand is "mpiexec -f nodeFile -n numMPI"
    #XXX
    runInfoDict['precommand'] = runInfoDict["MPIExec"]+" "+nodeCommand+" -n "+str(numMPI)+" "+runInfoDict['precommand']
    if(runInfoDict['NumThreads'] > 1):
      #add number of threads to the post command.
      #XXX
      runInfoDict['postcommand'] = " --n-threads=%NUM_CPUS% "+runInfoDict['postcommand']
    self.raiseAMessage("precommand: "+runInfoDict['precommand']+", postcommand: "+runInfoDict['postcommand'])

  def doOverrideRun(self, runInfoDict):
    """
      If doOverrideRun is true, then use runOverride instead of
      running the simulation normally.  This method should call
      simulation.run
      @ In, runInfoDict, dict, the run info dict
      @ Out, doOverrRun, bool, does the override?
    """
    # Check if the simulation has been run in PBS mode and if run QSUB
    # has been requested, in case, construct the proper command
    doOverrRun = (not self.__inPbs) and self.__runQsub
    return doOverrRun

  def runOverride(self, runInfoDict):
    """
      This  method completely overrides the Simulation's run method
      @ In, runInfoDict, dict, the run information
      @ Out, None
    """
    #Check and see if this is being accidently run
    assert self.__runQsub and not self.__inPbs
    Simulation.createAndRunQSUB(runInfoDict)

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
        self.raiseADebug("We should do something with child "+str(child))
