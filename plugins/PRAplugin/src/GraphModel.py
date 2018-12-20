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
Created on April 30, 2018

@author: mandd
"""

from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import xml.etree.ElementTree as ET
from utils import utils
from utils import graphStructure as GS
import copy
from utils import xmlUtils as xmlU
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End-----------------------------------------------------------


class GraphModel(ExternalModelPluginBase):
  """
    This class is designed to create a directed graph model which is employed to model Reliability Block Diagrams
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    ExternalModelPluginBase.__init__(self)

  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to GraphModel
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    container.modelFile = None # file name containing the RBD structure
    container.nodesIN   = None # ID of the RBD input nodes
    container.nodesOUT  = None # ID of the RBD output nodes

    container.mapping    = {} # Mapping dictionary for input variables
    container.InvMapping = {} # Inverse mapping dictionary for input variables

    for child in xmlNode:
      if   child.tag == 'nodesIN':
        container.nodesIN  = [str(var.strip()) for var in child.text.split(",")]
      elif child.tag == 'nodesOUT':
        container.nodesOUT = [str(var.strip()) for var in child.text.split(",")]
      elif child.tag == 'modelFile':
        container.modelFile = child.text.strip() + '.xml'
      elif child.tag == 'map':
        container.mapping[child.get('var')]      = child.text.strip()
        container.InvMapping[child.text.strip()] = child.get('var')
      elif child.tag == 'variables':
        variables = [str(var.strip()) for var in child.text.split(",")]
      else:
        print('xml error')

    if container.nodesIN is None:
      raise IOError("GraphModel: <nodesIN> XML block is not specified")
    if container.nodesOUT is None:
      raise IOError("GraphModel: <nodesOUT> XML block is not specified")
    if container.modelFile is None:
      raise IOError("GraphModel: <modelFile> XML block is not specified")

    if set(variables) != set(container.mapping.keys()):
      raise IOError("GraphModel: the set of variables specified in the <variables> " + str(set(variables)) + " XML block does not match with the specified mapping" + str(set(container.mapping.keys())))
    if not set(container.nodesOUT) <= set(container.mapping.values()):
      raise IOError("GraphModel: the set of out variables specified in the <nodesOUT> " + str(set(variables)) + " XML block does not match with the specified mapping" + str(set(container.mapping.values())))

  def initialize(self, container,runInfoDict,inputFiles):
    """
      Method to initialize the GraphModel
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    container.nodes = {}
    container.deg   = {}

    container.runInfo = runInfoDict
    self.createGraph(container,container.modelFile)

  def createGraph(self,container,file):
    """
      Method that actually creates from file the graph structure of the model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, file, file, file containing the structure of the model
      @ Out, None
    """
    graph = ET.parse(container.runInfo['WorkingDir'] + '/' + file)
    graph = xmlU.findAllRecursive(graph,'Graph')

    for node in xmlU.findAllRecursive(graph[0], 'node'):
      nodeName = node.get('name')
      nodeChilds = []
      deg = None
      for child in node:
        if child.tag == 'childs':
          nodeChilds = [var.strip() for var in child.text.split(",")]
        if child.tag == 'deg':
          deg = float(child.text)
      container.nodes[nodeName] = nodeChilds
      container.deg[nodeName]   = deg

  def run(self, container, Inputs):
    """
      This method computes all possible path from the input to the output nodes
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    if self.checkTypeOfAnalysis(container,Inputs):
      dictOUT = self.runTimeDep(container, Inputs)
    else:
      dictOUT = self.runStatic(container, Inputs)

    for var in dictOUT.keys():
      container.__dict__[var] = dictOUT[var]

  def checkTypeOfAnalysis(self,container,Inputs):
    """
      This method checks which type of analysis to be performed:
       - True:  dynamic (time dependent)
       - False: static
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ Out, analysisType, bool, type of analysis to be performed

    """
    arrayValues=set()
    for key in Inputs.keys():
      if key in container.mapping.keys():
        arrayValues.add(Inputs[key][0])
    analysisType = None
    if arrayValues.difference({0.,1.}):
      analysisType = True
    else:
      analysisType = False
    return analysisType

  def runStatic(self, container, Inputs):
    """
      This method performs a static analysis of the graph model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ Out, dictOut, dict, dictionary containing the status of all output variables
    """
    mapping = copy.deepcopy(container.mapping)
    nodes   = copy.deepcopy(container.nodes)

    for key in Inputs.keys():
      if key in mapping.keys():
        if mapping[key] in nodes.keys() and Inputs[key][0] == 1.0:
          nodes.pop(mapping[key],None)
          for node in nodes.keys():
            if mapping[key] in nodes[node]:
              nodes[node].remove(mapping[key])

    ravenGraph = GS.graphObject(nodes)

    dictOut = {}
    for nodeO in container.nodesOUT:
      paths = []
      for nodeI in container.nodesIN:
        paths = paths + ravenGraph.findAllPaths(nodeI,nodeO)
      var = container.InvMapping[nodeO]
      if paths:
        dictOut[var] = np.asarray(0.)
      else:
        dictOut[var] = np.asarray(1.)
    return dictOut

  def runTimeDep(self, container, Inputs):
    """
      This method performs a dynamic analysis of the graph model
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ Out, outcome, dict, dictionary containing the temporal status of all output variables
    """
    times = []
    times.append(0.)
    for key in Inputs.keys():
      if key in container.mapping.keys() and Inputs[key][0]!=1.:
        times.append(Inputs[key][0])
    times = sorted(times, key=float)

    outcome={}
    for var in container.nodesOUT:
      outcome[container.InvMapping[var]] = np.asarray([0.])

    for time in times:
      inputToPass=self.inputToBePassed(container,time,Inputs)
      tempOut = self.runStatic(container, inputToPass)
      for var in tempOut.keys():
        if tempOut[var] == 1.:
          if time == 0.:
            outcome[var] = np.asarray([1.])
          else:
            if outcome[var][0] <= 0:
              outcome[var] = np.asarray([time])
    return outcome

  def inputToBePassed(self,container,time,Inputs):
    """
      This method returns the status of the input variables at time t=time
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN
      @ In, time, float, time at which the input variables need to be evaluated
      @ Out, inputToBePassed, dict, value of the RBD nodes at t=time
    """
    inputToBePassed = {}
    for key in Inputs.keys():
      if key in container.mapping.keys():
        if Inputs[key][0] == 0. or Inputs[key][0] == 1.:
          inputToBePassed[key] = Inputs[key]
        else:
          if Inputs[key][0] > time:
            inputToBePassed[key] = np.asarray([0.])
          else:
            inputToBePassed[key] = np.asarray([1.])
    return inputToBePassed



