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
Created on Dec 21, 2017

@author: mandd

"""

from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import xml.etree.ElementTree as ET
import copy
import itertools
from collections import OrderedDict
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import InputData
from utils import xmlUtils as xmlU
from utils import utils
from .ETImporter import ETImporter
import Files
import Runners
#Internal Modules End-----------------------------------------------------------

class FTgate:
  def __init__(self,xmlNode):
    """
      Method that initializes the gate
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.name         = None
    self.gate         = None
    self.arguments    = []
    self.negations    = []
    self.params       = {}
    self.allowedGates = {'not':1,'and':'inf','or':'inf','xor':'inf','iff':2,'nand':'inf','nor':'inf','atleast':'inf','cardinality':'inf','imply':2}

    self.name = xmlNode.get('name')

    for child in xmlNode:
      if child.attrib:
        self.params = child.attrib
      if child.tag in self.allowedGates.keys():
        self.gate = child.tag
      else: 
        self.raiseAnError(IOError, 'FTImporterPostProcessor Post-Processor; gate ' + str(child.tag) + ' : is not recognized. Allowed gates are: '+ str(self.allowedGates.keys()))

    for node in findAllRecursive(xmlNode, 'gate'):
      self.arguments.append(node.get('name'))

    for node in findAllRecursive(xmlNode, 'basic-event'):
      self.arguments.append(node.get('name'))

    for node in findAllRecursive(xmlNode, 'house-event'):
      self.arguments.append(node.get('name'))
      
    for child in xmlNode:
      for childChild in child:
        if childChild.tag == 'not':
          event = list(childChild.iter())
          if len(event)>2:
              self.raiseAnError(IOError, 'FTImporterPostProcessor Post-Processor; gate ' + str(self.name) + ' contains a negations of multiple basic events')
          elif event[1].tag in ['gate','basic-event','house-event']:
            self.negations.append(event[1].get('name'))          
        
    if self.gate in ['iff'] and len(self.arguments)>self.allowedGates['iff']:
      self.raiseAnError(IOError, 'FTImporterPostProcessor Post-Processor; iff gate ' + str(self.name) + ' has more than 2 events')
    if self.gate in ['imply'] and len(self.arguments)>self.allowedGates['imply']:
      self.raiseAnError(IOError, 'FTImporterPostProcessor Post-Processor; imply gate ' + str(self.name) + ' has more than 2 events')

  def returnArguments(self):
    """
      Method that returns the arguments of the gate
      @ In, None
      @ Out, self.arguments, list, list that contains the arguments of the gate
    """
    return self.arguments    

  def evaluate(self,argValues):
    """
      Method that evaluates the gate 
      @ In, argValues, dict, dictionary containing all available variables
      @ Out, outcome, float, calculated outcome of the gate
    """  
    argumentValues = copy.deepcopy(argValues)
    for key in self.negations:
        if argumentValues[key]==1:
          argumentValues[key]=0
        else:
          argumentValues[key]=1
    if set(self.arguments) <= set(argumentValues.keys()):
      argumentsToPass = OrderedDict()
      for arg in self.arguments:
        argumentsToPass[arg] = argumentValues[arg]
      outcome = self.evaluateGate(argumentsToPass)
      return outcome
    else:
      self.raiseAnError(IOError, 'FTImporterPostProcessor Post-Processor; gate ' + str(self.name) + ' can receive these arguments ' + str(self.arguments) + ' but the following were passed ' + str(argumentValues.keys()) )

  def evaluateGate(self,argumentValues):
    """
      Method that evaluates the gate.
      Note that argumentValues is passed directly (instead of argumentValues.values()) in case of the imply gate since the events IDs are important
      @ In, argumentValues, OrderedDict, dictionary containing only the variables of interest to the gate
      @ Out, outcome, float, calculated outcome of the gate
    """ 
    if self.gate == 'and':
      outcome = ANDgate(argumentValues.values())
    elif self.gate == 'or':
      outcome = ORgate(argumentValues.values())
    elif self.gate == 'nor':
      outcome = NORgate(argumentValues.values())
    elif self.gate == 'nand':
      outcome = NANDgate(argumentValues.values())
    elif self.gate == 'xor':
      outcome = XORgate(argumentValues.values())
    elif self.gate == 'iff':
      outcome = IFFgate(argumentValues.values())
    elif self.gate == 'atleast':
      outcome = ATLEASTgate(argumentValues.values(),float(self.params['min']))
    elif self.gate == 'cardinality':
      outcome = CARDINALITYgate(argumentValues.values(),float(self.params['min']),float(self.params['max']))
    elif self.gate == 'imply':
      outcome = IMPLYgate(argumentValues)
    elif self.gate == 'not':
      outcome = NOTgate(argumentValues.values())
    return outcome

def NOTgate(value):
  """
    Method that evaluates the NOT gate 
    @ In, value, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  if len(value)>1:
    self.raiseAnError(IOError, 'NOT gate has received in input ' + str(len(value)) + ' values instead of 1.')
  if value[0]==0:
    outcome = 1
  else:
    outcome = 0
  return outcome 

def ANDgate(argumentValues):
  """
    Method that evaluates the AND gate 
    @ In, argumentValues, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  if 0 in argumentValues:
    outcome = 0
  else:
    outcome = 1
  return outcome

def ORgate(argumentValues):
  """
    Method that evaluates the OR gate 
    @ In, argumentValues, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  if 1 in argumentValues:
    outcome = 1
  else:
    outcome = 0
  return outcome

def NANDgate(argumentValues):
  """
    Method that evaluates the NAND gate 
    @ In, argumentValues, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  out = []
  out.append(ANDgate(argumentValues))
  outcome = NOTgate(out)
  return outcome

def NORgate(argumentValues):
  """
    Method that evaluates the NOR gate 
    @ In, argumentValues, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """   
  out = []
  out.append(ORgate(argumentValues))
  outcome = NOTgate(out)
  return outcome

def XORgate(argumentValues):
  """
    Method that evaluates the XOR gate 
    The XOR gate gives a true (1 or HIGH) output when the number of true inputs is odd.
    https://electronics.stackexchange.com/questions/93713/how-is-an-xor-with-more-than-2-inputs-supposed-to-work
    @ In, argumentValues, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  if argumentValues.count(1.) % 2 != 0:
    outcome = 1
  else:
    outcome = 0
  return outcome

def IFFgate(argumentValues):
  """
    Method that evaluates the IFF gate 
    @ In, argumentValues, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  out = []
  out.append(XORgate(argumentValues))
  outcome = NOTgate(out)
  return outcome

def IMPLYgate(argumentValues):
  """
    Method that evaluates the IMPLY gate 
    Note that this gate requires a specific definition of the two inputs. This definition is specifed in the order of the events provided in the input file
    @ In, argumentValues, list, list of values
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  keys = argumentValues.keys()
  if argumentValues[keys[0]]==1 and argumentValues[keys[1]]==0:
    outcome = 0
  else:
    outcome = 1
  return outcome

def ATLEASTgate(argumentValues,k):
  """
    Method that evaluates the ATLEAST gate 
    @ In, argumentValues, list, list of values
    @ In, k, float, max number of allowed events
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  if argumentValues.count(1) >= k:
    outcome = 1
  else:
    outcome = 0
  return outcome

def CARDINALITYgate(argumentValues,l,h):
  """
    Method that evaluates the CARDINALITY gate 
    @ In, argumentValues, list, list of values
    @ In, l, float, min number of allowed events
    @ In, h, float, max number of allowed events
    @ Out, outcome, float, calculated outcome of the gate
  """ 
  nOcc = argumentValues.count(1)
  if nOcc >= l and nOcc <= h:
    outcome = 1
  else:
    outcome = 0
  return outcome

class FTImporter(PostProcessor):
  """
    This is the base class of the postprocessor that imports Fault-Trees (ETs) into RAVEN as a PointSet
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR FT IMPORTER'
    self.FTFormat = None
    self.allowedFormats = ['OpenPSA']
    
    self.basicEvents = []
    self.houseEvents = {}
    self.gateList    = {}
    self.gateID      = []

    self.topEventID = None

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(FTImporter, cls).getInputSpecification()
    inputSpecification.addSub(InputData.parameterInputFactory("fileFormat", contentType=InputData.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("topEventID", contentType=InputData.StringType))
    return inputSpecification

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = FTImporter.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    fileFormat = paramInput.findFirst('fileFormat')
    self.fileFormat = fileFormat.value
    if self.fileFormat not in self.allowedFormats:
      self.raiseAnError(IOError, 'FTImporterPostProcessor Post-Processor ' + self.name + ', format ' + str(self.fileFormat) + ' : is not supported')
    
    topEventID = paramInput.findFirst('topEventID')
    self.topEventID = topEventID.value

  def run(self, inputs):
    """
      This method executes the postprocessor action.
      @ In,  inputs, list, list of file objects
      @ Out, out, dict, dict containing the processed FT
    """
    out = self.runOpenPSA(inputs)
    return out

  def runOpenPSA(self, inputs):
    """
      This method executes the postprocessor action.
      @ In,  inputs, list, list of file objects
      @ Out, outcome, dict, dict containing the processed FT
    """
    for file in inputs:
      faultTree = ET.parse(file.getPath() + file.getFilename())
      faultTree = findAllRecursive(faultTree,'opsa-mef')

      for gate in findAllRecursive(faultTree[0], 'define-gate'):
        FTGate = FTgate(gate)
        self.gateList[gate.get('name')] = FTGate
        self.gateID.append(gate.get('name'))
      
      for basicEvent in findAllRecursive(faultTree[0], 'basic-event'):
        self.basicEvents.append(basicEvent.get('name'))
      
      for houseEvent in findAllRecursive(faultTree[0], 'define-house-event'):
        value = houseEvent.find('constant').get('value')
        if value in ['True','true']:
          value = 1.
        elif value in ['False','false']:
          value = 0.
        else:
          self.raiseAnError(IOError,'FTImporterPostProcessor Post-Processor ' + self.name + ': house event ' + str(basicEvent.get('name')) + ' has a not boolean value (True or False)')
        self.houseEvents[houseEvent.get('name')] = value

    if self.topEventID not in self.gateID:
      self.raiseAnError(IOError,'FTImporterPostProcessor: specified top event ' + str(self.topEventID) + ' is not contained in the fault-tree; available gates are: ' + str(self.gateID))
    self.FTsolver()
    outcome = self.constructData()    
    return outcome

  def FTsolver(self):
    """
      This method determines the ordered sequence of gates to compute in order to solve the full FT.
      The determined ordered sequence is stored in self.gateSequence.
      @ In,  None
      @ Out, None
    """
    self.gateSequence = []
    availBasicEvents = copy.deepcopy(self.basicEvents)
    availBasicEvents = availBasicEvents + self.houseEvents.keys()
    counter = 0
    while True:
      complete=False
      for gate in self.gateList.keys():
        if set(self.gateList[gate].returnArguments()) <= set(availBasicEvents):
          self.gateSequence.append(gate)
          availBasicEvents.append(gate)
        if set(availBasicEvents) == set(self.basicEvents+self.gateID+self.houseEvents.keys()):
          complete=True
          break
        if counter > len(self.gateList.keys()):
          self.raiseAnError(IOError,'FTImporterPostProcessor Post-Processor ' + self.name + ': the provided FT cannot be computed')
        counter += 1
      if complete:
        break

  def evaluateFT(self,combination):
    """
      This method determines the outcome of the FT given a set of basic-event values
      @ In,  combination, dict, dictionary containing values for all basic-events
      @ Out, values, dict, dictionary containing calculated values for all gates
    """
    values = {}
    for gate in self.gateSequence:
      values[gate] = self.gateList[gate].evaluate(combination)
      combination[gate] = values[gate]
    return values

  def constructData(self):
    """
      This method determines the outcome of the FT given a set of basic-event values
      @ In,  None
      @ Out, outcome, dict, dictionary containing calculated values for all basic-events and the Top-event
    """
    combinations = list(itertools.product([0,1],repeat=len(self.basicEvents)))
    outcome={}
    outcome={key:np.zeros(len(combinations)) for key in self.basicEvents}
    outcome[self.topEventID] = np.zeros(len(combinations))
    for index,combination in enumerate(combinations):
      combinationDict = {key: combination[index] for index,key in enumerate(self.basicEvents)}
      for houseEvent in self.houseEvents.keys():
        combinationDict[houseEvent] = self.houseEvents[houseEvent]
      out = self.evaluateFT(combinationDict)
      for key in self.basicEvents:
        outcome[key][index]=float(combinationDict[key])
      outcome[self.topEventID][index] = out[self.topEventID]
    return outcome

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object, (DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict = evaluation[1]  
    # Output to file
    if output.type in ['PointSet']:
      for key in output.getParaKeys('inputs'):
        for value in outputDict[key]:
          output.updateInputValue(str(key),value)
      for key in output.getParaKeys('outputs'):
        for value in outputDict[key]:
          output.updateOutputValue(str(key),value)
    else:
        self.raiseAnError(RuntimeError, 'FTImporter failed: Output type ' + str(output.type) + ' is not supported.')    

  def collectOutput_NEW_dataObject(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object, (DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict = evaluation[1]  
    
    
def findAllRecursive(node, element):
  """
    A function for recursively traversing a node in an elementTree to find
    all instances of a tag.
    Note that this method differs from findall() since it goes for all nodes,
    subnodes, subsubnodes etc. recursively
    @ In, node, ET.Element, the current node to search under
    @ In, element, str, the string name of the tags to locate
    @ InOut, result, list, a list of the currently recovered results
  """
  result=[]
  for elem in node.iter(tag=element):
    result.append(elem)
  return result
