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

#Internal Modules---------------------------------------------------------------
import MessageHandler
from utils import utils
from .FTgate import FTgate
#Internal Modules End-----------------------------------------------------------

#External Modules---------------------------------------------------------------
import numpy as np
import xml.etree.ElementTree as ET
import copy
import itertools
from collections import OrderedDict
#External Modules End-----------------------------------------------------------

class FTstructure():
  """
    This is the base class of the FT structure which actually handles FT structures which is used by the FTimporter and the FTmodel
  """
  def __init__(self, inputs, topEventID):
    """
      This method executes the postprocessor action.
      @ In,  inputs, list, list of file objects
      @ Out, outcome, dict, dict containing the processed FT
    """
    self.basicEvents = []
    self.houseEvents = {}
    self.gateList    = {}
    self.gateID      = []
    self.topEventID  = topEventID

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
          raise IOError('FTImporterPostProcessor Post-Processor ' + self.name + ': house event ' + str(basicEvent.get('name')) + ' has a not boolean value (True or False)')
        self.houseEvents[houseEvent.get('name')] = value

    if not self.topEventID in self.gateID:
      raise IOError('FTImporterPostProcessor: specified top event ' + str(self.topEventID) + ' is not contained in the fault-tree; available gates are: ' + str(self.gateID))

  def returnDict(self):
    """
      This method returns
      @ In,  None
      @ Out, outcome, dict, dictionary containing
    """
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
          raise IOError('FTImporterPostProcessor Post-Processor ' + self.name + ': the provided FT cannot be computed')
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
