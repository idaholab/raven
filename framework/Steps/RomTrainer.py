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
  RomTrainer module
  This module contains the Step for Training ROMs
  Created on May 6, 2021
  @author: alfoa
  supercedes Steps.py from alfoa (2/16/2013)
"""
#External Modules------------------------------------------------------------------------------------
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Step import Step
from ..Models import ROM
#Internal Modules End--------------------------------------------------------------------------------

class RomTrainer(Step):
  """
    This step type is used only to train a ROM
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'STEP ROM TRAINER'

  def _localInputAndCheckParam(self,paramInput):
    """
      Place here specialized reading, input consistency check and
      initialization of what will not change during the whole life of the object
      @ In, paramInput, ParameterInput, node that represents the portion of the input that belongs to this Step class
      @ Out, None
    """
    if [item[0] for item in self.parList].count('Input')!=1:
      self.raiseAnError(IOError,'Only one Input and only one is allowed for a training step. Step name: '+str(self.name))
    if [item[0] for item in self.parList].count('Output')<1:
      self.raiseAnError(IOError,'At least one Output is need in a training step. Step name: '+str(self.name))
    for item in self.parList:
      if item[0]=='Output' and item[2] not in ['ROM']:
        self.raiseAnError(IOError,'Only ROM output class are allowed in a training step. Step name: '+str(self.name))

  def _localGetInitParams(self):
    """
      Place here a specialization of the exporting of what in the step is added to the initial parameters
      the printing format of paramDict is key: paramDict[key]
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    return {}

  def _localInitializeStep(self,inDictionary):
    """
      This is the API for the local initialization of the children classes of step
      The inDictionary contains the instances for each possible role supported in the step (dictionary keywords) the instances of the objects in list if more than one is allowed
      The role of _localInitializeStep is to call the initialize method instance if needed
      Remember after each initialization to put:
      self.raiseADebug('for the role "+key+" the item of class '+inDictionary['key'].type+' and name '+inDictionary['key'].name+' has been initialized')
      @ In, inDictionary, dict, the initialization dictionary
      @ Out, None
    """
    for output in inDictionary['Output']:
      if isinstance(output, ROM):
        output.initialize(inDictionary['jobHandler'].runInfoDict,inDictionary['Input'])

  def _localTakeAstepRun(self,inDictionary):
    """
      This is the API for the local run of a step for the children classes
      @ In, inDictionary, dict, contains the list of instances (see Simulation)
      @ Out, None
    """
    #Train the ROM... It is not needed to add the trainingSet since it's already been added in the initialization method
    for ROM in inDictionary['Output']:
      ROM.train(inDictionary['Input'][0])
