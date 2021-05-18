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
'''
Created on December 1, 2015

'''
import copy
import numpy as np

from PluginBaseClasses.PostProcessorPluginBase import PostProcessorPluginBase
from utils import InputData, InputTypes

class testInterfacedPP(PostProcessorPluginBase):
  """ This class represents the most basic interfaced post-processor
      This class inherits form the base class PostProcessorInterfaceBase and it contains the three methods that need to be implemented:
      - initialize
      - run
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super().getInputSpecification()
    inputSpecification.addSubSimple("xmlNodeExample", InputTypes.StringType)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.setInputDataType('dict')
    self.keepInputMeta(True)
    self.outputMultipleRealizations = True # True indicate multiple realizations are returned
    self.validDataType = ['HistorySet'] # The list of accepted types of DataObject

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs)>1:
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only one dataObject')
    if inputs[0].type != 'HistorySet':
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only HistorySet dataObject, but got "{}"'.format(inputs[0].type))

  def run(self,inputIn):
    """
     This method is transparent: it passes the inputDic directly as output
     @ In, inputIn, dict, dictionary which contains the data inside the input DataObject
     @ Out, outputDict, dict, the output dictionary, passing through HistorySet info
    """
    _, _, inputDict = inputIn['Data'][0]
    outputDict = {'data':{}}
    outputDict['dims'] = copy.deepcopy(inputDict['dims'])
    for key in inputDict['data'].keys():
      outputDict['data'][key] = copy.deepcopy(inputDict['data'][key])

    # add meta variables back
    for key in inputDict['metaKeys']:
      outputDict['data'][key] = inputDict['data'][key]
    return outputDict

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == 'xmlNodeExample':
        self.xmlNodeExample = child.value
