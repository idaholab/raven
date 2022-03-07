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
Created on March 10, 2021

@author: wangc
"""

#Internal Modules---------------------------------------------------------------
from .. import Files
from ..utils import InputData, InputTypes
from ..DataObjects import DataObject
from .PluginBase import PluginBase
from ..Models.PostProcessors import PostProcessorReadyInterface
from ..Models.PostProcessors import factory
#Internal Modules End-----------------------------------------------------------

class PostProcessorPluginBase(PluginBase, PostProcessorReadyInterface):
  """
    This class represents a specialized class from which each PostProcessor plugins must inherit from
  """
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin.
  _methodsToCheck = ['getInputSpecification', '_handleInput', 'run']
  entityType = 'PostProcessor'
  _interfaceFactory = factory
  ##################################################
  # Plugin APIs
  ##################################################
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
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.setInputDataType('dict') # Current accept two types: 1) 'dict', 2) 'xrDataset'
                                  # Set default to 'dict', this is consistent with current post-processors
    self.keepInputMeta(True)      # Meta keys from input data objects will be added to output data objects

  def initialize(self, runInfo, inputs, initDict=None):
    """
      This function is used to initialize the plugin, i.e. set up working dir,
      call the initializePlugin method from the plugin
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    super().initialize(runInfo, inputs, initDict)

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  ### "run" is required for each specific PostProcessor, it is an abstractmethod in
  ### PostProcessorInterface base class.
  # def run(self, inputDs):
  #   """
  #     This method executes the postprocessor action.
  #     @ In,  inputDs, list, list of Datasets
  #     @ Out, outputDs, dict, xarray.Dataset
  #   """
