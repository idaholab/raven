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

#External Modules---------------------------------------------------------------
import abc
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PluginBase import PluginBase
from Models.PostProcessors.PostProcessor import PostProcessor
#Internal Modules End-----------------------------------------------------------

class PostProcessorPluginBase(PostProcessor, PluginBase):
  """
    This class represents a specialized class from which each PostProcessor plugins must inherit from
  """
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin. This list needs to be populated by the derived class
  _methodsToCheck = ['getInputSpecification', 'handlePluginInput', 'initializePlugin', 'runPlugin']
  entityType = 'PostProcessor'

  ##################################################
  # Methods to link internal PostProcessor methods
  ##################################################
  def __init__(self, runInfoDict):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    PluginBase.__init__(self)
    PostProcessor.__init__(self, runInfoDict)

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    PostProcessor._handleInput(self, paramInput)
    self.handlePluginInput(paramInput)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)
    ### original "inputs" will be processed to generate certain type of inputs for users
    self.initializePlugin(inputs)

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In,  inputIn, object, object containing the data to process.
      Should avoid to use (inputToInternal output), and passing xarray directly/dataset
      Possible inputs include: dict, xarray.Dataset, pd.DataFrame
      @ Out, dict, xarray.Dataset, pd.DataFrame --> I think we can avoid collectoutput in the plugin pp
    """
    self.runPlugin(inputIn)

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
    inputSpecification = super(PostProcessorPluginBase, cls).getInputSpecification()
    return inputSpecification


  @abc.abstractmethod
  def handlePluginInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    pass

  def initializePlugin(self, inputs):
    """
      this needs to be over written if a re initialization of the model is need it gets called at every beginning of a step
      after this call the next one will be run
      @ In, inputs, type TBD
    """
    pass

  @abc.abstractmethod
  def runPlugin(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In,  inputIn, object, object containing the data to process.
      Should avoid to use (inputToInternal output), and passing xarray directly/dataset
      Possible inputs include: dict, xarray.Dataset, pd.DataFrame
      @ Out, dict, xarray.Dataset, pd.DataFrame --> I think we can avoid collectoutput in the plugin pp
    """
    pass
