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
  _methodsToCheck = ['getInputSpecification', 'handlePluginInput', 'initializePlugin', 'runPluginDataProcessor']
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
      This function is used to initialize the plugin, i.e. set up working dir,
      call the initializePlugin method from the plugin
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)
    self.initializePlugin()

  def _generatePluginInput(self, inputDataObjs):
    """
      convert input data objects into standardized format of data
      (xarray.Dataset, pandas.DataFrame, dict, numpy.array)
      @ In, inputDataObjs, list, list of DataObjects that needs to be converted
      @ Out, inputDs, list, list of current inputs for plugin
    """
    # convert to xarray.Dataset
    inputDs = [inp.asDataset() for inp in inputDataObjs]
    return inputDs

  def run(self, inputDataObjs):
    """
      This method executes the postprocessor action.
      @ In,  inputDataObjs, list, list of DataObjects
      @ Out, outputDs, dict, xarray.Dataset, pd.DataFrame
        --> I think we can avoid collectoutput in the plugin pp
    """
    inputDs = self._generatePluginInput(inputDataObjs)
    outputDs = self.runPluginDataProcessor(inputDs)
    return outputDs

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

  def initializePlugin(self):
    """
      Optional method to initialize the Plugin
    """
    pass

  @abc.abstractmethod
  def runPluginDataProcessor(self, inputDs):
    """
      This method executes the postprocessor action.
      @ In,  inputDs, list, list of current inputs
        (possible inputs include: dict, xarray.Dataset, pd.DataFrame)
      @ Out, outputDs, dict, xarray.Dataset, pd.DataFrame
        --> I think we can avoid collectoutput in the plugin pp
    """
    pass
