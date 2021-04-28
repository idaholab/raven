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
import Files
from utils import InputData, InputTypes
from DataObjects import DataObject
from .PluginBase import PluginBase
from Models.PostProcessors.PostProcessorInterface import PostProcessorInterface
#Internal Modules End-----------------------------------------------------------

class PostProcessorPluginBase(PostProcessorInterface, PluginBase):
  """
    This class represents a specialized class from which each PostProcessor plugins must inherit from
  """
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin.
  _methodsToCheck = ['getInputSpecification', '_handleInput', 'run']
  entityType = 'PostProcessor'

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
    self._inputDataType = 'dict' # Current accept two types: 1) 'dict', 2) 'xrDataset'
                                 # Set default to 'dict', this is consistent with current post-processors

  def setInputDataType(self, dataType='dict'):
    """
      Method to set the input data type that will be passed to "run" method
      @ In, dataType, str, the data type to which the internal DataObjects will be converted
      @ Out, None
    """
    if dataType not in ['dict', 'xrDataset']:
      self.raiseAnError(IOError, 'The dataType "{}" is not supported, please consider using "dict" or "xrDataset"'.format(dataType))
    self._inputDataType = dataType

  def getInputDataType(self):
    """
      Method to retrieve the input data type to which the internal DataObjects will be converted
      @ In, None
      @ Out, _inputDataType, str, the data type, i.e., 'dict', 'xrDataset'
    """
    return self._inputDataType

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

    ##################################################
    # Methods for Internal Use
    ##################################################

  def createPostProcessorInput(self, inputObjs, **kwargs):
    """
      This function is used to convert internal DataObjects to user-friendly format of data.
      The output from this function will be directly passed to the "run" method.
      @ In, inputObjs, list, list of DataObjects
      @ In, **kwargs, dict, is a dictionary that contains the information passed by "Step".
          Currently not used by PostProcessor. It can be useful by Step to control the input
          and output of the PostProcessor, as well as other control options for the PostProcessor
      @ Out, inputDict, list, list of data set that will be directly used by the "PostProcessor.run" method.
    """
    #### TODO: This method probably need to move to PostProcessor Base Class when we have converted
    #### all internal PostProcessors to use Dataset
    ## Type 1: DataObjects => Dataset or Dict
    ## Type 2: File => File
    assert type(inputObjs) == list
    inputDict = {'Data':[], 'Files':[]}
    for inp in inputObjs:
      if isinstance(inp, Files.File):
        inputDict['Files'].append(inp)
      elif isinstance(inp, DataObject.DataObject):
        dataType = self.getInputDataType()
        data = inp.asDataset(outType=dataType)
        inpVars = inp.getVars('input')
        outVars = inp.getVars('output')
        inputDict['Data'].append((inpVars, outVars, data))
      else:
        self.raiseAnError(IOError, "Unknown input is found", str(inp))
    return inputDict
