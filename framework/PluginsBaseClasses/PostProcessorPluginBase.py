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
import Files
from utils import InputData, InputTypes
from DataObjects import DataObject
from Databases import Database
from .PluginBase import PluginBase
from Models.PostProcessors.PostProcessor import PostProcessor
#Internal Modules End-----------------------------------------------------------

class PostProcessorPluginBase(PostProcessor, PluginBase):
  """
    This class represents a specialized class from which each PostProcessor plugins must inherit from
  """
  # List containing the methods that need to be checked in order to assess the
  # validity of a certain plugin. This list needs to be populated by the derived class
  _methodsToCheck = ['getInputSpecification', '_handleInput']
  entityType = 'PostProcessor'

  ##################################################
  # Methods for Internal Use
  ##################################################
  def createNewInput(self,inputObjs,samplerType,**kwargs):
    """
      This function is used to convert internal DataObjects to user-friendly format of data.
      The output from this function will be directly passed to the "run" method.
      @ In, inputObjs, list, list of DataObjects
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input.
          Not used for PostProcessor, and "None" is used during "Step" "PostProcess" handling
      @ In, **kwargs, dict, is a dictionary that contains the information coming from the sampler,
          a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}.
          Not used for PostProcessor, and {'SampledVars':{'prefix':'None'}, 'additionalEdits':{}}
          is used during "Step" "PostProcess" handling
      @ Out, inputDs, list, list of data set that will be directly used by the "PostProcessor.run" method.
    """
    #### TODO: This method probably need to move to PostProcessor Base Class when we have converted
    #### all internal PostProcessors to use Dataset

    ## Type 1: DataObjects => Dataset
    ## Type 2: File => File
    ## Type 3: HDF5 => ?
    assert type(inputObjs) == list
    inputDs = []
    for inp in inputObjs:
      if isinstance(inp, Files.File):
        inputDs.append(inp)
      elif isinstance(inp, DataObject.DataObject):
        # convert to xarray.Dataset
        inputDs.append(inp.asDataset())
      elif isinstance(inp, Database):
        self.raiseAnError(IOError, "Database", inp.name, "can not be handled directly by this Post Processor")
      else:
        self.raiseAnError(IOError, "Unknown input is found", str(inp))


  ##################################################
  # Plugin APIs
  ##################################################
  def __init__(self, runInfoDict):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    PluginBase.__init__(self)
    PostProcessor.__init__(self, runInfoDict)

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

  def initialize(self, runInfo, inputs, initDict=None):
    """
      This function is used to initialize the plugin, i.e. set up working dir,
      call the initializePlugin method from the plugin
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    PostProcessor._handleInput(self, paramInput)

  ### "run" is required for each specific PostProcessor, it is an abstractmethod in
  ### PostProcessor base class.
  # def run(self, inputDs):
  #   """
  #     This method executes the postprocessor action.
  #     @ In,  inputDs, list, list of Datasets
  #     @ Out, outputDs, dict, xarray.Dataset, pd.DataFrame
  #       --> I think we can avoid collectoutput in the plugin pp
  #   """
