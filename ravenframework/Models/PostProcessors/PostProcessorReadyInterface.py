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
from ... import Files
from ...utils import InputData, InputTypes
from ...DataObjects import DataObject
from ...Models.PostProcessors import PostProcessorInterface
#Internal Modules End-----------------------------------------------------------

class PostProcessorReadyInterface(PostProcessorInterface):
  """
    This class represents a specialized class from which each new PostProcessor must inherit from
    It is a temporary class for the transition of PostProcessor to use Dataset as in and out operations
    When all PostProcessors are converted to use this class as parent class. This class will be
    merged with "PostProcessorInterface" class to form the new base class for all PostProcessors
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
    self._keepInputMeta = False  # Meta keys from input data objects will be added to output data objects

  def keepInputMeta(self, keep=False):
    """
      Method to set the status of "self._keepInputMeta"
      @ In, keep, bool, If True, the meta keys from input data objects will be added to output data objects
      @ Out, None
    """
    self._keepInputMeta = keep

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
    if self._keepInputMeta:
      ## add meta keys from input data objects
      for inputObj in inputs:
        if isinstance(inputObj, DataObject.DataObject):
          metaKeys = inputObj.getVars('meta')
          self.addMetaKeys(metaKeys)

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  def createPostProcessorInput(self, inputObjs, **kwargs):
    """
      This function is used to convert internal DataObjects to user-friendly format of data.
      The output from this function will be directly passed to the "run" method.
      @ In, inputObjs, list, list of DataObjects
      @ In, **kwargs, dict, is a dictionary that contains the information passed by "Step".
          Currently not used by PostProcessor. It can be useful by Step to control the input
          and output of the PostProcessor, as well as other control options for the PostProcessor
      @ Out, inputDict, dict, dictionary of data that will be directly used by the "PostProcessor.run" method.
          inputDict = {'Data':listData, 'Files':listOfFiles},
          listData has the following format if 'xrDataset' is passed to self.setInputDataType('xrDataset')
          (listOfInputVars, listOfOutVars, xr.Dataset)
          Otherwise listData has the following format: (listOfInputVars, listOfOutVars, DataDict) with
          DataDict is a dictionary that has the format
              dataDict['dims']     = dict {varName:independentDimensions}
              dataDict['metadata'] = dict {metaVarName:metaVarValue}
              dataDict['type'] = str TypeOfDataObject
              dataDict['inpVars'] = list of input variables
              dataDict['outVars'] = list of output variables
              dataDict['numberRealization'] = int SizeOfDataObject
              dataDict['name'] = str DataObjectName
              dataDict['metaKeys'] = list of meta variables
              dataDict['data'] = dict {varName: varValue(1-D or 2-D numpy array)}
    """
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
