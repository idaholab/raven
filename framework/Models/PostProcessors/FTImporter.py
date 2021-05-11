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
#Internal Modules---------------------------------------------------------------
from utils import InputData, InputTypes
from PluginBaseClasses.PostProcessorPluginBase import PostProcessorPluginBase
from .FTStructure import FTStructure
#Internal Modules End-----------------------------------------------------------

class FTImporter(PostProcessorPluginBase):
  """
    This is the base class of the postprocessor that imports Fault-Trees (FTs) into RAVEN as a PointSet
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
    fileAllowedFormats = InputTypes.makeEnumType("FTFileFormat", "FTFileFormatType", ["OpenPSA"])
    inputSpecification.addSub(InputData.parameterInputFactory("fileFormat", contentType=fileAllowedFormats))
    inputSpecification.addSub(InputData.parameterInputFactory("topEventID", contentType=InputTypes.StringType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR FT IMPORTER'
    self.FTFormat = None # chosen format of the FT file
    self.topEventID = None
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject
    ## Currently, we have used both DataObject.addRealization and DataObject.load to
    ## collect the PostProcessor returned outputs. DataObject.addRealization is used to
    ## collect single realization, while DataObject.load is used to collect multiple realizations
    ## However, the DataObject.load can not be directly used to collect single realization
    self.outputMultipleRealizations = True

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    fileFormat = paramInput.findFirst('fileFormat')
    self.fileFormat = fileFormat.value
    topEventID = paramInput.findFirst('topEventID')
    self.topEventID = topEventID.value

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In,  inputIn, dict, dictionary contains the input data and input files, i.e.,
          {'Data':[DataObjects.asDataset('dict')], 'Files':[FileObject]}, only 'Files'
          will be used by this PostProcessor
      @ Out, outputDict, dict, dictionary of outputs, i.e.,
          {'data':dict of realizations, 'dim':{varName:independent dimensions that the variable depends on}}
    """
    faultTreeModel = FTStructure(inputIn['Files'], self.topEventID)
    outputDict = faultTreeModel.returnDict()
    outputDict = {'data': outputDict, 'dims':{}}
    return outputDict
