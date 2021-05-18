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
Created on Nov 1, 2017

@author: dan maljovec, mandd
"""
#Internal Modules---------------------------------------------------------------
from utils import InputData, InputTypes
from PluginBaseClasses.PostProcessorPluginBase import PostProcessorPluginBase
from .ETStructure import ETStructure
#Internal Modules End-----------------------------------------------------------

class ETImporter(PostProcessorPluginBase):
  """
    This is the base class of the PostProcessor that imports Event-Trees (ETs) into RAVEN as a PointSet
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag  = 'POSTPROCESSOR ET IMPORTER'
    self.expand    = None  # option that controls the structure of the ET. If True, the tree is expanded so that
                           # all possible sequences are generated. Sequence label is maintained according to the
                           # original tree
    self.fileFormat = None # chosen format of the ET file
    self.allowedFormats = ['OpenPSA'] # ET formats that are supported
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject
    ## Currently, we have used both DataObject.addRealization and DataObject.load to
    ## collect the PostProcessor returned outputs. DataObject.addRealization is used to
    ## collect single realization, while DataObject.load is used to collect multiple realizations
    ## However, the DataObject.load can not be directly used to collect single realization
    self.outputMultipleRealizations = True

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
    inputSpecification.addSub(InputData.parameterInputFactory("fileFormat", contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory("expand"    , contentType=InputTypes.BoolType))
    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    fileFormat = paramInput.findFirst('fileFormat')
    self.fileFormat = fileFormat.value
    if self.fileFormat not in self.allowedFormats:
      self.raiseAnError(IOError, 'ETImporter Post-Processor ' + self.name + ', format ' + str(self.fileFormat) + ' : is not supported')
    expand = paramInput.findFirst('expand')
    self.expand = expand.value

  def run(self, inputIn):
    """
      This method executes the PostProcessor action.
      @ In,  inputIn, dict, dictionary contains the input data and input files, i.e.,
          {'Data':[DataObjects.asDataset('dict')], 'Files':[FileObject]}, only 'Files'
          will be used by this PostProcessor
      @ Out, outputDict, dict, dictionary of outputs, i.e.,
          {'data':dict of realizations, 'dim':{varName:independent dimensions that the variable depends on}}
    """
    eventTreeModel = ETStructure(self.expand, inputIn['Files'])
    outputDict, variables = eventTreeModel.returnDict()
    outputDict = {'data': outputDict, 'dims':{}}
    return outputDict
