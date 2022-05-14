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
Created on April 6, 2021

@author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import copy
import abc
import os
from xarray import Dataset as ds
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...BaseClasses import BaseInterface
from ...utils import InputTypes, InputData
from ... import Runners
#Internal Modules End--------------------------------------------------------------------------------

class PostProcessorInterface(BaseInterface):
  """
    Base class for other postprocessor interfaces (i.e., BasicStatistics, ETImporter).
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
    spec = super().getInputSpecification()
    spec.addParam("subType", InputTypes.StringType, True)
    return spec

  def __init__(self):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    super().__init__()
    self.inputCheckInfo  = []     # List of tuple, i.e input objects info [('name','type')]
    self.action = None            # action
    self.workingDir = ''          # path for working directory
    self.printTag = 'PostProcessorInterface'
    self.outputDataset  = False # True if the user wants to dump the outputs to dataset
    self.validDataType = ['PointSet','HistorySet'] # The list of accepted types of DataObject
    ## Currently, we have used both DataObject.addRealization and DataObject.load to
    ## collect the PostProcessor returned outputs. DataObject.addRealization is used to
    ## collect single realization, while DataObject.load is used to collect multiple realizations
    ## However, the DataObject.load can not be directly used to collect single realization
    ## One possible solution is all postpocessors return a list of realizations, and we only
    ## use addRealization method to add the collections into the DataObjects
    self.outputMultipleRealizations = False

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the PostProcessor
      @ In, runInfo, dict, it is the run info from the jobHandler
      @ In, inputs, list, it is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    super().initialize()
    self.inputs = inputs
    if 'stepName' in runInfo:
      self.workingDir = os.path.join(runInfo['WorkingDir'],runInfo['stepName']) #generate current working dir
    else:
      self.workingDir = runInfo['WorkingDir']

  @abc.abstractmethod
  def run(self, input):
    """
      This method executes the postprocessor action.
      @ In,  input, object, object containing the data to process.
      Should avoid to use (inputToInternal output), and passing xarray directly/dataset
      Possible inputs include: dict, xarray.Dataset, pd.DataFrame
      @ Out, dict, xarray.Dataset, pd.DataFrame --> I think we can avoid collectoutput in the plugin pp
    """

  def collectOutput(self,finishedJob, output):
    """
      Method that collects the outputs from the "run" method of the PostProcessor
      @ In, finishedJob, InternalRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    if output.type not in self.validDataType:
      self.raiseAnError(IOError, 'Output type', str(output.type), 'is not allowed!')
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")
    outputRealization = evaluation[1]

    if output.type in ['PointSet','HistorySet']:
      if self.outputDataset:
        self.raiseAnError(IOError, "DataSet output is required, but the provided type of DataObject is", output.type)
      self.raiseADebug('Dumping output in data object named ' + output.name)
      if self.outputMultipleRealizations:
        if isinstance(outputRealization, ds):
          #  it is a dataset
          output.load(outputRealization, style='dataset')
        else:
          if 'dims' in outputRealization:
            dims = outputRealization['dims']
          else:
            dims = {}
          output.load(outputRealization['data'], style='dict', dims=dims)
      else:
        output.addRealization(outputRealization)
    elif output.type in ['DataSet']:
      self.raiseADebug('Dumping output in DataSet named ' + output.name)
      output.load(outputRealization, style='dataset')
