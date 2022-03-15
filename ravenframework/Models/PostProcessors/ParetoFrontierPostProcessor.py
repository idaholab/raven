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
Created on March 25, 2020

@author: mandd
"""

#External Modules---------------------------------------------------------------
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import utils
from ...utils import InputData, InputTypes
from ...utils import frontUtils
from ... import Runners
#Internal Modules End-----------------------------------------------------------

class ParetoFrontier(PostProcessorInterface):
  """
    This postprocessor selects the points that lie on the Pareto frontier
    The postprocessor acts only on PointSet and return a subset of such PointSet
  """
  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.valueLimit = None   # variable associated with the lower limit of the value dimension
    self.costLimit  = None   # variable associated with the upper limit of the cost dimension
    self.invCost    = False  # variable which indicates if the cost dimension is inverted (e.g., it represents savings rather than costs)
    self.invValue   = False  # variable which indicates if the value dimension is inverted (e.g., it represents a lost value rather than value)
    self.validDataType = ['PointSet'] # The list of accepted types of DataObject
    self.outputMultipleRealizations = True # True indicate multiple realizations are returned


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

    objDataType = InputTypes.makeEnumType("objective", "objectiveType", ['min','max'])

    objective = InputData.parameterInputFactory('objective', contentType=InputTypes.StringType)
    objective.addParam('goal',       param_type=objDataType,           required=True)
    objective.addParam('upperLimit', param_type=InputTypes.FloatType,  required=False)
    objective.addParam('lowerLimit', param_type=InputTypes.FloatType,  required=False)
    inputSpecification.addSub(objective)

    return inputSpecification


  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    self.objectives = {}

    for child in paramInput.subparts:
      if child.getName() == 'objective':
        self.objectives[child.value]={}
        self.objectives[child.value]['goal']       = child.parameterValues['goal']
        self.objectives[child.value]['upperLimit'] = child.parameterValues.get('upperLimit')
        self.objectives[child.value]['lowerLimit'] = child.parameterValues.get('lowerLimit')


  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      In this case, we only want data objects!
      @ In, currentInp, list, an object that needs to be converted
      @ Out, currentInp, DataObject.HistorySet, input data
    """
    if len(currentInp) > 1:
      self.raiseAnError(IOError, 'ParetoFrontier postprocessor {} expects one input DataObject, but received {} inputs!".'
                                  .format(self.name,len(currentInp)))
    currentInp = currentInp[0]
    if currentInp.type not in ['PointSet']:
      self.raiseAnError(IOError, 'ParetoFrontier postprocessor "{}" requires a DataObject input! Got "{}".'
                                 .format(self.name, currentInp.type))
    return currentInp


  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, DataObject, point set that contains the data to be processed
      @ Out, paretoFrontierDict, dict, dictionary containing the Pareto Frontier information
    """
    inData = self.inputToInternal(inputIn)
    data = inData.asDataset()

    dataTemp = data[list(self.objectives.keys())]
    for index,obj in enumerate(self.objectives.keys()):
      if self.objectives[obj]['goal']=='max':
        dataTemp[obj] = (-1.) * dataTemp[obj]

    paretoFrontMask = frontUtils.nonDominatedFrontier(np.transpose(dataTemp.to_array().values), returnMask=False)
    selection = data.isel(RAVEN_sample_ID=np.array(paretoFrontMask))

    for obj in self.objectives.keys():
      if self.objectives[obj]['upperLimit']:
        selection = selection.where(selection[obj]<=self.objectives[obj]['upperLimit'])
      if self.objectives[obj]['lowerLimit']:
        selection = selection.where(selection[obj]>=self.objectives[obj]['lowerLimit'])

    filteredParetoFrontier = selection.to_array().values
    paretoFrontierData = np.transpose(filteredParetoFrontier)
    paretoFrontierDict = {}
    for index,varID in enumerate(data.data_vars):
      paretoFrontierDict[varID] = paretoFrontierData[:,index]
    paretoFrontierDict = {'data':paretoFrontierDict, 'dims':{}}
    return paretoFrontierDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    super().collectOutput(finishedJob, output)
