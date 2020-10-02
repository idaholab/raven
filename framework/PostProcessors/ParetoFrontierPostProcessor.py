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
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData, InputTypes
import Runners
#Internal Modules End-----------------------------------------------------------

class ParetoFrontier(PostProcessor):
  """
    This postprocessor selects the points that lie on the Pareto frontier
    The postprocessor acts only on PointSet and return a subset of such PointSet
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.valueLimit = None   # variable associated with the lower limit of the value dimension
    self.costLimit  = None   # variable associated with the upper limit of the cost dimension
    self.invCost    = False  # variable which indicates if the cost dimension is inverted (e.g., it represents savings rather than costs)
    self.invValue   = False  # variable which indicates if the value dimension is inverted (e.g., it represents a lost value rather than value)

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(ParetoFrontier, cls).getInputSpecification()

    costIDInput = InputData.parameterInputFactory("costID", contentType=InputTypes.StringType)
    costIDInput.addParam("inv", InputTypes.BoolType, True)
    inputSpecification.addSub(costIDInput)

    valueIDInput = InputData.parameterInputFactory("valueID", contentType=InputTypes.StringType)
    valueIDInput.addParam("inv", InputTypes.BoolType, True)
    inputSpecification.addSub(valueIDInput)

    costLimitInput = InputData.parameterInputFactory("costLimit", contentType=InputTypes.FloatType)
    costLimitInput.addParam("type", InputTypes.StringType, True)
    inputSpecification.addSub(costLimitInput)

    valueLimitInput = InputData.parameterInputFactory("valueLimit", contentType=InputTypes.FloatType)
    valueLimitInput.addParam("type", InputTypes.StringType, True)
    inputSpecification.addSub(valueLimitInput)

    return inputSpecification

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    costID  = paramInput.findFirst('costID')
    self.costID  = costID.value
    self.invCost = costID.parameterValues['inv']

    valueID = paramInput.findFirst('valueID')
    self.valueID = valueID.value
    self.invValue = valueID.parameterValues['inv']

    costLimit  = paramInput.findFirst('costLimit')
    if costLimit is not None:
      self.costLimit  = costLimit.value
      self.costLimitType = costLimit.parameterValues['type']

    valueLimit = paramInput.findFirst('valueLimit')
    if valueLimit is not None:
      self.valueLimit = valueLimit.value
      self.valueLimitType = valueLimit.parameterValues['type']


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

    if self.invCost:
      data[self.costID] = (-1.) * data[self.costID]
    if self.invValue:
      data[self.valueID] = (-1.) * data[self.valueID]

    sortedData = data.sortby(self.costID)
    coordinates = np.zeros(1,dtype=int)
    for index,elem in enumerate(sortedData[self.costID].values):
      if (index>1) and (sortedData[self.valueID].values[index]>sortedData[self.valueID].values[coordinates[-1]]):
        # the point at index is part of the pareto frontier
        coordinates = np.append(coordinates,index)

    selection = sortedData.isel(RAVEN_sample_ID=coordinates)

    if self.invCost:
      selection[self.costID] = (-1.) * selection[self.costID]
    if self.invValue:
      selection[self.valueID] = (-1.) * selection[self.valueID]

    if self.valueLimit is not None:
      if self.valueLimitType=="upper":
        selection = selection.where(selection[self.valueID]<=self.valueLimit)
      else:
        selection = selection.where(selection[self.valueID]>=self.valueLimit)
    if self.costLimit is not None:
      if self.costLimitType=="upper":
        selection = selection.where(selection[self.costID]<=self.costLimit)
      else:
        selection = selection.where(selection[self.costID]>=self.costLimit)

    filteredParetoFrontier = selection.to_array().values
    paretoFrontierData = np.transpose(filteredParetoFrontier)
    paretoFrontierDict = {}
    for index,varID in enumerate(sortedData.data_vars):
      paretoFrontierDict[varID] = paretoFrontierData[:,index]

    return paretoFrontierDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()

    outputDict ={}
    outputDict['data'] = evaluation[1]

    if output.type in ['PointSet']:
      outputDict['dims'] = {}
      for key in outputDict.keys():
        outputDict['dims'][key] = []
      output.load(outputDict['data'], style='dict', dims=outputDict['dims'])
    else:
        self.raiseAnError(RuntimeError, 'ParetoFrontier failed: Output type ' + str(output.type) + ' is not supported.')
