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
    inputSpecification.addSub(InputData.parameterInputFactory('costID' , contentType=InputTypes.StringType))
    inputSpecification.addSub(InputData.parameterInputFactory('valueID', contentType=InputTypes.StringType))
    return inputSpecification

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = ParetoFrontier.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already-parsed input.
      @ Out, None
    """
    costID  = paramInput.findFirst('costID')
    self.costID  = costID.value
    valueID = paramInput.findFirst('valueID')
    self.valueID = valueID.value

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
                                 .format(self.name, currentInput.type))
    return currentInp

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, DataObject, point set that contains the data to be processed
      @ Out, paretoFrontierDict, dict, dictionary containning the Pareto Frontier information
    """
    inData = self.inputToInternal(inputIn)
    data = inData.asDataset()
    sortedData = data.sortby(self.costID)

    coordinates = np.zeros(1,dtype=int)
    for index,elem in enumerate(sortedData[self.costID].values):
      if (index>1) and (sortedData[self.valueID].values[index]>sortedData[self.valueID].values[coordinates[-1]]):
        coordinates = np.append(coordinates,index)

    selection = sortedData.isel(RAVEN_sample_ID=coordinates).to_array().values
    paretoFrontierData = np.transpose(selection)

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
