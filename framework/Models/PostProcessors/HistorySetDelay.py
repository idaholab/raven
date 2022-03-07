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
Created on 2021-April-19

@author: cogljj

This is to implement a delay or lagged parameters in a HistorySet
"""

import copy
import numpy as np
import xarray as xr

from ...utils import InputData, InputTypes
from .PostProcessorReadyInterface import PostProcessorReadyInterface

class HistorySetDelay(PostProcessorReadyInterface):
  """
    Class to get lagged or delayed data out of a history set.
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
    delayClass = InputData.parameterInputFactory("delay", contentType=InputTypes.StringType,
                                                 descr="Adds a delay variable that"
                                                 +" is a copy of an existing variable"
                                                 +" but offset along the pivot parameter.")
    delayClass.addParam("original", InputTypes.StringType, True,
                        descr="Original variable name to copy data from")
    delayClass.addParam("new", InputTypes.StringType, True,
                        descr="New (delayed) variable name to create data in")
    delayClass.addParam("steps", InputTypes.IntegerType, True,
                        descr="Steps to offset (-1 is previous step) the new variable")
    delayClass.addParam("default", InputTypes.FloatType, True,
                        descr="Default value to use for unavailable steps where"
                        +" the offset would go outside existing data.")
    inputSpecification.addSub(delayClass, InputData.Quantity.one_to_infinity)
    inputSpecification.addSub(InputData.parameterInputFactory("method", contentType=InputTypes.StringType))
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.delays = [] # list of delay variables (original, new, steps, default)
    self.validDataType = ['HistorySet']       #only available output is HistorySet
    self.outputMultipleRealizations = True    #this PP will return a full set of realization
    self.printTag = 'PostProcessor HistorySetDelay'
    self.setInputDataType('xrDataset')
    self.keepInputMeta(True)

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == 'delay':
        self.delays.append((child.parameterValues['original'],
                            child.parameterValues['new'],
                            child.parameterValues['steps'],
                            child.parameterValues['default']))

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs)>1:
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only one dataObject')
    if inputs[0].type != 'HistorySet':
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only HistorySet dataObject, but got "{}"'.format(inputs[0].type))

  def run(self,inputIn):
    """
      Method to post-process the dataObjects
      @ In, inputIn, dict, dictionary of data.
          inputIn = {'Data':listData, 'Files':listOfFiles},
          listData has the following format: (listOfInputVars, listOfOutVars, xr.Dataset)
      @ Out, data, xarray.DataSet, output dataset
    """
    inpVars, outVars, data = inputIn['Data'][0]
    for delay in self.delays:
      original, new, steps, default = delay
      coords = {key: data[original][key] for key in data[original].dims}
      orig_data = data[original].values
      new_data = copy.copy(orig_data)
      if steps < 0:
        new_data[:, :-steps] = default
        new_data[:, -steps:] = orig_data[:,:steps]
      elif steps > 0:
        new_data[:, -steps:] = default
        new_data[:, :-steps] = orig_data[:,steps:]
      # else:
      # steps is 0, so just keep the copy
      data[new] = xr.DataArray(data=new_data, coords=coords, dims=coords.keys())
    return data
