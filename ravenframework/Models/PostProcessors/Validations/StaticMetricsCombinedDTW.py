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
  Created on April 04, 2021

  @author: alfoa

  This class represents a base class for the validation algorithms
  It inherits from the PostProcessor directly
  ##TODO: Recast it once the new PostProcesso API gets in place
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import xarray as xr
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ....utils import utils
from ....utils import InputData, InputTypes
from ..ValidationBase import ValidationBase
#Internal Modules End--------------------------------------------------------------------------------

class StaticMetricsCombinedDTW(ValidationBase):
  """
    StaticMetricsCombinedDTW is class that computes a metric that combines DTW and Static Distance Metrics
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(StaticMetricsCombinedDTW, cls).getInputSpecification()
    DTW = InputData.parameterInputFactory("DTW", contentType=InputTypes.StringType)
    DTW.addParam("class", InputTypes.StringType, True)
    DTW.addParam("type", InputTypes.StringType, True)
    dobs = InputData.parameterInputFactory("DataObject", contentType=InputTypes.StringType)
    DTW.addSub(dobs) 
    specs.addSub(DTW)    
    static = InputData.parameterInputFactory("Static", contentType=InputTypes.StringType)
    static.addParam("class", InputTypes.StringType, True)
    static.addParam("type", InputTypes.StringType, True)
    dobs = InputData.parameterInputFactory("DataObject", contentType=InputTypes.StringType)
    static.addSub(dobs)
    specs.addSub(static)
    
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR StaticMetricsCombinedDTW'
    self.dynamicType = ['static','dynamic'] #  for now only static is available
    self.name = 'StaticMetricsCombinedDTW'
    self.dtwDataObjectName = None
    self.staticDataObjectName = None
    self.addAssemblerObject('DTW', InputData.Quantity.zero_to_infinity)
    self.addAssemblerObject('Static', InputData.Quantity.zero_to_infinity)
    
    # self.pivotParameter = None

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    DTWcounter = 0
    Staticcounter = 0
    for child in paramInput.subparts:
      if child.getName() == 'Features' or child.getName() == 'Targets':
        continue
      elif child.getName() == 'DTW':
        if 'type' not in child.parameterValues.keys() or 'class' not in child.parameterValues.keys():
          self.raiseAnError(IOError, 'Tag DTW must have attributes "class" and "type"')
        dobs = child.findFirst('DataObject')
        if dobs is not None:
          self.dtwDataObjectName = dobs.value
        else:
          self.raiseAnError(IOError, 'Tag DTW must contain a <DataObject> XML node')
        DTWcounter += 1
      elif child.getName() == 'Static':
        if 'type' not in child.parameterValues.keys() or 'class' not in child.parameterValues.keys():
          self.raiseAnError(IOError, 'Tag Static must have attributes "class" and "type"')
        dobs = child.findFirst('DataObject')
        if dobs is not None:
          self.staticDataObjectName = dobs.value
        else:
          self.raiseAnError(IOError, 'Tag Static must contain a <DataObject> XML node')        
        Staticcounter += 1
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.getName(), " is provided for StaticMetricsCombinedDTW Postprocessor!")

    if not DTWcounter or not Staticcounter:
      self.raiseAnError(IOError, f"XML node 'DTW' and 'Static' are both required but not provided. PP {self.name}")
    if DTWcounter > 1 or Staticcounter > 1:
      self.raiseAnError(IOError, f"Only 1 'DTW' and 'Static' XML nodes can be provided. Got more than 1. PP {self.name}")
 
  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    super().initialize(runInfo, inputs, initDict)
    self.dtw = self.assemblerDict['DTW'][0][3]
    self.static = self.assemblerDict['Static'][0][3]

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    dataDict = {self.getDataSetName(data): data for _, _, data in inputIn['Data']}
    names = [self.getDataSetName(inp[-1]) for inp in inputIn['Data']]
    nVarsDTW = len(self.dtw._pp.metricsDict)
    dtwScaling = 1e23
    for metric in self.dtw._pp.metricsDict:
      dtwScaling = min(dtwScaling, self.dtw._pp.metricsDict[metric]._getInterface().lenPath)
    nVarsStatic = len(self.static._pp.metrics)
    
    evaluation ={k: np.atleast_1d(val) for k, val in  self._evaluate(dataDict, **{'dataobjectNames': names,
                                                                                  'nVarsDTW': nVarsDTW,
                                                                                  'nVarsStatic': nVarsStatic,
                                                                                  'dtwScaling': dtwScaling}).items()}
    return evaluation

  ### utility functions
  def _evaluate(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target":value}
    """
    names = kwargs.get('dataobjectNames')
    #nVarsDTW =  kwargs.get('nVarsDTW')
    #nVarsStatic =  kwargs.get('nVarsStatic')
    dtwScaling =  kwargs.get('dtwScaling')
    self.staticDataObjectName
    self.dtwDataObjectName
    
    staticGlobal = 0
    
    outputDict = {}
    for feat in self.features:
      for targ in self.targets:
        sourceName, featData = self._getDataFromDataDict(datasets, feat, names)
        if sourceName == self.dtwDataObjectName:
          featData /= dtwScaling
        sourceName, targData = self._getDataFromDataDict(datasets, targ, names)
        if sourceName == self.dtwDataObjectName:
          targData /= dtwScaling
        
        name = "{}_{}".format(feat.split("|")[-1], targ.split("|")[-1])
        outputDict[name] = np.sqrt(featData**2 + targData**2)
        if sourceName == self.staticDataObjectName:
          staticGlobal += outputDict[name]
    return outputDict

  def _getDataFromDataDict(self, datasets, var, names=None):
    """
      Utility function to retrieve the data from dataDict
      @ In, datasets, list, list of datasets (data1,data2,etc.) to search from.
      @ In, names, list, optional, list of datasets names (data1,data2,etc.). If not present, the search will be done on the full list.
      @ In, var, str, the variable to find (either in fromat dataobject|var or simply var)
      @ Out, data, (sourceName,numpy.ndarray), the retrived data and dataobject name (sourceName, data)
    """
    if "|" in var and names is not None:
      do, feat =  var.split("|")
      data = datasets[do][feat]
    else:
      for doIndex, ds in enumerate(datasets):
        if var in ds:
          data = ds[var]
          break
    dim = len(data.shape)
    # (numRealizations,  numHistorySteps) for MetricDistributor
    data = data.values
    if dim == 1:
      #  the following reshaping does not require a copy
      data.shape = (data.shape[0], 1)
    return do, data
