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
from ..ValidationBase import ValidationBase
#Internal Modules End--------------------------------------------------------------------------------

class Probabilistic(ValidationBase):
  """
    Probabilistic is a base class for validation problems
    It represents the base class for most validation problems
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
    specs = super(Probabilistic, cls).getInputSpecification()
    #specs.addSub(metricInput)
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR Probabilistic'
    self.dynamicType = ['static','dynamic'] #  for now only static is available
    self.acceptableMetrics = ["CDFAreaDifference", "PDFCommonArea"] #  acceptable metrics
    self.name = 'Probabilistic'
    # self.pivotParameter = None

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)




  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    dataDict = {self.getDataSetName(data): data for _, _, data in inputIn['Data']}
    pivotParameter = self.pivotParameter
    names = [self.getDataSetName(inp[-1]) for inp in inputIn['Data']]
    if len(inputIn['Data'][0][-1].indexes) and self.pivotParameter is None:
      if 'dynamic' not in self.dynamicType: #self.model.dataType:
        self.raiseAnError(IOError, "The validation algorithm '{}' is not a dynamic model but time-dependent data has been inputted in object {}".format(self._type, inputIn['Data'][0][-1].name))
    evaluation ={k: np.atleast_1d(val) for k, val in  self._evaluate(dataDict, **{'dataobjectNames': names}).items()}

    if pivotParameter:
      if len(inputIn['Data'][0][-1]['time']) != len(list(evaluation.values())[0]):
        self.raiseAnError(RuntimeError, "The pivotParameter value '{}' has size '{}' and validation output has size '{}'".format( len(dataSets[0][self.pivotParameter]), len(evaluation.values()[0])))
      if pivotParameter not in evaluation:
        evaluation[pivotParameter] = inputIn['Data'][0][-1]['time']
    return evaluation

  ### utility functions
  def _evaluate(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    names = kwargs.get('dataobjectNames')
    outputDict = {}
    for feat, targ in zip(self.features, self.targets):
      featData = self._getDataFromDataDict(datasets, feat, names)
      targData = self._getDataFromDataDict(datasets, targ, names)
      for metric in self.metrics:
        name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.estimator.name)
        outputDict[name] = metric.evaluate((featData, targData), multiOutput='raw_values')
    return outputDict

  def _getDataFromDataDict(self, datasets, var, names=None):
    """
      Utility function to retrieve the data from dataDict
      @ In, datasets, list, list of datasets (data1,data2,etc.) to search from.
      @ In, names, list, optional, list of datasets names (data1,data2,etc.). If not present, the search will be done on the full list.
      @ In, var, str, the variable to find (either in fromat dataobject|var or simply var)
      @ Out, data, tuple(numpy.ndarray, xarray.DataArray or None), the retrived data (data, probability weights (None if not present))
    """
    pw = None
    if "|" in var and names is not None:
      do, feat =  var.split("|")
      dat = datasets[do][feat]
    else:
      for doIndex, ds in enumerate(datasets):
        if var in ds:
          dat = ds[var]
          break
    if 'ProbabilityWeight-{}'.format(feat) in datasets[do]:
      pw = datasets[do]['ProbabilityWeight-{}'.format(feat)].values
    elif 'ProbabilityWeight' in datasets[do]:
      pw = datasets[do]['ProbabilityWeight'].values
    dim = len(dat.shape)
    # (numRealizations,  numHistorySteps) for MetricDistributor
    dat = dat.values
    if dim == 1:
      #  the following reshaping does not require a copy
      dat.shape = (dat.shape[0], 1)
    data = dat, pw
    return data
