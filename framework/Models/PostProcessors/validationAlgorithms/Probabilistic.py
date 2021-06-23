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
from utils import utils
from .ValidationBase import ValidationBase
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
    specs = super(ValidationBase, cls).getInputSpecification()
    #specs.addSub(metricInput)
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR ValidationBase'
    self.dynamicType = ['static','dynamic'] #  for now only static is available
    self.acceptableMetrics = ["CDFAreaDifference", "PDFCommonArea"] #  acceptable metrics
    self.name = 'Probabilistic'
    # self.pivotParameter = None


  def inputToInternal(self, currentInputs):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInputs, list or DataObject, data object or a list of data objects
      @ Out, measureList, list of (feature, target), the list of the features and targets to measure the distance between
    """
    if type(currentInputs) != list:
      currentInputs = [currentInputs]
    hasPointSet = False
    hasHistorySet = False
    #Check for invalid types
    for currentInput in currentInputs:
      inputType = None
      if hasattr(currentInput, 'type'):
        inputType = currentInput.type

      if isinstance(currentInput, Files.File):
        self.raiseAnError(IOError, "Input type '", inputType, "' can not be accepted")
      elif isinstance(currentInput, Distributions.Distribution):
        pass #Allowed type
      elif inputType == 'HDF5':
        self.raiseAnError(IOError, "Input type '", inputType, "' can not be accepted")
      elif inputType == 'PointSet':
        hasPointSet = True
      elif inputType == 'HistorySet':
        hasHistorySet = True
        if self.multiOutput == 'raw_values':
          self.dynamic = True
          if self.pivotParameter not in currentInput.getVars('indexes'):
            self.raiseAnError(IOError, self, 'Pivot parameter', self.pivotParameter,'has not been found in DataObject', currentInput.name)
          if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
            self.raiseAnError(IOError, "HistorySet", currentInput.name," is not syncronized, please use Interfaced PostProcessor HistorySetSync to pre-process it")
          pivotValues = currentInput.asDataset()[self.pivotParameter].values
          if len(self.pivotValues) == 0:
            self.pivotValues = pivotValues
          elif set(self.pivotValues) != set(pivotValues):
            self.raiseAnError(IOError, "Pivot values for pivot parameter",self.pivotParameter, "in provided HistorySets are not the same")
      else:
        self.raiseAnError(IOError, "Metric cannot process "+inputType+ " of type "+str(type(currentInput)))
    if self.multiOutput == 'raw_values' and hasPointSet and hasHistorySet:
        self.multiOutput = 'mean'
        self.raiseAWarning("Reset 'multiOutput' to 'mean', since both PointSet and HistorySet are provided as Inputs. Calculation outputs will be aggregated by averaging")

    measureList = []

    for cnt in range(len(self.features)):
      feature = self.features[cnt]
      target = self.targets[cnt]
      featureData =  self.__getMetricSide(feature, currentInputs)
      targetData = self.__getMetricSide(target, currentInputs)
      measureList.append((featureData, targetData))

    return measureList

  def initialize(self, features, targets, **kwargs):
    """
      Set up this interface for a particular activity
      @ In, features, list, list of features
      @ In, targets, list, list of targets
      @ In, kwargs, dict, keyword arguments
    """
    super().initialize(features, targets, **kwargs)


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
    # inpVars, outVars, dataSet = inputIn['Data'][0]
    dataSets = [data for _, _, data in inputIn['Data']]
    pivotParameter = self.pivotParameter
    # # names = [dataSets[i].attrs['name'] for i in len(dataSets)]
    # names = ['simulation','experiment']
    # pivotParameter = self.pivotParameter
    # outs = {}
    # for feat, targ in zip(self.features, self.targets):
    #   featData = self._getDataFromDatasets(dataSets, feat, names)
    #   targData = self._getDataFromDatasets(dataSets, targ, names)
    #   for metric in self.metrics:
    #     name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.estimator.name)
    #     outs[name] = metric.evaluate((featData, targData), multiOutput='raw_values')
    # return outs

    names = [inp[-1].attrs['name'] for inp in inputIn['Data']]
    if len(inputIn['Data'][0][-1].indexes) and self.pivotParameter is None:
      if 'dynamic' not in self.dynamicType: #self.model.dataType:
        self.raiseAnError(IOError, "The validation algorithm '{}' is not a dynamic model but time-dependent data has been inputted in object {}".format(self._type, inputIn['Data'][0][-1].name))
      # else:
      #   pivotParameter = self.pivotParameter
    # #  check if pivotParameter
    # if pivotParameter:
    #   #  in case of dataobjects we check that the dataobject is either an HistorySet or a DataSet
    #   if isinstance(inputIn['Data'][0][-1], xr.Dataset) and not all([True if inp.type in ['HistorySet', 'DataSet']  else False for inp in inputIn]):
    #     self.raiseAnError(RuntimeError, "The pivotParameter '{}' has been inputted but PointSets have been used as input of PostProcessor '{}'".format(pivotParameter, self.name))
    #   if not all([True if pivotParameter in inp else False for inp in dataSets]):
    #     self.raiseAnError(RuntimeError, "The pivotParameter '{}' not found in datasets used as input of PostProcessor '{}'".format(pivotParameter, self.name))
    evaluation ={k: np.atleast_1d(val) for k, val in  self._evaluate(dataSets, **{'dataobjectNames': names}).items()}

    if pivotParameter:
      if len(dataSets[0][pivotParameter]) != len(list(evaluation.values())[0]):
        self.raiseAnError(RuntimeError, "The pivotParameter value '{}' has size '{}' and validation output has size '{}'".format( len(dataSets[0][self.pivotParameter]), len(evaluation.values()[0])))
      if pivotParameter not in evaluation:
        evaluation[pivotParameter] = dataSets[0][pivotParameter]
    return evaluation

  # ## merge the following run method with above run method
  # def run(self, datasets, **kwargs):
  #   """
  #     Main method to "do what you do".
  #     @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
  #     @ In, kwargs, dict, keyword arguments
  #     @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
  #   """
  #   names = kwargs.get('dataobjectNames')
  #   outs = {}
  #   for feat, targ in zip(self.features, self.targets):
  #     featData = self._getDataFromDatasets(datasets, feat, names)
  #     targData = self._getDataFromDatasets(datasets, targ, names)
  #     # featData = (featData[0], None)
  #     # targData = (targData[0], None)
  #     for metric in self.metrics:
  #       name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.estimator.name)
  #       outs[name] = metric.evaluate((featData, targData), multiOutput='raw_values')
  #   return outs

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
      featData = self._getDataFromDatasets(datasets, feat, names)
      targData = self._getDataFromDatasets(datasets, targ, names)
      for metric in self.metrics:
        name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.estimator.name)
        outputDict[name] = metric.evaluate((featData, targData), multiOutput='raw_values')
    return outputDict


  def _getDataFromDatasets(self, datasets, var, names=None):
    """
      Utility function to retrieve the data from datasets
      @ In, datasets, list, list of datasets (data1,data2,etc.) to search from.
      @ In, names, list, optional, list of datasets names (data1,data2,etc.). If not present, the search will be done on the full list.
      @ In, var, str, the variable to find (either in fromat dataobject|var or simply var)
      @ Out, data, tuple(numpy.ndarray, xarray.DataArray or None), the retrived data (data, probability weights (None if not present))
    """
    data = None
    pw = None
    dat = None
    if "|" in var and names is not None:
      do, feat =  var.split("|")
      doindex = names.index(do)
      dat = datasets[doindex][feat]
    else:
      for doindex, ds in enumerate(datasets):
        if var in ds:
          dat = ds[var]
          break
    if 'ProbabilityWeight-{}'.format(feat) in datasets[names.index(do)]:
      pw = datasets[doindex]['ProbabilityWeight-{}'.format(feat)].values
    elif 'ProbabilityWeight' in datasets[names.index(do)]:
      pw = datasets[doindex]['ProbabilityWeight'].values
    dim = len(dat.shape)
    # (numRealizations,  numHistorySteps) for MetricDistributor
    dat = dat.values
    if dim == 1:
      #  the following reshaping does not require a copy
      dat.shape = (dat.shape[0], 1)
    data = dat, pw
    return data
