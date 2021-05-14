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

  def run(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    names = kwargs.get('dataobjectNames')
    outs = {}
    for feat, targ in zip(self.features, self.targets):
      featData = self._getDataFromDatasets(datasets, feat, names)
      targData = self._getDataFromDatasets(datasets, targ, names)
      # featData = (featData[0], None)
      # targData = (targData[0], None)
      for metric in self.metrics:
        name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.estimator.name)
        outs[name] = metric.evaluate((featData, targData), multiOutput='raw_values')
    return outs

