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
  Created on April 29, 2021

  @author: Mohammad Abdo (@Jimmy-INL)

  This class represents a base class for the validation algorithms
  It inherits from the PostProcessor directly
  ##TODO: Recast it once the new PostProcesso API gets in place
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
#from utils import xmlUtils
from utils import InputData, InputTypes
#import Files
#import Distributions
#import MetricDistributor
from utils import utils
from .ValidationBase import ValidationBase
from utils.mathUtils import partialDerivative, derivatives
#Internal Modules End--------------------------------------------------------------------------------

class Representativity(ValidationBase):
  """
    Representativity is a base class for validation problems
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
    parametersInput = InputData.parameterInputFactory("Parameters", contentType=InputTypes.StringListType)
    parametersInput.addParam("type", InputTypes.StringType)
    specs.addSub(parametersInput)
    targetParametersInput = InputData.parameterInputFactory("targetParameters", contentType=InputTypes.StringListType)
    targetParametersInput.addParam("type", InputTypes.StringType)
    specs.addSub(targetParametersInput)
    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    from Models.PostProcessors import factory as ppFactory # delay import to allow definition
    self.printTag = 'POSTPROCESSOR ValidationBase'
    self.dynamicType = ['static'] #  for now only static is available
    self.acceptableMetrics = ["RepresentativityFactors"] #  acceptable metrics
    self.name = 'Represntativity'
    self.stat = ppFactory.returnInstance('BasicStatistics')
    self.stat.what = ['NormalizedSensitivities'] # expected value calculation


  # def inputToInternal(self, currentInputs):
  #   """
  #     Method to convert an input object into the internal format that is
  #     understandable by this pp.
  #     @ In, currentInputs, list or DataObject, data object or a list of data objects
  #     @ Out, measureList, list of (feature, target), the list of the features and targets to measure the distance between
  #   """
  #   if type(currentInputs) != list:
  #     currentInputs = [currentInputs]
  #   hasPointSet = False
  #   hasHistorySet = False
  #   #Check for invalid types
  #   for currentInput in currentInputs:
  #     inputType = None
  #     if hasattr(currentInput, 'type'):
  #       inputType = currentInput.type

  #     if isinstance(currentInput, Files.File):
  #       self.raiseAnError(IOError, "Input type '", inputType, "' can not be accepted")
  #     elif isinstance(currentInput, Distributions.Distribution):
  #       pass #Allowed type
  #     elif inputType == 'HDF5':
  #       self.raiseAnError(IOError, "Input type '", inputType, "' can not be accepted")
  #     elif inputType == 'PointSet':
  #       hasPointSet = True
  #     elif inputType == 'HistorySet':
  #       hasHistorySet = True
  #       if self.multiOutput == 'raw_values':
  #         self.dynamic = True
  #         if self.pivotParameter not in currentInput.getVars('indexes'):
  #           self.raiseAnError(IOError, self, 'Pivot parameter', self.pivotParameter,'has not been found in DataObject', currentInput.name)
  #         if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
  #           self.raiseAnError(IOError, "HistorySet", currentInput.name," is not syncronized, please use Interfaced PostProcessor HistorySetSync to pre-process it")
  #         pivotValues = currentInput.asDataset()[self.pivotParameter].values
  #         if len(self.pivotValues) == 0:
  #           self.pivotValues = pivotValues
  #         elif set(self.pivotValues) != set(pivotValues):
  #           self.raiseAnError(IOError, "Pivot values for pivot parameter",self.pivotParameter, "in provided HistorySets are not the same")
  #     else:
  #       self.raiseAnError(IOError, "Metric cannot process "+inputType+ " of type "+str(type(currentInput)))
  #   if self.multiOutput == 'raw_values' and hasPointSet and hasHistorySet:
  #       self.multiOutput = 'mean'
  #       self.raiseAWarning("Reset 'multiOutput' to 'mean', since both PointSet and HistorySet are provided as Inputs. Calculation outputs will be aggregated by averaging")

  #   measureList = []

  #   for cnt in range(len(self.features)):
  #     feature = self.features[cnt]
  #     target = self.targets[cnt]
  #     featureData =  self.__getMetricSide(feature, currentInputs)
  #     targetData = self.__getMetricSide(target, currentInputs)
  #     measureList.append((featureData, targetData))

    # return measureList

  def initialize(self, features, targets, **kwargs):
    """
      Set up this interface for a particular activity
      @ In, features, list, list of features
      @ In, targets, list, list of targets
      @ In, kwargs, dict, keyword arguments
    """
    super().initialize(features, targets, **kwargs)
    self.stat.toDo = {'NormalizedSensitivity':[{'targets':set(self.targets), 'prefix':'nsen'}]}
    # self.stat.toDo = {'NormalizedSensitivity'[{'targets':set([self.targets]), 'prefix':'nsen'}]}
    fakeRunInfo = {'workingDir':'','stepName':''}
    self.stat.initialize(fakeRunInfo, self.Parameters, features, **kwargs)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'Parameters':
        self.Parameters = child.value
      elif child.getName() == 'targetParameters':
        self.targetParameters = child.value

  def run(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    self.stat.run({'targets':{self.target:xarray.DataArray(self.functionS.evaluate(tempDict)[self.target])}})[self.computationPrefix +"_"+self.target]
    names = kwargs.get('dataobjectNames')
    outs = {}
    for feat, targ, param, targParam in zip(self.features, self.targets, self.Parameters, self.targetParameters):
      featData = self._getDataFromDatasets(datasets, feat, names)
      targData = self._getDataFromDatasets(datasets, targ, names)
      Parameters = self._getDataFromDatasets(datasets, param, names)
      targetParameters = self._getDataFromDatasets(datasets, targParam, names)
      senFOMs = partialDerivative(featData.data,np.atleast_2d(Parameters.data)[0,:],'x1')
      # senFOMs = np.atleast_2d(Parameters.data)
      senMeasurables = np.atleast_2d(targetParameters.data)
      covParameters = senFOMs.T @ senMeasurables
      for metric in self.metrics:
        name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.name)
        outs[name] = metric.evaluate(featData, targData, senFOMs = senFOMs, senMeasurables=senMeasurables, covParameters=covParameters)
    return outs