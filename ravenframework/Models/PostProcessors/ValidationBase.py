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
  Created on March 20, 2021
  @author: alfoa, wangc
  description: Validation Postprocessor Base class. It is aimed to
               to represent a base class for any validation tecniques and processes
"""

#External Modules---------------------------------------------------------------
import numpy as np
import copy
import time
import xarray as xr
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import utils, mathUtils, xmlUtils
from ...utils import InputData, InputTypes
from ... import DataObjects
from ... import MetricDistributor
#Internal Modules End-----------------------------------------------------------


class ValidationBase(PostProcessorReadyInterface):
  """
    Validation class. It will apply the specified validation algorithms in
    the models to a dataset, each specified algorithm's output can be loaded to
    dataObject.
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
    ## This will replace the lines above
    specs = super(ValidationBase, cls).getInputSpecification()
    preProcessorInput = InputData.parameterInputFactory("PreProcessor", contentType=InputTypes.StringType)
    preProcessorInput.addParam("class", InputTypes.StringType)
    preProcessorInput.addParam("type", InputTypes.StringType)
    specs.addSub(preProcessorInput)
    pivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    specs.addSub(pivotParameterInput)
    featuresInput = InputData.parameterInputFactory("Features", contentType=InputTypes.StringListType)
    specs.addSub(featuresInput)
    targetsInput = InputData.parameterInputFactory("Targets", contentType=InputTypes.StringListType)
    specs.addSub(targetsInput)
    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    metricInput.addParam("class", InputTypes.StringType)
    metricInput.addParam("type", InputTypes.StringType)
    specs.addSub(metricInput)

    return specs

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR VALIDATION'
    self.pivotParameter = None  ## default pivotParameter for HistorySet
    self._type = None           ## the type of library that are used for validation, i.e. DSS
    # add assembly objects (and set up pointers)
    self.PreProcessor = None    ## Instance of PreProcessor, default is None
    self.metrics = None          ## Instance of Metric, default is None

    self.dataType = ['static', 'dynamic'] # the type of data can be passed in (static aka PointSet, dynamic aka HistorySet) (if both are present the validation algorithm can work for both data types)
    self.acceptableMetrics = [] # if not populated all types of metrics are accepted, otherwise list the metrics (see Probablistic.py for an example)
    self.features = None        # list of feature variables
    self.targets = None         # list of target variables
    self.pivotValues = None     # pivot values (present if dynamic == True)

    self.addAssemblerObject('Metric', InputData.Quantity.zero_to_infinity)
    self.addAssemblerObject('PreProcessor', InputData.Quantity.zero_to_infinity)
    ## dataset option
    self.setInputDataType('xrDataset')
    # swith to 'dict' when you are using dict as operations
    # self.setInputDataType('dict')
    # If you want to keep the input meta data, please pass True, otherwise False
    self.keepInputMeta(False)

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In , None, None
      @ Out, dict, dictionary of objects needed
    """
    return {'internal':[(None,'jobHandler')]}

  def _localGenerateAssembler(self,initDict):
    """Generates the assembler.
      @ In, initDict, dict, init objects
      @ Out, None
    """
    self.jobHandler = initDict['internal']['jobHandler']


  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    # this loop set the pivot parameter (it could use paramInput.findFirst but we want to show how to add more paramters)
    for child in paramInput.subparts:
      if child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      elif child.getName() == 'Features':
        self.features = child.value
      elif child.getName() == 'Targets':
        self.targets = child.value
    if 'static' not in self.dataType and self.pivotParameter is None:
      self.raiseAnError(IOError, "The validation algorithm '{}' is a dynamic model ONLY but no <pivotParameter> node has been inputted".format(self._type))
    if not self.features:
      self.raiseAnError(IOError, "XML node 'Features' is required but not provided")

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if 'PreProcessor' in self.assemblerDict:
      self.PreProcessor = self.assemblerDict['PreProcessor'][0][3]
    if 'Metric' in self.assemblerDict:
      metrics = [metric[3] for metric in self.assemblerDict['Metric']]
      self.metrics = [MetricDistributor.factory.returnInstance('MetricDistributor', metric) for metric in metrics]

    if len(inputs) > 1:
      # if inputs > 1, check if the | is present to understand where to get the features and target
      notStandard = [k for k in self.features + self.targets if "|" not in k]
      if notStandard:
        self.raiseAnError(IOError, "# Input Datasets/DataObjects > 1! features and targets must use the syntax DataObjectName|feature to be usable! Not standard features are: {}!".format(",".join(notStandard)))
    # now lets check that the variables are in the dataobjects
    if isinstance(inputs[0], DataObjects.DataSet):
      do = [inp.name for inp in inputs]
      if len(inputs) > 1:
        allFound = [feat.split("|")[0].strip() in do for feat in self.features]
        allFound += [targ.split("|")[0].strip() in do for targ in self.targets]
        if not all(allFound):
          self.raiseAnError(IOError, "Targets and Features are linked to DataObjects that have not been listed as inputs in the Step. Please check input!")
      # check variables
      for indx, dobj in enumerate(do):
        variables = [var.split("|")[-1].strip() for var in (self.features + self.targets) if dobj in var]
        if not utils.isASubset(variables,inputs[indx].getVars()):
          self.raiseAnError(IOError, "The variables '{}' not found in input DataObjet '{}'!".format(",".join(list(set(list(inputs[indx].getVars())) - set(variables))), dobj))

    if self.acceptableMetrics:
      acceptable = [True if metric.estimator.isInstanceString(self.acceptableMetrics) else False for metric in self.metrics]
      if not all(acceptable):
        notAcceptable = [self.metrics[i].estimator.interfaceKind for i, x in enumerate(acceptable) if not x]
        self.raiseAnError(IOError,
            "The metrics '{}' are not acceptable for validation algorithm: '{}'".format(', '.join(notAcceptable), self.name))

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

  # Each individual validation pp should implement their own run method.
  # def run(self, input):

  @staticmethod
  def getDataSetName(ds):
    """
    """
    datasetMeta = ds.attrs['DataSet'].getRoot()
    name = xmlUtils.findPath(datasetMeta, 'general/datasetName').text
    return name
