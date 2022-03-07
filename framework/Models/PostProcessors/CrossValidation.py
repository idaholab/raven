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
Created on August 30, 2017

@author: wangc
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import os
import six
from collections import OrderedDict
import copy
#External Modules End--------------------------------------------------------------------------------

from .PostProcessorInterface import PostProcessorInterface
from ...utils import utils
from ...utils import InputData, InputTypes
from ... import Files
from ... import Models
from ... import CrossValidations
#Internal Modules End--------------------------------------------------------------------------------

class CrossValidation(PostProcessorInterface):
  """
    Cross Validation  class.
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
    inputSpecification = super(CrossValidation, cls).getInputSpecification()

    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    metricInput.addParam("class", InputTypes.StringType, True)
    metricInput.addParam("type", InputTypes.StringType, True)
    inputSpecification.addSub(metricInput)

    sciKitLearnInput = InputData.parameterInputFactory("SciKitLearn")

    sklTypeInput = InputData.parameterInputFactory("SKLtype", contentType=InputTypes.StringType)

    sciKitLearnInput.addSub(sklTypeInput)

    for name, inputType in [("p",InputTypes.IntegerType),
                            ("n_splits",InputTypes.IntegerType),
                            ("shuffle",InputTypes.BoolType),
                            ("random_state",InputTypes.IntegerType),
                            ('max_train_size',InputTypes.IntegerType),
                            ('test_fold', InputTypes.FloatListType),
                            ("test_size",InputTypes.FloatOrIntType),
                            ("train_size",InputTypes.FloatOrIntType),
                            ('n_groups',InputTypes.IntegerType),
                            ("labels",InputTypes.FloatListType),
                            ("scores",InputTypes.StringType)]:
      dataType = InputData.parameterInputFactory(name, contentType=inputType)
      sciKitLearnInput.addSub(dataType)

    inputSpecification.addSub(sciKitLearnInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR CROSS VALIDATION'
    self.dynamic        = False # is it time-dependent?
    self.metricsDict    = {}    # dictionary of metrics that are going to be assembled
    self.pivotParameter = None
    self.cvScore        = None
    # assembler objects to be requested
    self.addAssemblerObject('Metric', InputData.Quantity.one_to_infinity)
    #self.validMetrics = ['mean_absolute_error', 'explained_variance_score', 'r2_score', 'mean_squared_error', 'median_absolute_error']
    # 'median_absolute_error' is removed, the reasons for that are:
    # 1. this metric can not accept multiple ouptuts
    # 2. we seldom use this metric.
    self.validMetrics = ['mean_absolute_error', 'explained_variance_score', 'r2_score', 'mean_squared_error']
    self.invalidRom = ['GaussPolynomialRom', 'HDMRRom']
    self.cvID = 'RAVEN_CV_ID'

  def initialize(self, runInfo, inputs, initDict=None) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    super().initialize(runInfo, inputs, initDict)

    for metricIn in self.assemblerDict['Metric']:
      if metricIn[2] in self.metricsDict.keys():
        self.metricsDict[metricIn[2]] = metricIn[3]

    if list(self.metricsDict.values()).count(None) != 0:
      metricName = self.metricsDict.keys()[list(self.metricsDict.values()).index(None)]
      self.raiseAnError(IOError, "Missing definition for Metric: ", metricName)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    self.initializationOptionDict = {}
    scoreList = ['maximum', 'average', 'median']
    cvNode = paramInput.findFirst('SciKitLearn')
    for child in cvNode.subparts:
      if child.getName() == 'scores':
        score = child.value.strip().lower()
        if score in scoreList:
          self.cvScore = score
        else:
          self.raiseAnError(IOError, "Unexpected input '", score, "' for XML node 'scores'! Valid inputs include: ", ",".join(scoreList))
        break
    for child in paramInput.subparts:
      if child.getName() == 'SciKitLearn':
        self.initializationOptionDict[child.getName()] = self._localInputAndCheckParam(child)
        self.initializationOptionDict[child.getName()].pop("scores",'False')
      elif child.getName() == 'Metric':
        if 'type' not in child.parameterValues or 'class' not in child.parameterValues:
          self.raiseAnError(IOError, 'Tag Metric must have attributes "class" and "type"')
        else:
          metricName = child.value.strip()
          self.metricsDict[metricName] = None
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.getName(), " is provided for metric system")

  def _localInputAndCheckParam(self, inputParam):
    """
      Function to read the portion of the xml input
      @ In, inputParam, ParameterInput, the xml element node that will be checked against the available options specific to this Sampler
      @ Out, initDict, dict, dictionary contains the information about the given xml node
    """
    initDict = {}
    for child in inputParam.subparts:
      if len(child.parameterValues) > 0:
        initDict[child.getName()] = dict(child.parameterValues)
      else:
        initDict[child.getName()] = child.value
    return initDict

  def inputToInternal(self, currentInp, full = False):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, list or DataObject, data object or a list of data objects
      @ In, full, bool, optional, True to retrieve the whole input or False to get the last element of the input
        TODO, full should be removed
      @ Out, newInputs, tuple, (dictionary of input and output data, instance of estimator)
    """
    if type(currentInp) != list:
      self.raiseAnError(IOError, "A list of inputs is required by postprocessor", self.name, "while got", type(currentInp))
    if len(currentInp) != 2:
      self.raiseAnError(IOError, "Two inputs are required by postprocessor", self.name, "while the number of provided inputs is", len(currentInp))
    # This postprocessor accepts one input of Models.ROM
    cvEstimator = None
    inpDs = None
    for inp in currentInp:
      if isinstance(inp, Models.ROM):
        if inp.amITrained:
          inp.raiseAnError(RuntimeError, "ROM model '%s' has been already trained! " %inp.name +\
                                                  "Cross validation will not be performed")
        if not cvEstimator:
          cvEstimator = inp
        else:
          self.raiseAnError(IOError, "This postprocessor '%s' only accepts one input of Models.ROM!" %self.name)
      else:
        inpDs = inp
    inputType = None
    if hasattr(inpDs, 'type'):
      inputType = inpDs.type

    if isinstance(inpDs, Files.File):
      self.raiseAnError(IOError, "File object can not be accepted as an input")
    if inputType == 'HDF5':
      self.raiseAnError(IOError, "Input type '", inputType, "' can not be accepted")

    if type(inpDs) != dict:
      dictKeys = cvEstimator._interfaceROM.features + cvEstimator._interfaceROM.target
      newInput = dict.fromkeys(dictKeys, None)
      if not len(inpDs) == 0:
        dataSet = inpDs.asDataset()
        if inputType == 'PointSet':
          for elem in inpDs.getVars('input') + inpDs.getVars('output'):
            if elem in newInput.keys():
              newInput[elem] = copy.copy(dataSet[elem].values)
        elif inputType == 'HistorySet':
          sizeIndex = 0
          for hist in range(len(inpDs)):
            for elem in inpDs.indexes + inpDs.getVars('outputs'):
              if elem in newInput.keys():
                if newInput[elem] is None:
                  newInput[elem] = []
                newInput[elem].append(dataSet.isel(RAVEN_sample_ID=hist)[elem].values)
                sizeIndex = len(newInput[elem][-1])
            for elem in inpDs.getVars('input'):
              if elem in newInput.keys():
                if newInput[elem] is None:
                  newInput[elem] = []
                newInput[elem].append(np.full((sizeIndex,), dataSet.isel(RAVEN_sample_ID=hist)[elem].values))
        else:
          self.raiseAnError(IOError, "The input type '", inputType, "' can not be accepted")
    else:
      #here we do not make a copy since we assume that the dictionary is for just for the model usage and any changes are not impacting outside
      newInput = inpDs

    if any(x is None for x in newInput.values()):
      varName = newInput.keys()[list(newInput.values()).index(None)]
      self.raiseAnError(IOError, "The variable: ", varName, " is not exist in the input: ", currentInput.name, " which is required for model: ", cvEstimator.name)

    newInputs = newInput, cvEstimator
    return newInputs

  #FIXME: Temporary method. Need to be rethought when the new Hybrid Model is developed
  def _returnCharacteristicsOfCvGivenOutputName(self,outputName):
    """
      Method to return the metric name, type and target name given the output name
      @ In, outputName, str, the output name
      @ Out, info, dict, the dictionary containing the info
    """
    assert(len(outputName.split("_")) == 3)
    info = {}
    _, info['metricName'], info['targetName']  = outputName.split("_")
    info['metricType'] = self.metricsDict[info['metricName']].getAlgorithmType()
    return info

  def __generateTrainTestInputs(self, inputDict, trainIndex, testIndex):
    """
      Genenerate train and test set based on the given train index and test index
      @ In, inputDict, dict, dictionary of input and output data
      @ In, trainIndex, numpy.ndarray, indices of training set
      @ In, testIndex, numpy.ndarray, indices of test set
      @ Out, trainTest, tuple, (dictionary of train set, dictionary of test set)
    """
    trainInput = dict.fromkeys(inputDict.keys(), None)
    testInput = dict.fromkeys(inputDict.keys(), None)
    for key, value in inputDict.items():
      if np.asarray(value).size != trainIndex.size + testIndex.size:
        self.raiseAnError(IOError, "The number of samples provided in the input is not equal the number of samples used in the cross-validation: "+str(np.asarray(value).size) +"!="+str(trainIndex.size + testIndex.size))
      trainInput[key] = np.asarray(value)[trainIndex]
      testInput[key] = np.asarray(value)[testIndex]
    trainTest = trainInput, testInput
    return trainTest

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, list, list of objects, i.e. the object contained the data to process, the instance of model.
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputDict, cvEstimator = self.inputToInternal(inputIn, full = True)
    if cvEstimator.subType in self.invalidRom:
      self.raiseAnError(IOError, cvEstimator.subType, " can not be retrained, thus can not be used in Cross Validation post-processor ", self.name)
    if self.dynamic:
      self.raiseAnError(IOError, "Not implemented yet")
    initDict = copy.deepcopy(self.initializationOptionDict)
    cvEngine = None
    groups = None
    for key, value in initDict.items():
      if key == "SciKitLearn":
        groups = value.pop("labels",None)
        cvEngine = CrossValidations.factory.returnInstance(key, **value)
        break
    if cvEngine is None:
      self.raiseAnError(IOError, "No cross validation engine is provided!")
    outputDict = {}
    # In SciKit-Learn (version > 0.18), module model_selection is used to perform cross validation
    # A wrapper in RAVEN is created, and the method .split is replaced with generateTrainTestIndices
    # In the old version, 'labels' is used for the label-related cross validation. In the new versions
    # Both keywords 'y' and 'groups' can be used to specify the labels. The keyword 'y' is mainly used by
    # SciKit-Learn supervised learning problems, and 'groups' become additional option to specify the group
    # labels that can be used while splitting the dataset into train/test set. For our purpose, only one
    # label option is needed. ~ wangc
    for trainIndex, testIndex in cvEngine.generateTrainTestIndices(list(inputDict.values())[0], y=groups, groups=groups):
      trainDict, testDict = self.__generateTrainTestInputs(inputDict, trainIndex, testIndex)
      ## Train the rom
      cvEstimator.train(trainDict)
      ## evaluate the rom
      outputEvaluation = cvEstimator.evaluate(testDict)
      ## Compute the distance between ROM and given data using Metric system
      for targetName, targetValue in outputEvaluation.items():
        for metricInstance in self.metricsDict.values():
          metricValue = metricInstance.evaluate(targetValue, testDict[targetName])
          metricType = metricInstance.getAlgorithmType()
          if metricType is not None:
            if metricType[1] not in self.validMetrics:
              self.raiseAnError(IOError,
                  f'The metric type "{metricType[1]}" cannot be used. ' +
                  f'Accepted metric types include: {", ".join(self.validMetrics)}.')
            metricName = metricType[1]
          else:
            self.raiseAnError(IOError,
                f'The metric named "{metricInstance.name}" is not a valid type. ' +
                f'Accepted metric types include: {", ".join(self.validMetrics)}.')
          varName = f'cv_{metricInstance.name}_{targetName}'
          if varName not in outputDict.keys():
            outputDict[varName] = np.array([])
          outputDict[varName] = np.append(outputDict[varName], metricValue)
    scoreDict = {}
    if not self.cvScore:
      return outputDict
    else:
      for varName, metricValues in outputDict.items():
        if self.cvScore.lower() == 'maximum':
          scoreDict[varName] = np.atleast_1d(np.amax(np.atleast_1d(metricValues)))
        elif self.cvScore.lower() == 'median':
          scoreDict[varName] = np.atleast_1d(np.median(np.atleast_1d(metricValues)))
        elif self.cvScore.lower() == 'average':
          scoreDict[varName] = np.atleast_1d(np.mean(np.atleast_1d(metricValues)))
      return scoreDict

  def collectOutput(self,finishedJob, output, options=None):
    """
      Function to place all of the computed data into the output object, i.e. Files
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict = evaluation[1]
    if self.cvScore is not None:
      output.addRealization(outputDict)
    else:
      cvIDs = {self.cvID: np.atleast_1d(range(len(utils.first(outputDict.values()))))}
      outputDict.update(cvIDs)
      output.load(outputDict, style='dict')
