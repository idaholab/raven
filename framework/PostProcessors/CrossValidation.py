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
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import numpy as np
import os
import six
from collections import OrderedDict
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData
import Files
import Models
import Runners
import CrossValidations
#Internal Modules End--------------------------------------------------------------------------------

class CrossValidation(PostProcessor):
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

    ## TODO: fill this in with the appropriate tags
    #MetricInput = InputData.parameterInputFactory("Metric", contentType=InputData.StringType)
    #MetricInput.addParam("class", InputData.StringType)
    #MetricInput.addParam("type", InputData.StringType)
    #inputSpecification.addSub(MetricInput)

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR CROSS VALIDATION'
    self.dynamic        = False # is it time-dependent?
    self.metricsDict    = {}    # dictionary of metrics that are going to be assembled
    self.pivotParameter = None
    self.averageScores  = False
    # assembler objects to be requested
    self.addAssemblerObject('Metric', 'n', True)
    self.CVEstimator    = None   # instance of estimator that is used to for Cross Validation
    self.CVEngine       = None   # Engine used for cross validation

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)

    for metricIn in self.assemblerDict['Metric']:
      if metricIn[2] in self.metricsDict.keys():
        self.metricsDict[metricIn[2]] = metricIn[3]

    if self.metricsDict.values().count(None) != 0:
      metricName = self.metricsDict.keys()[list(self.metricsDict.values()).index(None)]
      self.raiseAnError(IOError, "Missing definition for Metric: ", metricName)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs
      @ In, xmlNode, xml.etree.ElementTree Element Objects, the xml element node that will be checked against
        the available options specific to this Sampler
      @ Out, None
    """
    #paramInput = CrossValidation.getInputSpecification()()
    #paramInput.parseNode(xmlNode)

    self.initializationOptionDict = {}
    average = None
    cvNode = xmlNode.find('SciKitLearn')
    for child in cvNode:
      if child.tag == 'average':
        average = child
        break
    if average is not None:
      self.averageScores = average.text.lower() in utils.stringsThatMeanTrue()
    for child in xmlNode:
      if child.tag == 'SciKitLearn':
        self.initializationOptionDict[child.tag] = self._localInputAndCheck(child)
        self.initializationOptionDict[child.tag].pop("average",'False')
        self.CVEngine = CrossValidations.returnInstance(child.tag, self, **self.initializationOptionDict[child.tag])
      elif child.tag == 'Metric':
        if 'type' not in child.attrib.keys() or 'class' not in child.attrib.keys():
          self.raiseAnError(IOError, 'Tag Metric must have attributes "class" and "type"')
        else:
          metricName = child.text.strip()
          self.metricsDict[metricName] = None
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.tag, " is provided for metric system")

  def _localInputAndCheck(self, xmlNode):
    """
      Function to read the portion of the xml input
      @ In, xmlNode, xml.etree.ElementTree Element Objects, the xml element node that will be checked against the available options specific to this Sampler
      @ Out, initDict, dict, dictionary contains the information about the given xml node
    """
    initDict = {}
    for child in xmlNode:
      if child.attrib:
        initDict[child.tag] = dict(child.attrib)
      else:
        initDict[child.tag] = utils.tryParse(child.text)
    return initDict

  def inputToInternal(self, currentInp, full = False):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, list or DataObject, data object or a list of data objects
      @ In, full, bool, optional, True to retrieve the whole input or False to get the last element of the input
      @ Out, newInputs, tuple, (dictionary of input and output data, instance of estimator)
    """
    if type(currentInp) != list:
      self.raiseAnError(IOError, "Only one input is provided for postprocessor", self.name, "while two inputs are required")
    else:
      currentInputs = copy.deepcopy(currentInp)

    # This postprocessor accepts one input of Models.ROM
    cvEstimator = None
    for currentInput in currentInputs:
      if isinstance(currentInput, Models.ROM):
        if currentInput.amITrained:
          currentInput.raiseAnError(RuntimeError, "ROM model '%s' has been already trained! " %currentInput.name +\
                                                  "Cross validation will not be performed")
        if not cvEstimator:
          cvEstimator = currentInput
        else:
          self.raiseAnError(IOError, "This postprocessor '%s' only accepts one input of Models.ROM!" %self.name)

    currentInputs.remove(cvEstimator)

    currentInput = copy.deepcopy(currentInputs[-1])
    inputType = None
    if hasattr(currentInput, 'type'):
      inputType = currentInput.type

    if isinstance(currentInput, Files.File):
      self.raiseAnError(IOError, "File object can not be accepted as an input")
    if inputType == 'HDF5':
      self.raiseAnError(IOError, "Input type '", inputType, "' can not be accepted")

    if type(currentInput) != dict:
      dictKeys = list(cvEstimator.initializationOptionDict['Features'].split(',')) + list(cvEstimator.initializationOptionDict['Target'].split(','))
      newInput = dict.fromkeys(dictKeys, None)
      if not currentInput.isItEmpty():
        if inputType == 'PointSet':
          for elem in currentInput.getParaKeys('inputs'):
            if elem in newInput.keys():
              newInput[elem] = copy.copy(np.array(currentInput.getParam('input', elem))[0 if full else -1:])
          for elem in currentInput.getParaKeys('outputs'):
            if elem in newInput.keys():
              newInput[elem] = copy.copy(np.array(currentInput.getParam('output', elem))[0 if full else -1:])
        elif inputType == 'HistorySet':
          if full:
            for hist in range(len(currentInput)):
              realization = currentInput.getRealization(hist)
              for elem in currentInput.getParaKeys('inputs'):
                if elem in newInput.keys():
                  if newInput[elem] is None:
                    newInput[elem] = c1darray(shape = (1,))
                  newInput[elem].append(realization['inputs'][elem])
              for elem in currentInput.getParaKeys('outputs'):
                if elem in newInput.keys():
                  if newInput[elem] is None:
                    newInput[elem] = []
                  newInput[elem].append(realization['outputs'][elem])
          else:
            realization = currentInput.getRealization(len(currentInput) - 1)
            for elem in currentInput.getParaKeys('inputs'):
              if elem in newInput.keys():
                newInput[elem] = [realization['inputs'][elem]]
            for elem in currentInput.getParaKeys('outputs'):
              if elem in newInput.keys():
                newInput[elem] = [realization['outputs'][elem]]
        else:
          self.raiseAnError(IOError, "The input type '", inputType, "' can not be accepted")
      #Now if an OutputPlaceHolder is used it is removed, this happens when the input data is not representing is internally manufactured
      if 'OutputPlaceHolder' in currentInput.getParaKeys('outputs'):
        # this remove the counter from the inputs to be placed among the outputs
        newInput.pop('OutputPlaceHolder')
    else:
      #here we do not make a copy since we assume that the dictionary is for just for the model usage and any changes are not impacting outside
      newInput = currentInput

    if any(x is None for x in newInput.values()):
      varName = newInput.keys()[list(newInput.values()).index(None)]
      self.raiseAnError(IOError, "The variable: ", varName, " is not exist in the input: ", currentInput.name, " which is required for model: ", cvEstimator.name)

    newInputs = newInput, cvEstimator
    return newInputs

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
    inputDict, self.CVEstimator = self.inputToInternal(inputIn, full = True)
    outputDict = {}

    if self.dynamic:
      self.raiseAnError(IOError, "Not implemented yet")

    for trainIndex, testIndex in self.CVEngine.generateTrainTestIndices():
      trainDict, testDict = self.__generateTrainTestInputs(inputDict, trainIndex, testIndex)
      cvEstimator = copy.deepcopy(self.CVEstimator)
      ## Train the rom
      cvEstimator.train(trainDict)
      ## evaluate the rom
      outputEvaluation = cvEstimator.evaluate(testDict)
      ## Compute the distance between ROM and given data using Metric system
      for targetName, targetValue in outputEvaluation.items():
        if targetName not in outputDict.keys():
          outputDict[targetName] = {}
        for metricInstance in self.metricsDict.values():
          metricValue = metricInstance.distance(targetValue, testDict[targetName])
          if hasattr(metricInstance, 'metricType'):
            metricName = metricInstance.metricType
          else:
            metricName = metricInstance.type
          if metricName not in outputDict[targetName].keys():
            outputDict[targetName][metricName] = []
          outputDict[targetName][metricName].append(metricValue[0])
    if self.averageScores:
      for targetName in outputEvaluation.keys():
        for metricInstance in self.metricsDict.values():
          metricName = metricInstance.metricType if hasattr(metricInstance, 'metricType') else metricInstance.type
          outputDict[targetName][metricName] = [np.atleast_1d(outputDict[targetName][metricName]).mean()]
    return outputDict

  def collectOutput(self,finishedJob, output):
    """
      Function to place all of the computed data into the output object, i.e. Files
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, ' No available output to collect')
    outputDict = evaluation[1]

    if isinstance(output, Files.File):
      availExtens = ['xml', 'csv']
      outputExtension = output.getExt().lower()
      if outputExtension not in availExtens:
        self.raiseAMessage('Cross Validation postprocessor did not recognize extension ".', str(outputExtension), '". The output will be dumped to a text file')
      output.setPath(self._workingDir)
      self.raiseADebug('Write Cross Validation prostprocessor output in file with name: ', output.getAbsFile())
      output.open('w')
      if outputExtension == 'xml':
        self._writeXML(output, outputDict)
      else:
        separator = ' ' if outputExtension != 'csv' else ','
        self._writeText(output, outputDict, separator)
    elif output.type == 'PointSet':
      self._writeDataObject(output, outputDict)
    else:
      self.raiseAnError(IOError, 'Output type ', str(output.type), ' can not be used for postprocessor', self.name)

  def _writeXML(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores cross validation scores
      @ Out, None
    """
    if output.isOpen():
      output.close()
    if self.dynamic:
      outputInstance = Files.returnInstance('DynamicXMLOutput', self)
    else:
      outputInstance = Files.returnInstance('StaticXMLOutput', self)
    outputInstance.initialize(output.getFilename(), self.messageHandler, path=output.getPath())
    outputInstance.newTree('CrossValidationPostProcessor', pivotParam=self.pivotParameter)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for ts, outputDict in enumerate(outputResults):
      pivotVal = outputDictionary.keys()[ts]
      for nodeName, nodeValues in outputDict.items():
        for metricName, metricValues in nodeValues.items():
          cvRuns = ['cv-' + str(i) for i in range(len(metricValues))]
          valueDict = dict(zip(cvRuns, metricValues))
          outputInstance.addVector(nodeName, metricName, valueDict, pivotVal = pivotVal)
    outputInstance.writeFile()

  def _writeText(self,output,outputDictionary, separator=' '):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores metrics' results of outputs
      @ In, separator, string, optional, separator string
      @ Out, None
    """
    if self.dynamic:
      output.write('Dynamic Cross Validation', separator, 'Pivot Parameter', separator, self.pivotParameter, separator, os.linesep)
      self.raiseAnError(IOError, 'The method to dump the dynamic cross validation results into a csv file is not implemented yet')

    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for ts, outputDict in enumerate(outputResults):
      if self.dynamic:
        output.write('Pivot value', separator, str(outputDictionary.keys()[ts]), os.linesep)
      nodeNames, nodeValues = outputDict.keys(), outputDict.values()
      metricNames = nodeValues[0].keys()
      output.write('CV-Run-Number')
      for nodeName in nodeNames:
        for metricName in metricNames:
          output.write(separator + nodeName + '-' + metricName)
      output.write(os.linesep)
      for cvRunNum in range(len(nodeValues[0].values()[0])):
        output.write(str(cvRunNum))
        for valueDict in nodeValues:
          for metricName in metricNames:
            output.write(separator+str(valueDict[metricName][cvRunNum]))
        output.write(os.linesep)

  def _writeDataObject(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a a PointSet
      @ In, output, PointSet object, PointSet to dump the results to
      @ In, outputDictionary, dict, dictionary stores metics' results of outputs
      @ Out, None
    """
    if self.dynamic:
      self.raiseAnError(IOError, 'The method to dump the dynamic cross validation results into a HistorySet file is not implemented yet')

    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    outputKeys = output.getParaKeys('outputs')
    inputKeys  = output.getParaKeys('inputs')
    keysToFill = dict.fromkeys(outputKeys+inputKeys)
    if 'CV-Run-Number' not in outputKeys + inputKeys:
      self.raiseAnError(Exception, "CV-Run-Number key is present neither in the <Input> nor <Output> nodes of the DataObject! Check your input!")
    keysToFill['CV-Run-Number'] = True
    for ts, outputDict in enumerate(outputResults):
      nodeNames, nodeValues = outputDict.keys(), outputDict.values()
      metricNames = nodeValues[0].keys()
      parameterNames = []
      for nodeName in nodeNames:
        for metricName in metricNames:
          parameterNames.append(nodeName + '-' + metricName)
      for cvRunNum in range(len(nodeValues[0].values()[0])):
        if 'CV-Run-Number' in outputKeys:
          output.updateOutputValue('CV-Run-Number',cvRunNum)
        else:
          output.updateInputValue('CV-Run-Number',cvRunNum)
        cnt = 0
        for valueDict in nodeValues:
          for metricName in metricNames:
            keysToFill[parameterNames[cnt]] = True
            if parameterNames[cnt] in outputKeys:
              output.updateOutputValue(parameterNames[cnt],valueDict[metricName][cvRunNum])
            elif parameterNames[cnt] in inputKeys:
              output.updateInputValue(parameterNames[cnt],valueDict[metricName][cvRunNum])
            else:
              keysToFill[parameterNames[cnt]] = False
            cnt+=1
    keysNotFilled = []
    for key, value in keysToFill.items():
      if not value:
        keysNotFilled.append(key)
    if len(keysNotFilled) > 0:
      self.raiseAnError(Exception, "The following keys are present in the DataObject but are not filled by the PostProcessor: "+",".join(keysNotFilled))
