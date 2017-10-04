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
Created on August 23, 2017

@author: wangc
"""
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import numpy as np
import os
from collections import OrderedDict
import itertools
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData
import Files
import Metrics
import Runners
import Distributions
#Internal Modules End--------------------------------------------------------------------------------

class Metric(PostProcessor):
  """
    Metrics class.
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
    inputSpecification = super(Metric, cls).getInputSpecification()
    FeaturesInput = InputData.parameterInputFactory("Features", contentType=InputData.StringType)
    FeaturesInput.addParam("type", InputData.StringType)
    inputSpecification.addSub(FeaturesInput)
    TargetsInput = InputData.parameterInputFactory("Targets", contentType=InputData.StringType)
    TargetsInput.addParam("type", InputData.StringType)
    inputSpecification.addSub(TargetsInput)
    MetricInput = InputData.parameterInputFactory("Metric", contentType=InputData.StringType)
    MetricInput.addParam("class", InputData.StringType, True)
    MetricInput.addParam("type", InputData.StringType, True)
    inputSpecification.addSub(MetricInput)

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR Metrics'
    self.dynamic        = False # is it time-dependent?
    self.features       = None  # list of feature variables
    self.targets        = None  # list of target variables
    self.metricsDict    = {}    # dictionary of metrics that are going to be assembled
    self.pivotParameter = None
    # assembler objects to be requested
    self.addAssemblerObject('Metric', 'n', True)

  def __getMetricSide(self, metricDataName, currentInputs):
    """
      Gets the metricDataName and stores it in inputDict.
      @ In, metricDataName, string, the name of the metric data to find in currentInputs
      @ In, currentInputs, list of inputs to the step.
      @ Out, metricData, (data, probability) or Distribution
    """
    origMetricDataName = metricDataName
    metricData = None
    if metricDataName.count("|") == 2:
      #Split off the data name and if this is input or output.
      dataName, inputOrOutput, metricDataName = metricDataName.split("|")
      inputOrOutput = [inputOrOutput.lower()]
    else:
      dataName = None
      inputOrOutput = ['input','output']
    for currentInput in currentInputs:
      inputType = None
      if hasattr(currentInput, 'type'):
        inputType = currentInput.type

      if inputType == 'PointSet':
        if dataName is not None and dataName != currentInput.name:
          #The dataname is not a match
          continue
        metadata = currentInput.getAllMetadata()
        for ioType in inputOrOutput:
          if metricDataName in currentInput.getParaKeys(ioType):
            if metricData is not None:
              self.raiseAnError(IOError, "Same feature or target variable " + metricDataName + "is found in multiple input objects")
            #Found the data, now put it in the return value.
            metricData = (currentInput.getParam(ioType, metricDataName, nodeId = 'ending'), metadata['ProbabilityWeight'])
      elif isinstance(currentInput, Distributions.Distribution):
        if currentInput.name == metricDataName and dataName is None:
          if metricData is not None:
            self.raiseAnError(IOError, "Same feature or target variable " + metricDataName + "is found in multiple input objects")
          #Found the distribution, now put it in the return value
          metricData = currentInput
    if metricData is None:
      self.raiseAnError(IOError, "Feature or target variable " + origMetricDataName + "is not found")
    return metricData

  def inputToInternal(self, currentInputs):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInputs, list or DataObject, data object or a list of data objects
      @ Out, measureList, list of (feature, target), the list of the features and targets to measure the distance between
    """
    if type(currentInputs) == dict and 'features' in currentInputs.keys():
      return currentInputs

    if type(currentInputs) != list:
      currentInputs = [currentInputs]

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
        pass #Allowed type
      elif inputType == 'HistorySet':
        self.dynamic = True
        self.raiseAnError(IOError, "Metric can not process HistorySet, because this capability is not implemented yet")
      else:
        self.raiseAnError(IOError, "Metric cannot process "+inputType+ " of type "+str(type(currentInput)))

    measureList = []

    if not self.dynamic:
      for cnt in range(len(self.features)):
        feature = self.features[cnt]
        target = self.targets[cnt]
        featureData =  self.__getMetricSide(feature, currentInputs)
        targetData = self.__getMetricSide(target, currentInputs)
        measureList.append((featureData, targetData))
    else:
      self.raiseAnError(IOError, "Dynamic not implemented yet")


    return measureList

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)
    for metricIn in self.assemblerDict['Metric']:
      self.metricsDict[metricIn[2]] = metricIn[3]

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs
      @ In, xmlNode, xml.etree.ElementTree Element Objects, the xml element node that will be checked against the available options specific to this Sampler
      @ Out, None
    """
    paramInput = Metric.getInputSpecification()()
    paramInput.parseNode(xmlNode)

    for child in paramInput.subparts:
      if child.getName() == 'Metric':
        if 'type' not in child.parameterValues.keys() or 'class' not in child.parameterValues.keys():
          self.raiseAnError(IOError, 'Tag Metric must have attributes "class" and "type"')
      elif child.getName() == 'Features':
        self.features = list(var.strip() for var in child.value.split(','))
        self.featuresType = child.parameterValues['type']
      elif child.getName() == 'Targets':
        self.targets = list(var.strip() for var in child.value.split(','))
        self.TargetsType = child.parameterValues['type']
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.getName(), " is provided for metric system")

    if not self.features:
      self.raiseAnError(IOError, "XML node 'Features' is required but not provided")
    elif len(self.features) != len(self.targets):
      self.raiseAnError(IOError, 'The number of variables found in XML node "Features" is not equal the number of variables found in XML node "Targets"')

  def collectOutput(self,finishedJob, output):
    """
      Function to place all of the computed data into the output object, (Files or DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "Job ", finishedJob.identifier, "failed!")
    outputDict = evaluation[1]

    if isinstance(output, Files.File):
      availExtens = ['xml', 'csv']
      outputExtension = output.getExt().lower()
      if outputExtension not in availExtens:
        self.raiseAMessage('Metric postprocessor did not recognize extension ".', str(outputExtension), '". The output will be dumped to a text file')
      output.setPath(self._workingDir)
      self.raiseADebug('Write Metric prostprocessor output in file with name: ', output.getAbsFile())
      output.open('w')
      if outputExtension == 'xml':
        self._writeXML(output, outputDict)
      else:
        separator = ' ' if outputExtension != 'csv' else ','
        self._writeText(output, outputDict, separator)
    else:
      self.raiseAnError(IOError, 'Output type ', str(output.type), ' can not be used for postprocessor', self.name)

  def _writeXML(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, None
    """
    if output.isOpen():
      output.close()
    if self.dynamic:
      outputInstance = Files.returnInstance('DynamicXMLOutput', self)
    else:
      outputInstance = Files.returnInstance('StaticXMLOutput', self)
    outputInstance.initialize(output.getFilename(), self.messageHandler, path=output.getPath())
    outputInstance.newTree('MetricPostProcessor', pivotParam=self.pivotParameter)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for ts, outputDict in enumerate(outputResults):
      pivotVal = outputDictionary.keys()[ts]
      for nodeName, nodeValues in outputDict.items():
        for metricName, value in nodeValues.items():
          if type(value) == float:
            outputInstance.addScalar(nodeName, metricName, value, pivotVal=pivotVal)
          elif type(value) in [list, np.ndarray]:
            if len(list(value)) == 1:
              outputInstance.addScalar(nodeName, metricName, value[0], pivotVal=pivotVal)
            else:
              self.raiseAnError(IOError, "Multiple values are returned from metric '", metricName, "', this is currently not allowed")
          elif type(value) == dict:
            ## FIXME: The following are used to accept timedependent data, and should be checked later.
            outputInstance.addVector(nodeName, metricName, value, pivotVal=pivotVal)
          else:
            self.raiseAnError(IOError, "Unrecognized type of input value '", type(value), "'")
    outputInstance.writeFile()

  def _writeText(self, output, outputDictionary, separator=' '):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores metric outputs
      @ In, separator, string, optional, separator string
      @ Out, None
    """
    if self.dynamic:
      output.write('Dynamic Metric', separator, 'Pivot Parameter', separator, self.pivotParameter, separator, os.linesep)
      self.raiseAnError(IOError, 'The method to dump the dynamic metric into a CSV file is not implemented yet!')
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for ts, outputDict in enumerate(outputResults):
      if self.dynamic:
        output.write('Pivot value', separator, str(outputDictionary.keys()[ts]), os.linesep)
      for nodeName, nodeValues in outputDict.items():
        output.write('Metrics' + separator)
        output.write(nodeName + os.linesep)
        for metricName, value in nodeValues.items():
          output.write(metricName+separator)
          if type(value) == float:
            output.write(str(value) + os.linesep)
          elif type(value) in [list, np.ndarray]:
            if len(list(value)) == 1:
              output.write(str(value[0]) + os.linesep)
            else:
              self.raiseAnError(IOError, "Multiple values are returned from metric '", metricName, "', this is currently not allowed")
          else:
            self.raiseAnError(IOError, "Unrecognized type of input value '", type(value), "'")


  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    measureList = self.inputToInternal(inputIn)
    outputDict = OrderedDict()
    if not self.dynamic:
      assert len(self.features) == len(measureList)
      for cnt in range(len(self.features)):
        nodeName = (str(self.features[cnt]) + '-' + str(self.targets[cnt])).replace("|","_")
        outputDict[nodeName] = {}
        for metricInstance in self.metricsDict.values():
          inData = list(measureList[cnt])
          metricCanHandleData = True
          if hasattr(metricInstance, 'metricType'):
            metricName = metricInstance.metricType
          else:
            metricName = metricInstance.type
          metricName = metricInstance.name + '_' + metricName
          for i in range(len(inData)):
            if not metricInstance.acceptsProbability and type(inData[i]) == tuple:
              #Strip off the probability data if it can't be used
              inData[i] = inData[i][0]
            elif not metricInstance.acceptsDistribution and isinstance(inData[i], Distributions.Distribution):
              metricCanHandleData = False
              self.raiseAWarning('Cannot handle '+nodeName+' with metric '+metricName +' because it contains a distribution')
          if metricCanHandleData:
            output = metricInstance.distance(inData[0], inData[1])
            outputDict[nodeName][metricName] = output
    else:
      self.raiseAnError(IOError, "Not implemented yet")
    return outputDict

