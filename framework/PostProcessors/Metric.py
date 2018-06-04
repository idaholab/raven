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
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData
from utils.cached_ndarray import c1darray
import Files
import Metrics
import Runners
import Distributions
import MetricDistributor
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
    MultiOutputInput = InputData.parameterInputFactory("multiOutput", contentType=InputData.StringType)
    inputSpecification.addSub(MultiOutputInput)
    WeightInput = InputData.parameterInputFactory("weight", contentType=InputData.StringType)
    inputSpecification.addSub(WeightInput)
    PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
    inputSpecification.addSub(PivotParameterInput)
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
    self.multiOutput    = 'mean'# defines aggregating of multiple outputs for HistorySet
                                # currently allow mean, max, min, raw_values
    self.weight         = None  # 'mean' is provided for self.multiOutput, weights can be used
                                # for each individual output when all outputs are averaged
    self.pivotParameter = None
    self.pivotValues    = []
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
      inputOrOutput = ['inputs','outputs']
    for currentInput in currentInputs:
      inputType = None
      if hasattr(currentInput, 'type'):
        inputType = currentInput.type

      if dataName is not None and dataName != currentInput.name:
        #The dataname is not a match
        continue

      if inputType in ['PointSet', 'HistorySet']:
        dataSet = currentInput.asDataset()
        metadata = currentInput.getMeta(pointwise=True)
        for ioType in inputOrOutput:
          if metricDataName in currentInput.getVars(ioType):
            if metricData is not None:
              self.raiseAnError(IOError, "Same feature or target variable " + metricDataName + "is found in multiple input objects")
            #Found the data, now put it in the return value.
            requestData = copy.copy(dataSet[metricDataName].values)
            if len(requestData.shape) == 1:
              requestData = requestData.reshape(-1,1)
            # If requested data are from input space, the shape will be (nSamples, 1)
            # If requested data are from history output space, the shape will be (nSamples, nTimeSteps)
            metricData = (requestData, metadata['ProbabilityWeight'].values)
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
          if self.pivotParameter not in currentInput.getParaKeys('output'):
            self.raiseAnError(IOError, self, 'Pivot parameter', self.pivotParameter,'has not been found in DataObject', currentInput.name)
          outputs = currentInput.getParametersValues('outputs', nodeId='ending')
          numSteps = len(outputs.values()[0].values()[0])
          pivotValues = []
          for step in range(len(outputs.values()[0][self.pivotParameter])):
            currentSnapShot = [outputs[i][self.pivotParameter][step] for i in outputs.keys()]
            if len(set(currentSnapShot)) > 1:
              self.raiseAnError(IOError, "HistorySet", currentInput.name," is not syncronized, please use Interfaced PostProcessor HistorySetSync to pre-process it")
            pivotValues.append(currentSnapShot[-1])
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
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
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
      elif child.getName() == 'multiOutput':
        self.multiOutput = child.value.strip()
      elif child.getName() == 'weight':
        self.weight = np.asarray(list(float(var) for var in child.value.split(',')))
      elif child.getName() == 'pivotParameter':
        self.pivotParameter = child.value.strip()
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
    # FIXME store the data in DataObjects
    output.addRealization(outputDict)

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
    if self.dynamic:
      for ts, pivotVal in enumerate(self.pivotValues):
        for metricName, metricValues in outputDictionary.items():
          for nodeName, nodeValues in metricValues.items():
            if type(nodeValues) in [list, np.ndarray]:
              outputInstance.addScalar(nodeName, metricName,nodeValues[ts], pivotVal=pivotVal)
            else:
              self.raiseAnError(IOError, "Invalid format for the return output dictionary")
    else:
      for metricName, metricValues in outputDictionary.items():
        for nodeName, nodeValues in metricValues.items():
          if type(nodeValues) == float:
            outputInstance.addScalar(nodeName, metricName, nodeValues)
          elif type(nodeValues) in [list, np.ndarray]:
            if len(list(nodeValues)) == 1:
              outputInstance.addScalar(nodeName, metricName, nodeValues[0])
            else:
              self.raiseAnError(IOError, "Multiple values are returned from metric '", metricName, "', this is currently not allowed")
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
<<<<<<< HEAD
    if self.dynamic:
      output.write('Dynamic Metric', separator, 'Pivot Parameter', separator, self.pivotParameter, separator, os.linesep)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
=======
    outputResults = [outputDictionary]
>>>>>>> add options to output time-dependent metrics results
    for ts, outputDict in enumerate(outputResults):
      if self.dynamic:
        output.write('Pivot value' + separator + str(outputDictionary.keys()[ts]) + os.linesep)
      for metricName, nodeValues in outputDict.items():
        output.write('Metrics' + separator)
        output.write(metricName + os.linesep)
        for nodeName, value in nodeValues.items():
          output.write(nodeName)
          if type(value) == float:
            output.write(str(value) + os.linesep)
          elif type(value) in [list, np.ndarray]:
            if len(list(value)) == 1:
              output.write(str(value[0]) + os.linesep)
            else:
              output.write(''.join( [separator + str(item) for item in value]) + os.linesep)
              #self.raiseAnError(IOError, "Multiple values are returned from metric '", metricName, "', this is currently not allowed")
          else:
            self.raiseAnError(IOError, "Unrecognized type of input value '", type(value), "'")

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    measureList = self.inputToInternal(inputIn)
    outputDict = {}
    assert len(self.features) == len(measureList)
<<<<<<< HEAD
    for cnt in range(len(self.features)):
      nodeName = (str(self.features[cnt]) + '-' + str(self.targets[cnt])).replace("|","_")
      for metricInstance in self.metricsDict.values():
        if hasattr(metricInstance, 'metricType'):
          metricName = "_".join(metricInstance.metricType)
        else:
          metricName = metricInstance.type
        metricName = metricInstance.name + '_' + metricName
        varName = metricName + '|' + nodeName
        metricEngine = MetricDistributor.returnInstance('MetricDistributor',metricInstance,self)
        output = metricEngine.evaluate(measureList[cnt], weights=None, multiOutput='mean')
        outputDict[varName] = np.atleast_1d(output)

=======
    for metricInstance in self.metricsDict.values():
      if hasattr(metricInstance, 'metricType'):
        metricName = "_".join(metricInstance.metricType)
      else:
        metricName = metricInstance.type
      metricName = metricInstance.name + '_' + metricName
      outputDict[metricName] = {}
      metricEngine = MetricDistributor.returnInstance('MetricDistributor',metricInstance,self)
      for cnt in range(len(self.features)):
        nodeName = (str(self.features[cnt]) + '-' + str(self.targets[cnt])).replace("|","_")
        output = metricEngine.evaluate(measureList[cnt], weights=self.weight, multiOutput=self.multiOutput)
        outputDict[metricName][nodeName] = output
>>>>>>> add options to output time-dependent metrics results
    return outputDict
