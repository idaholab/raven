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
#External Modules------------------------------------------------------------------------------------
import numpy as np
import os
from collections import OrderedDict
import itertools
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...utils import xmlUtils
from ...utils import InputData, InputTypes
from ... import Files
from ... import Distributions
from ... import MetricDistributor
from .PostProcessorInterface import PostProcessorInterface
#Internal Modules End--------------------------------------------------------------------------------

class Metric(PostProcessorInterface):
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
    featuresInput = InputData.parameterInputFactory("Features", contentType=InputTypes.StringListType)
    featuresInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(featuresInput)
    targetsInput = InputData.parameterInputFactory("Targets", contentType=InputTypes.StringListType)
    targetsInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(targetsInput)
    multiOutputInput = InputData.parameterInputFactory("multiOutput", contentType=InputTypes.StringType)
    inputSpecification.addSub(multiOutputInput)
    multiOutput = InputTypes.makeEnumType('MultiOutput', 'MultiOutputType', ['mean','max','min','raw_values'])
    multiOutputInput = InputData.parameterInputFactory("multiOutput", contentType=multiOutput)
    inputSpecification.addSub(multiOutputInput)
    weightInput = InputData.parameterInputFactory("weight", contentType=InputTypes.FloatListType)
    inputSpecification.addSub(weightInput)
    pivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    inputSpecification.addSub(pivotParameterInput)
    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    metricInput.addParam("class", InputTypes.StringType, True)
    metricInput.addParam("type", InputTypes.StringType, True)
    inputSpecification.addSub(metricInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
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
    self.addAssemblerObject('Metric', InputData.Quantity.one_to_infinity)

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
      if dataName is not None and dataName != currentInput.name:
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
            if 'ProbabilityWeight' in metadata:
              weights = metadata['ProbabilityWeight'].values
            else:
              # TODO is this correct sizing generally?
              weights = np.ones(requestData.shape[0])
            metricData = (requestData, weights)
      elif isinstance(currentInput, Distributions.Distribution):
        if currentInput.name == metricDataName and dataName is None:
          if metricData is not None:
            self.raiseAnError(IOError, "Same feature or target variable " + metricDataName + "is found in multiple input objects")
          #Found the distribution, now put it in the return value
          metricData = currentInput

    if metricData is None:
      self.raiseAnError(IOError, "Feature or target variable " + origMetricDataName + " is not found")
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

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    super().initialize(runInfo, inputs, initDict)
    for metricIn in self.assemblerDict['Metric']:
      self.metricsDict[metricIn[2]] = metricIn[3]

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'Metric':
        if 'type' not in child.parameterValues.keys() or 'class' not in child.parameterValues.keys():
          self.raiseAnError(IOError, 'Tag Metric must have attributes "class" and "type"')
      elif child.getName() == 'Features':
        self.features = child.value
        self.featuresType = child.parameterValues['type']
      elif child.getName() == 'Targets':
        self.targets = child.value
        self.TargetsType = child.parameterValues['type']
      elif child.getName() == 'multiOutput':
        self.multiOutput = child.value
      elif child.getName() == 'weight':
        self.weight = np.asarray(child.value)
      elif child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.getName(), " is provided for metric system")

    if not self.features:
      self.raiseAnError(IOError, "XML node 'Features' is required but not provided")
    elif len(self.features) != len(self.targets):
      self.raiseAnError(IOError, 'The number of variables found in XML node "Features" is not equal the number of variables found in XML node "Targets"')

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object, (Files or DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    outputDict = evaluation[1]
    # FIXED: writing directly to file is no longer an option!
    #if isinstance(output, Files.File):
    #  availExtens = ['xml']
    #  outputExtension = output.getExt().lower()
    #  if outputExtension not in availExtens:
    #    self.raiseAMessage('Metric postprocessor did not recognize extension ".', str(outputExtension), '". The output will be dumped to a text file')
    #  output.setPath(self._workingDir)
    #  self.raiseADebug('Write Metric prostprocessor output in file with name: ', output.getAbsFile())
    #  self._writeXML(output, outputDict)
    if output.type in ['PointSet', 'HistorySet']:
      self.raiseADebug('Adding output in data object named', output.name)
      rlz = {}
      for key, val in outputDict.items():
        newKey = key.replace("|","_")
        rlz[newKey] = val
      if self.dynamic:
        rlz[self.pivotParameter] = np.atleast_1d(self.pivotValues)
      output.addRealization(rlz)
      # add metadata
      xml = self._writeXML(output, outputDict)
      output._meta['MetricPP'] = xml
    elif output.type == 'HDF5':
      self.raiseAnError(IOError, 'Output type', str(output.type), 'is not yet implemented. Skip it')
    else:
      self.raiseAnError(IOError, 'Output type ', str(output.type), ' can not be used for postprocessor', self.name)

  def _writeXML(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to the metadata within a data object
      @ In, output, DataObject, instance to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, xml, xmlUtils.StaticXmlElement instance, written data in XML format
    """
    if self.dynamic:
      outputInstance = xmlUtils.DynamicXmlElement('MetricPostProcessor', pivotParam=self.pivotParameter)
    else:
      outputInstance = xmlUtils.StaticXmlElement('MetricPostProcessor')
    if self.dynamic:
      for key, values in outputDictionary.items():
        assert("|" in key)
        metricName, nodeName = key.split('|')
        for ts, pivotVal in enumerate(self.pivotValues):
          if values.shape[0] == 1:
            outputInstance.addScalar(nodeName, metricName,values[0], pivotVal=pivotVal)
          else:
            outputInstance.addScalar(nodeName, metricName,values[ts], pivotVal=pivotVal)
    else:
      for key, values in outputDictionary.items():
        assert("|" in key)
        metricName, nodeName = key.split('|')
        if len(list(values)) == 1:
          outputInstance.addScalar(nodeName, metricName, values[0])
        else:
          self.raiseAnError(IOError, "Multiple values are returned from metric '", metricName, "', this is currently not allowed")
    return outputInstance

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    measureList = self.inputToInternal(inputIn)
    outputDict = {}
    assert(len(self.features) == len(measureList))
    for metricInstance in self.metricsDict.values():
      metricEngine = MetricDistributor.factory.returnInstance('MetricDistributor', metricInstance)
      for cnt in range(len(self.targets)):
        nodeName = (str(self.targets[cnt]) + '_' + str(self.features[cnt])).replace("|","_")
        varName = metricInstance.name + '|' + nodeName
        output = metricEngine.evaluate(measureList[cnt], weights=self.weight, multiOutput=self.multiOutput)
        outputDict[varName] = np.atleast_1d(output)
    return outputDict
