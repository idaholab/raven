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
Created on January XX, 2021

@author: yoshrk
"""
from __future__ import division, print_function , unicode_literals, absolute_import

#External Modules------------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import os
from collections import OrderedDict
import itertools
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from utils import InputData, InputTypes
import Files
import Distributions
import MetricDistributor
from .PostProcessor import PostProcessor
#Internal Modules End--------------------------------------------------------------------------------

class PPDSS(PostProcessor):
  """
    DSS Scaling class.
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
    inputSpecification = super(PPDSS, cls).getInputSpecification()
    featuresInput = InputData.parameterInputFactory("Features", contentType=InputTypes.StringListType)
    featuresInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(featuresInput)
    targetsInput = InputData.parameterInputFactory("Targets", contentType=InputTypes.StringListType)
    targetsInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(targetsInput)
    multiOutputInput = InputData.parameterInputFactory("multiOutput", contentType=InputTypes.StringType)
    inputSpecification.addSub(multiOutputInput)
    multiOutput = InputTypes.makeEnumType('MultiOutput', 'MultiOutputType', 'raw_values')
    multiOutputInput = InputData.parameterInputFactory("multiOutput", contentType=multiOutput)
    inputSpecification.addSub(multiOutputInput)
    #pivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    #inputSpecification.addSub(pivotParameterInput)
    #
    # Have added the new pivotParameters for feature and target. The original has been commented out.
    pivotParameterFeatureInput = InputData.parameterInputFactory("pivotParameterFeature", contentType=InputTypes.StringType)
    inputSpecification.addSub(pivotParameterFeatureInput)
    pivotParameterTargetInput = InputData.parameterInputFactory("pivotParameterTarget", contentType=InputTypes.StringType)
    inputSpecification.addSub(pivotParameterTargetInput)
    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    metricInput.addParam("class", InputTypes.StringType, True)
    metricInput.addParam("type", InputTypes.StringType, True)
    inputSpecification.addSub(metricInput)
    scaleTypeInput = InputData.parameterInputFactory("scale", contentType=InputTypes.StringType)
    inputSpecification.addSub(scaleTypeInput)
    scaleRatioBetaInput = InputData.parameterInputFactory("scaleBeta", contentType=InputTypes.FloatListType)
    inputSpecification.addSub(scaleRatioBetaInput)
    scaleRatioOmegaInput = InputData.parameterInputFactory("scaleOmega", contentType=InputTypes.FloatListType)
    inputSpecification.addSub(scaleRatioOmegaInput)
    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR DSS Scaling and Metrics'
    self.dynamic               = True  # is it time-dependent?
    self.features              = None  # list of feature variables
    self.targets               = None  # list of target variables
    self.metricsDict           = {}    # dictionary of metrics that are going to be assembled
    self.multiOutput           = 'raw_values' # defines aggregating of multiple outputs for HistorySet
                                # currently allow raw_values
    #self.pivotParameter        = None  # list of pivot parameters
    #self.pivotValues           = []
    self.pivotParameterFeature = None
    self.pivotValuesFeature    = []
    self.pivotParameterTarget  = None
    self.pivotValuesTarget     = []
    self.scaleType             = None
    #self.scaleType             = ['DataSynthesis','2_2_Affine','Dilation','beta_strain','omega_strain','identity']
    # assembler objects to be requested
    self.scaleRatioBeta        = []
    self.scaleRatioOmega       = []
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
      elif isinstance(currentInput, Distributions.Distribution):
        if currentInput.name == metricDataName and dataName is None:
          if metricData is not None:
            self.raiseAnError(IOError, "Same feature or target variable " + metricDataName + "is found in multiple input objects")
          #Found the distribution, now put it in the return value
          metricData = currentInput

    if metricData is None:
      self.raiseAnError(IOError, "Feature or target variable " + origMetricDataName + " is not found")
    # TODO: Ramon Added Prints to Inspect where This Function is used
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
          if self.pivotParameterFeature not in currentInput.getVars('indexes'):
            self.raiseAnError(IOError, self, 'Feature Pivot parameter', self.pivotParameterFeature,'has not been found in DataObject', currentInput.name)
          if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameterFeature):
            self.raiseAnError(IOError, "HistorySet", currentInput.name," is not syncronized, please use Interfaced PostProcessor HistorySetSync to pre-process it")
          pivotValuesFeature  = currentInput.asDataset()[self.pivotParameterFeature].values
          if len(self.pivotValuesFeature) == 0:
            self.pivotValuesFeature = pivotValuesFeature
          elif set(self.pivotValuesFeature) != set(pivotValuesFeature):
            self.raiseAnError(IOError, "Pivot values for feature pivot parameter",self.pivotParameterFeature, "in provided HistorySets are not the same")
          if self.pivotParameterTarget not in currentInput.getVars('indexes'):
            self.raiseAnError(IOError, self, 'Target Pivot parameter', self.pivotParameterTarget,'has not been found in DataObject', currentInput.name)
          if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameterTarget):
            self.raiseAnError(IOError, "HistorySet", currentInput.name," is not syncronized, please use Interfaced PostProcessor HistorySetSync to pre-process it")
          pivotValuesTarget  = currentInput.asDataset()[self.pivotParameterTarget].values
          if len(self.pivotValuesTarget) == 0:
            self.pivotValuesTarget = pivotValuesTarget
          elif set(self.pivotValuesTarget) != set(pivotValuesTarget):
            self.raiseAnError(IOError, "Pivot values for target pivot parameter",self.pivotParameterTarget, "in provided HistorySets are not the same")
      else:
        self.raiseAnError(IOError, "Metric cannot process "+inputType+ " of type "+str(type(currentInput)))

    # TODO: Current form does not support multiple variables in features and targets
    # TODO: The order of the feature, target, and scaling ratios have to be in order
    if len(self.features) == len(self.targets) and len(self.targets) == len(self.scaleRatioBeta) and len(self.scaleRatioBeta) == len(self.scaleRatioOmega):
      pass
    else:
      self.raiseAnError(IOError, "The list size of features, targets, scaleRatioBeta, and scaleRatioOmega must be the same")

    if self.pivotParameterFeature in currentInput.getVars('indexes'):
      x_count = len(self.features)
      y_count = len(self.__getMetricSide(self.features[0], currentInputs)[0])
      z_count = len(self.__getMetricSide(self.features[0], currentInputs)[0][0])
      featureProcessTimeNorm = np.zeros((x_count,y_count,z_count))
      featureOmegaNorm = np.zeros((x_count,y_count,z_count))
      featureBeta = np.zeros((x_count,y_count,z_count))
      pivotFeature = currentInput.getVarValues(currentInput.getVars('indexes')).get(self.pivotParameterFeature).values # in ndarray
      for cnt in range(len(self.features)):
        feature = self.features[cnt]
        featureData =  self.__getMetricSide(feature, currentInputs)[0] # in ndarray
        for cnt2 in range(len(featureData)):
          featureBeta[cnt][cnt2] = featureData[cnt2]
          featureOmega = np.gradient(featureBeta,pivotFeature)
          featureProcessTime = featureBeta/featureOmega
          featureDiffOmega = np.gradient(featureOmega,pivotFeature)
          featureD = -featureBeta/featureOmega**2*featureDiffOmega
          featureInt = featureD+1
          featureProcessAction = simps(featureInt, pivotFeature)
          featureProcessTimeNorm[cnt][cnt2] = featureProcessTime/featureProcessAction
          featureOmegaNorm[cnt][cnt2] = featureProcessAction*featureOmega

    if self.pivotParameterTarget in currentInput.getVars('indexes'):
      x_count = len(self.targets)
      y_count = len(self.__getMetricSide(self.targets[0], currentInputs)[0])
      z_count = len(self.__getMetricSide(self.targets[0], currentInputs)[0][0])
      targetD = np.zeros((x_count,y_count,z_count))
      targetProcessTimeNorm = np.zeros((x_count,y_count,z_count))
      targetOmegaNorm = np.zeros((x_count,y_count,z_count))
      targetBeta = np.zeros((x_count,y_count,z_count))
      pivotTarget = currentInput.getVarValues(currentInput.getVars('indexes')).get(self.pivotParameterTarget).values # in ndarray
      for cnt in range(len(self.targets)):
        target = self.targets[cnt]
        targetData =  self.__getMetricSide(target, currentInputs)[0] # in ndarray
        for cnt2 in range(len(targetData)):
          targetBeta[cnt][cnt2] = targetData[cnt2]
          targetOmega = np.gradient(targetBeta,pivotTarget)
          targetProcessTime = targetBeta/targetOmega
          targetDiffOmega = np.gradient(targetOmega,pivotTarget)
          targetD[cnt][cnt2] = -targetBeta/targetOmega**2*targetDiffOmega
          targetInt = targetD+1
          targetProcessAction = simps(targetInt, pivotTarget)
          targetProcessTimeNorm[cnt][cnt2] = targetProcessTime/targetProcessAction
          targetOmegaNorm[cnt][cnt2] = targetProcessAction*targetOmega

    measureList = []
    for cnt in range(len(self.features)):
      if (isinstance(self.scaleRatioBeta[cnt],int) or isinstance(self.scaleRatioBeta[cnt],float)) and (isinstance(self.scaleRatioOmega[cnt],int) or isinstance(self.scaleRatioOmega[cnt],float)) is True:
        if self.scaleType == 'DataSynthesis':
          timeScalingRatio = self.scaleRatioBeta[cnt]/self.scaleRatioOmega[cnt]
        elif self.scaleType == '2_2_Affine':
          timeScalingRatio = 1
          if abs(1-self.scaleRatioBeta[cnt]/self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioBeta[cnt],"and",self.scaleRatioOmega[cnt],"are not nearly equivalent")
        elif self.scaleType == 'Dilation':
          timeScalingRatio = self.scaleRatioBeta[cnt]
          if abs(1-self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioOmega[cnt],"must be 1")
        elif self.scaleType == 'omega_strain':
          timeScalingRatio = 1/self.scaleRatioOmega[cnt]
          if abs(1-self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioBeta[cnt],"must be 1")
        elif self.scaleType == 'identity':
          timeScalingRatio = 1
          if abs(1-self.scaleRatioBeta[cnt]) and abs(1-self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioBeta[cnt],"must be 1")
        else:
          self.raiseAnError(IOError, "Scaling Type",self.scaleType, "is not provided")
      else:
        self.raiseAnError(IOError, self.scaleRatioBeta[cnt],"or",self.scaleRatioOmega[cnt],"is not a numerical number")

      if len(self.__getMetricSide(self.features[0], currentInputs)[0][0]) >= len(self.__getMetricSide(self.targets[0], currentInputs)[0][0]):
        y_count = len(self.__getMetricSide(self.targets[0], currentInputs)[0])
        z_count = len(self.__getMetricSide(self.targets[0], currentInputs)[0][0])
        newfeatureOmegaNorm = np.zeros((y_count,z_count))
      else:
        y_count = len(self.__getMetricSide(self.features[0], currentInputs)[0])
        z_count = len(self.__getMetricSide(self.features[0], currentInputs)[0][0])
        newtargetOmegaNorm = np.zeros((y_count,z_count))
        newtargetBeta = np.zeros((y_count,z_count))

      featureProcessTimeNormScaled = np.zeros((len(featureProcessTimeNorm[cnt]),len(featureProcessTimeNorm[cnt][0])))
      featureOmegaNormScaled = np.zeros((len(featureOmegaNorm[cnt]),len(featureOmegaNorm[cnt][0])))
      newfeatureOmegaNorm = np.zeros((len(targetProcessTimeNorm[cnt]),len(targetProcessTimeNorm[cnt][0])))
      newtargetOmegaNorm = np.zeros((len(featureProcessTimeNormScaled[cnt]),len(featureProcessTimeNormScaled[cnt][0])))
      for cnt3 in range(len(featureProcessTimeNorm[cnt])):
        featureProcessTimeNormScaled[cnt3] = featureProcessTimeNorm[cnt][cnt3]/timeScalingRatio
        featureOmegaNormScaled[cnt3] = featureOmegaNorm[cnt][cnt3]/self.scaleRatioBeta[cnt]
        if len(featureProcessTimeNormScaled[cnt3]) >= len(targetProcessTimeNorm[cnt][cnt3]):
          interp1dGrid = targetProcessTimeNorm[cnt][cnt3]
          newfeatureFunction = interp1d(featureProcessTimeNormScaled[cnt3],featureOmegaNormScaled[cnt3],kind='linear',fill_value='extraplotate')
          newfeatureOmegaNorm[cnt3] = newfeatureFunction(interp1dGrid)
        else:
          interp1dGrid = featureProcessTimeNormScaled[cnt3]
          newtargetFunction = interp1d(targetProcessTimeNorm[cnt][cnt3],targetOmegaNorm[cnt][cnt3],kind='linear',fill_value='extraplotate')
          newtargetOmegaNorm[cnt3] = newtargetFunction(interp1dGrid)
          newtargetBeta[cnt3] = newtargetOmegaNorm[cnt3]*interp1dGrid
      # TODO: Need to add "D" and "Process Time Normalized" to measureList
      if len(self.__getMetricSide(self.features[0], currentInputs)[0][0]) >= len(self.__getMetricSide(self.targets[0], currentInputs)[0][0]):
        newfeatureData = np.asarray([newfeatureOmegaNorm,targetProcessTimeNorm[cnt],featureBeta[cnt]])
        newtargetData = np.asarray([targetOmegaNorm[cnt],targetD[cnt],targetBeta[cnt]])
        measureList.append((newfeatureData, newtargetData))
      else:
        newfeatureData = np.asarray([featureOmegaNorm[cnt],featureProcessTimeNormScaled,featureBeta[cnt]])
        newtargetData = np.asarray([newtargetOmegaNorm,targetD[cnt],newtargetBeta])
        measureList.append((newfeatureData, newtargetData))

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
        self.features = child.value
        self.featuresType = child.parameterValues['type']
      elif child.getName() == 'Targets':
        self.targets = child.value
        self.TargetsType = child.parameterValues['type']
      elif child.getName() == 'multiOutput':
        self.multiOutput = child.value
      elif child.getName() == 'pivotParameterFeature':
        self.pivotParameterFeature = child.value
      elif child.getName() == 'pivotParameterTarget':
        self.pivotParameterTarget = child.value
      elif child.getName() == 'scale':
        self.scaleType = child.value
      elif child.getName() == 'scaleBeta':
        self.scaleRatioBeta = child.value
      elif child.getName() == 'scaleOmega':
        self.scaleRatioOmega = child.value
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.getName(), " is provided for metric system")
    if not self.features:
      self.raiseAnError(IOError, "XML node 'Features' is required but not provided")
    elif len(self.features) != len(self.targets):
      self.raiseAnError(IOError, 'The number of variables found in XML node "Features" is not equal the number of variables found in XML node "Targets"')

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

  def collectOutput(self,finishedJob, output):
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
      metricEngine = MetricDistributor.returnInstance('MetricDistributor',metricInstance,self)
      for cnt in range(len(self.targets)):
        nodeName = (str(self.targets[cnt]) + '_' + str(self.features[cnt])).replace("|","_")
        varName = metricInstance.name + '|' + nodeName
        output = metricEngine.evaluate(measureList[cnt], weights=None, multiOutput=self.multiOutput)
        outputDict[varName] = np.atleast_1d(output)
    return outputDict
