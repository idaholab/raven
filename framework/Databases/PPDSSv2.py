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
import xarray as xr
import os
from collections import OrderedDict
import itertools
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from utils import InputData, InputTypes
from utils import xmlUtils
import Files
import Distributions
import MetricDistributor
from .PostProcessor import PostProcessor
#Internal Modules End--------------------------------------------------------------------------------

class PPDSSv2(PostProcessor):
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
    inputSpecification = super(PPDSSv2, cls).getInputSpecification()
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
    #self.processTimeRecord     = []
    #self.scaleType             = ['DataSynthesis','2_2_Affine','Dilation','beta_strain','omega_strain','identity']
    # assembler objects to be requested
    self.scaleRatioBeta        = []
    self.scaleRatioOmega       = []
    self.addAssemblerObject('Metric', InputData.Quantity.one_to_infinity)

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
            pass
          elif self.pivotParameterTarget not in currentInput.getVars('indexes'):
            pass
          else:
            self.raiseAnError(IOError, self, 'Feature Pivot parameter or Target Pivot parameter', self.pivotParameterFeature,'has not been found in DataObject', currentInput.name)
          #if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameterFeature):
          #  pass
          #elif not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameterTarget):
          #  pass
          #else:
          #  self.raiseAnError(IOError, "HistorySet", currentInput.name," is not syncronized, please use Interfaced PostProcessor HistorySetSync to pre-process it")
          if count == 1:
            pivotValuesFeature  = currentInput.asDataset()[self.pivotParameterFeature].values
            if len(self.pivotValuesFeature) == 0:
              self.pivotValuesFeature = pivotValuesFeature
            elif set(self.pivotValuesFeature) != set(pivotValuesFeature):
              self.raiseAnError(IOError, "Pivot values for feature pivot parameter",self.pivotParameterFeature, "in provided HistorySets are not the same")
          elif count == 2:
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

    return currentInputs

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

  def collectOutput(self,finishedJob, output):
    """
      Function to place all of the computed data into the output object, (Files or DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    realizations = evaluation[1]
    for rlz in realizations:
      output.addRealization(rlz)

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    currentInputs = self.inputToInternal(inputIn)
    #-------------------------------------------------------------------------------------------
    timeScalingRatio = np.zeros(len(self.features))
    for cnt in range(len(self.features)):
      if (isinstance(self.scaleRatioBeta[cnt],int) or isinstance(self.scaleRatioBeta[cnt],float)) and (isinstance(self.scaleRatioOmega[cnt],int) or isinstance(self.scaleRatioOmega[cnt],float)) is True:
        if self.scaleType == 'DataSynthesis':
          timeScalingRatio[cnt] = self.scaleRatioBeta[cnt]/self.scaleRatioOmega[cnt]
        elif self.scaleType == '2_2_Affine':
          timeScalingRatio[cnt] = 1
          if abs(1-self.scaleRatioBeta[cnt]/self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioBeta[cnt],"and",self.scaleRatioOmega[cnt],"are not nearly equivalent")
        elif self.scaleType == 'Dilation':
          timeScalingRatio[cnt] = self.scaleRatioBeta[cnt]
          if abs(1-self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioOmega[cnt],"must be 1")
        elif self.scaleType == 'omega_strain':
          timeScalingRatio[cnt] = 1/self.scaleRatioOmega[cnt]
          if abs(1-self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioBeta[cnt],"must be 1")
        elif self.scaleType == 'identity':
          timeScalingRatio[cnt] = 1
          if abs(1-self.scaleRatioBeta[cnt]) and abs(1-self.scaleRatioOmega[cnt]) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",self.scaleRatioBeta[cnt],"must be 1")
        else:
          self.raiseAnError(IOError, "Scaling Type",self.scaleType, "is not provided")
      else:
        self.raiseAnError(IOError, self.scaleRatioBeta[cnt],"or",self.scaleRatioOmega[cnt],"is not a numerical number")

    pivotFeature = currentInputs[0].getVarValues(currentInputs[0].getVars('indexes')).get(self.pivotParameterFeature).values # in ndarray
    pivotTarget = currentInputs[1].getVarValues(currentInputs[1].getVars('indexes')).get(self.pivotParameterTarget).values # in ndarray
    pivotFeatureSize = pivotFeature.shape[0]
    pivotTargetSize = pivotTarget.shape[0]
    if pivotFeatureSize >= pivotTargetSize:
      pivotSize = pivotTargetSize
    else:
      pivotSize = pivotFeatureSize

    if self.pivotParameterFeature in currentInputs[0].getVars('indexes'):
      if pivotFeatureSize == pivotSize:
        x_count = len(self.features)
        y_count = currentInputs[0].getVarValues(self.features[0]).values.shape[0]
        z_count = currentInputs[0].getVarValues(self.features[0]).values.shape[1]
      else:
        x_count = len(self.targets)
        y_count = currentInputs[1].getVarValues(self.targets[0]).values.shape[0]
        z_count = currentInputs[1].getVarValues(self.targets[0]).values.shape[1]
      featureProcessTimeNorm = np.zeros((x_count,y_count,z_count))
      featureOmegaNorm = np.zeros((x_count,y_count,z_count))
      featureBeta = np.zeros((x_count,y_count,z_count))
      for cnt in range(len(self.features)):
        feature = self.features[cnt]
        featureData =  currentInputs[0].getVarValues(feature).values # in ndarray
        for cnt2 in range(y_count):
          if pivotFeatureSize == pivotSize:
            featureBeta[cnt][cnt2] = featureData[cnt2]
            interpGrid = pivotFeature
          else:
            interpFunction = interp1d(pivotFeature,featureData[cnt2],kind='linear',fill_value='extrapolate')
            interpGrid = timeScalingRatio[cnt]*pivotTarget
            featureBeta[cnt][cnt2] = interpFunction(interpGrid)
          featureOmega = np.gradient(featureBeta[cnt][cnt2],interpGrid)
          featureProcessTime = featureBeta[cnt][cnt2]/featureOmega
          featureDiffOmega = np.gradient(featureOmega,interpGrid)
          featureD = -featureBeta[cnt][cnt2]/featureOmega**2*featureDiffOmega
          featureInt = featureD+1
          featureProcessAction = simps(featureInt, interpGrid)
          featureProcessTimeNorm[cnt][cnt2] = featureProcessTime/featureProcessAction
          featureOmegaNorm[cnt][cnt2] = featureProcessAction*featureOmega

    if self.pivotParameterTarget in currentInputs[1].getVars('indexes'):
      if pivotTargetSize == pivotSize:
        x_count = len(self.targets)
        y_count = currentInputs[1].getVarValues(self.targets[0]).values.shape[0]
        z_count = currentInputs[1].getVarValues(self.targets[0]).values.shape[1]
      else:
        x_count = len(self.features)
        y_count = currentInputs[0].getVarValues(self.features[0]).values.shape[0]
        z_count = currentInputs[0].getVarValues(self.features[0]).values.shape[1]
      targetD = np.zeros((x_count,y_count,z_count))
      targetProcessTimeNorm = np.zeros((x_count,y_count,z_count))
      targetOmegaNorm = np.zeros((x_count,y_count,z_count))
      targetBeta = np.zeros((x_count,y_count,z_count))
      for cnt in range(len(self.targets)):
        target = self.targets[cnt]
        targetData =  currentInputs[1].getVarValues(target).values # in ndarray
        for cnt2 in range(len(targetData)):
          if pivotTargetSize == pivotSize:
            targetBeta[cnt][cnt2] = targetData[cnt2]
            interpGrid = pivotTarget
          else:
            interpFunction = interp1d(pivotTarget,targetData[cnt2],kind='linear',fill_value='extrapolate')
            interpGrid = 1/timeScalingRatio[cnt]*pivotFeature
            targetBeta[cnt][cnt2] = interpFunction(interpGrid)
          targetOmega = np.gradient(targetBeta[cnt][cnt2],interpGrid)
          targetProcessTime = targetBeta[cnt][cnt2]/targetOmega
          targetDiffOmega = np.gradient(targetOmega,interpGrid)
          targetD[cnt][cnt2] = -targetBeta[cnt][cnt2]/targetOmega**2*targetDiffOmega
          targetInt = targetD[cnt][cnt2]+1
          targetProcessAction = simps(targetInt, interpGrid)
          targetProcessTimeNorm[cnt][cnt2] = targetProcessTime/targetProcessAction
          targetOmegaNorm[cnt][cnt2] = targetProcessAction*targetOmega

    measureList = []

    for cnt in range(len(self.features)):
      featureProcessTimeNormScaled = np.zeros((len(featureProcessTimeNorm[cnt]),len(featureProcessTimeNorm[cnt][0])))
      featureOmegaNormScaled = np.zeros((len(featureOmegaNorm[cnt]),len(featureOmegaNorm[cnt][0])))
      for cnt3 in range(len(featureProcessTimeNorm[cnt])):
        featureProcessTimeNormScaled[cnt3] = featureProcessTimeNorm[cnt][cnt3]/timeScalingRatio[cnt]
        featureOmegaNormScaled[cnt3] = featureOmegaNorm[cnt][cnt3]/self.scaleRatioBeta[cnt]
      newfeatureData = np.asarray([featureOmegaNormScaled,featureProcessTimeNormScaled,featureBeta[cnt]])
      newtargetData = np.asarray([targetOmegaNorm[cnt],targetD[cnt],targetBeta[cnt]])
      measureList.append((newfeatureData, newtargetData))
    #-------------------------------------------------------------------------------------------
    realizations = []
    if pivotTargetSize == pivotSize:
      x_count = len(self.targets)
      y_count = currentInputs[1].getVarValues(self.targets[0]).values.shape[0]
      z_count = currentInputs[1].getVarValues(self.targets[0]).values.shape[1]
      timeParameter = pivotTarget
    else:
      x_count = len(self.features)
      y_count = currentInputs[0].getVarValues(self.features[0]).values.shape[0]
      z_count = currentInputs[0].getVarValues(self.features[0]).values.shape[1]
      timeParameter = pivotFeature
    nameList = []
    valueList = []
    distanceTotal = np.zeros((x_count,y_count,z_count))
    sigma = np.zeros((x_count,y_count,z_count))
    assert(len(self.features) == len(measureList))
    for metricInstance in self.metricsDict.values():
      metricEngine = MetricDistributor.returnInstance('MetricDistributor',metricInstance,self)
      for cnt in range(len(self.targets)):
        nodeName = (str(self.targets[cnt]) + '_' + str(self.features[cnt])).replace("|","_")
        varName = metricInstance.name + '_' + nodeName
        output = metricEngine.evaluate(measureList[cnt], weights=None, multiOutput=self.multiOutput)
        nameList.append(varName)
        for cnt2 in range(y_count):
          distanceSum = abs(np.sum(output[cnt2]))
          sigmaSum = 0
          for cnt3 in range(z_count):
            distanceTotal[cnt][cnt2][cnt3] = distanceSum
            sigmaSum += output[cnt2][cnt3]**2
          for cnt3 in range(z_count):
            sigma[cnt][cnt2][cnt3] = (1/z_count*sigmaSum)**0.5
        valueList.append(output)
    for cnt in range(y_count):
      outputDict = {}
      outputDict['pivot_parameter'] = timeParameter
      for cnt2 in range(x_count):
        feature = self.features[cnt2]
        target = self.targets[cnt2]
        outputDict[nameList[cnt2]] = np.atleast_1d(valueList[cnt2][cnt])
        outputDict[target+'_'+feature+'_total_distance'] = distanceTotal[cnt2][cnt]
        outputDict[target+'_'+feature+'_process_time'] = measureList[cnt2][0][1][cnt]
        outputDict[target+'_'+feature+'_standard_deviation'] = sigma[cnt2][cnt]
      realizations.append(outputDict)
    return realizations
