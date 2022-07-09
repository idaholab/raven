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
from ....utils import utils
from ....utils import InputData, InputTypes
from ....utils import xmlUtils
from .... import Files
from .... import DataObjects
from ..ValidationBase import ValidationBase
#Internal Modules End--------------------------------------------------------------------------------

class PPDSS(ValidationBase):
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
    # Have added the new pivotParameters for feature and target. The original has been commented out.
    pivotParameterFeatureInput = InputData.parameterInputFactory("pivotParameterFeature", contentType=InputTypes.StringType,
                                                                descr="""Pivot parameter for feature inputs""")
    inputSpecification.addSub(pivotParameterFeatureInput)
    pivotParameterTargetInput = InputData.parameterInputFactory("pivotParameterTarget", contentType=InputTypes.StringType,
                                                                descr="""Pivot parameter for target inputs""")
    inputSpecification.addSub(pivotParameterTargetInput)
    separateFeatureDataInput = InputData.parameterInputFactory("separateFeatureData", contentType=InputTypes.StringType,
                                                                descr="""Time points to separate feature data and conduct DSS postprocessing
                                                                on selected time interval. Values must be fractions of the full time interval.
                                                                To distinguish strart and end, '|' should be placed between both values. Example: 0.0|0.4
                                                                If 'None' is provided, the DSS postprocessing will be applied to the full interval.""")
    separateFeatureDataInput.addParam("type", InputTypes.StringType, descr=r"""Specifies what type of user defined time intervals have been selected from
                                                                    the feature pivot parameter. Options are currently `ratio' or `raw\_values'""")
    inputSpecification.addSub(separateFeatureDataInput)
    separateTargetDataInput = InputData.parameterInputFactory("separateTargetData", contentType=InputTypes.StringType,
                                                                descr="""Time points to separate target data and conduct DSS postprocessing
                                                                on selected time interval. Values must be fractions of the full time interval.
                                                                To distinguish strart and end, '|' should be placed between both values. Example: 0.0|0.4
                                                                If 'None' is provided, the DSS postprocessing will be applied to the full interval.""")
    separateTargetDataInput.addParam("type", InputTypes.StringType, descr=r"""Specifies what type of user defined time intervals have been selected from
                                                                    the target pivot parameter. Options are currently `ratio' or `raw\_values'""")
    inputSpecification.addSub(separateTargetDataInput)
    scaleTypeInput = InputData.parameterInputFactory("scale", contentType=InputTypes.makeEnumType("scale","scaleType",['DataSynthesis','2_2_affine','dilation','beta_strain','omega_strain','identity']),
                                                      descr="""Scaling type for the time transformation. Available types are DataSynthesis,
                                                      2_2_affine, dilation, beta_strain, omega_strain, and identity""")
    inputSpecification.addSub(scaleTypeInput)
    scaleRatioBetaInput = InputData.parameterInputFactory("scaleBeta", contentType=InputTypes.FloatType,
                                                          descr="""Scaling ratio for the parameter of interest""")
    inputSpecification.addSub(scaleRatioBetaInput)
    scaleRatioOmegaInput = InputData.parameterInputFactory("scaleOmega", contentType=InputTypes.FloatType,
                                                            descr="""Scaling ratio for the agents of change""")
    inputSpecification.addSub(scaleRatioOmegaInput)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR DSS Scaling and Metrics' # Naming
    self.name = 'PPDSS' # Postprocessor name
    self.dynamic               = True  # Must be time-dependent?
    self.dynamicType = ['dynamic'] # Specification of dynamic type
    self.features              = None  # list of feature variables
    self.targets               = None  # list of target variables
    self.multiOutput           = 'raw_values' # defines aggregating of multiple outputs for HistorySet
                                # currently allow raw_values
    self.pivotParameterFeature = None # Feature pivot parameter variable
    self.pivotValuesFeature    = [] # Feature pivot parameter values
    self.pivotParameterTarget  = None # Target pivot parameter variable
    self.pivotValuesTarget     = [] # Target pivot parameter values
    self.separateFeatureData   = None # String of data separation points. Default is None.
    self.separateFeatureType   = 'ratio' # defines the type of feature time separation to apply. Options are ratio and raw_values
    self.separateTargetData    = None # String of data separation points. Default is None.
    self.separateTargetType    = 'ratio' # defines the type of target time separation to apply. Options are ratio and raw_values
    self.scaleType             = None # Scaling type
    # assembler objects to be requested
    self.scaleRatioBeta        = [] # Scaling ratio for the parameter of interest
    self.scaleRatioOmega       = [] # Scaling ratio for the agents of change

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
      elif child.getName() == 'Targets':
        self.targets = child.value
      elif child.getName() == 'multiOutput':
        self.multiOutput = child.value
      elif child.getName() == 'pivotParameterFeature':
        self.pivotParameterFeature = child.value
      elif child.getName() == 'pivotParameterTarget':
        self.pivotParameterTarget = child.value
      elif child.getName() == 'separateFeatureData':
        if 'type' in child.parameterValues.keys():
          self.separateFeatureType = child.parameterValues["type"]
          self.separateFeatureData = child.value
      elif child.getName() == 'separateTargetData':
        if 'type' in child.parameterValues.keys():
          self.separateTargetType = child.parameterValues["type"]
          self.separateTargetData = child.value
      elif child.getName() == 'scale':
        self.scaleType = child.value
      elif child.getName() == 'scaleBeta':
        self.scaleRatioBeta = child.value
      elif child.getName() == 'scaleOmega':
        self.scaleRatioOmega = child.value
      else:
        self.raiseAnError(IOError, "Unknown xml node ", child.getName(), " is provided for metric system")

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    # assert
    assert(isinstance(inputIn["Data"], list))
    assert(isinstance(inputIn["Data"][0][-1], xr.Dataset) and isinstance(inputIn["Data"][1][-1], xr.Dataset))
    # the input can be either be a list of dataobjects or a list of datasets (xarray)
    datasets = [data for _, _, data in inputIn['Data']]
    names = []
    pivotParameterTarget = self.pivotParameterTarget
    pivotParameterFeature = self.pivotParameterFeature
    names = [self.getDataSetName(inp[-1]) for inp in inputIn['Data']]
    if len(inputIn['Data'][0][-1].indexes) and (self.pivotParameterTarget is None or self.pivotParameterFeature is None):
      if 'dynamic' not in self.dynamicType: #self.model.dataType:
        self.raiseAnError(IOError, "The validation algorithm '{}' is not a dynamic model but time-dependent data has been inputted in object {}".format(self._type, inputIn['Data'][0][-1].name))
      else:
        pivotParameterTarget = inputIn['Data'][1][2]
        pivotParameterFeature = inputIn['Data'][0][2]
    #  check if pivotParameter
    evaluation = self._evaluate(datasets)
    if not isinstance(evaluation, list):
      self.raiseAnError(IOError,"The data type in evaluation is not list")
    if pivotParameterFeature and pivotParameterTarget:
      if self.separateFeatureData == None and self.separateTargetData:
        if len(datasets[0][pivotParameterFeature]) != len(list(evaluation[0].values())[0]) and len(datasets[1][pivotParameterTarget]) != len(list(evaluation[0].values())[0]):
          self.raiseAnError(RuntimeError, "The pivotParameterFeature values has size '{}' and pivotParameterTarget values has size '{}'. The validation output has size '{}' and must match either pivot parameters.".format(len(datasets[0][self.pivotParameterFeature]), len(datasets[0][self.pivotParameterTarget]), len(evaluation[0].values()[0])))
    return evaluation

  def _evaluate(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    realizations = []
    realizationArray = []
    if len(self.features) > 1 or len(self.targets) > 1:
      self.raiseAnError(IOError, "The number of inputs for features or targets is greater than 1. Please restrict to one set per step.")
    feat = self.features[0]
    targ = self.targets[0]
    scaleRatioBeta = self.scaleRatioBeta
    scaleRatioOmega = self.scaleRatioOmega
    nameFeat = feat.split("|")
    nameTarg = targ.split("|")
    names = [nameFeat[0],nameTarg[0]]
    featDataSet = self._getDataFromDatasets(datasets, feat, names)[0]
    targDataSet = self._getDataFromDatasets(datasets, targ, names)[0]
    if (isinstance(scaleRatioBeta,int) or isinstance(scaleRatioBeta,float)) and (isinstance(scaleRatioOmega,int) or isinstance(scaleRatioOmega,float)) is True:
      if self.scaleType == 'DataSynthesis':
        timeScalingRatio = 1
        if abs(1-scaleRatioBeta) > 10**(-4) or abs(1-scaleRatioOmega) > 10**(-4):
          self.raiseAnError(IOError, "Either beta or omega scaling ratio are not 1. Both must be 1")
      elif self.scaleType == '2_2_affine':
        timeScalingRatio = scaleRatioBeta/scaleRatioOmega
      elif self.scaleType == 'dilation':
        timeScalingRatio = 1
        if abs(1-scaleRatioBeta/scaleRatioOmega) > 10**(-4):
          self.raiseAnError(IOError, "Beta scaling ratio:",scaleRatioBeta,"and Omega scaling ratio:",scaleRatioOmega,"are not nearly equivalent")
      elif self.scaleType == 'beta_strain':
        timeScalingRatio = scaleRatioBeta
        if abs(1-scaleRatioOmega) > 10**(-4):
          self.raiseAnError(IOError, "Omega scaling ratio:",scaleRatioOmega,"must be 1")
      elif self.scaleType == 'omega_strain':
        timeScalingRatio = 1/scaleRatioOmega
        if abs(1-scaleRatioBeta) > 10**(-4):
          self.raiseAnError(IOError, "Beta scaling ratio:",scaleRatioBeta,"must be 1")
      elif self.scaleType == 'identity':
        timeScalingRatio = 1
        if abs(1-scaleRatioBeta) > 10**(-4) or abs(1-scaleRatioOmega) > 10**(-4):
          self.raiseAnError(IOError, "Either beta or omega scaling ratio are not 1. Both must be 1")
      else:
        self.raiseAnError(IOError, "Scaling Type",self.scaleType, "is not provided")
    else:
      self.raiseAnError(IOError, scaleRatioBeta,"or",scaleRatioOmega,"is not a numerical number")

    pivotFeatureData = self._getDataFromDatasets(datasets, names[0]+"|"+self.pivotParameterFeature, names)[0]
    pivotFeatureData = np.transpose(pivotFeatureData)[0]
    pivotTargetData = self._getDataFromDatasets(datasets, names[1]+"|"+self.pivotParameterTarget, names)[0]
    pivotTargetData = np.transpose(pivotTargetData)[0]
    if self.separateFeatureData != None:
      separateTime = self.separateFeatureData
      separateTime = separateTime.split("|")
      startTime = separateTime[0]
      endTime = separateTime[1]
      if startTime >= endTime:
        self.raiseAnError(IOError, "Feature start time", startTime, "is equal or larger than end time", endTime)
      if self.separateFeatureType == "ratio":
        featureStartTime = pivotFeatureData[0]+float(startTime)*(pivotFeatureData[len(pivotFeatureData)-1]-pivotFeatureData[0])
        featureEndTime = pivotFeatureData[0]+float(endTime)*(pivotFeatureData[len(pivotFeatureData)-1]-pivotFeatureData[0])
      elif self.separateFeatureType == "raw_values":
        featureStartTime = float(startTime)
        featureEndTime = float(endTime)
      else:
        self.raiseAnError(IOError, 'separateFeatureData attribute "type" must be either "ratio" or "raw_values"')
      featureStartCount = 0
      startFeatureLocation = None
      endFeatureLocation = None
      for i in range(len(pivotFeatureData)):
        if pivotFeatureData[i] >= featureStartTime and featureStartCount == 0:
          startFeatureLocation = i
          featureStartCount += 1
        if pivotFeatureData[i] > featureEndTime:
          endFeatureLocation = i-1
          break
        elif pivotFeatureData[i] == featureEndTime:
          endFeatureLocation = i
          break
      pivotFeature = pivotFeatureData[startFeatureLocation:endFeatureLocation]
      featData = np.zeros((np.shape(featDataSet)[0],endFeatureLocation-startFeatureLocation))
      for i in range(len(featDataSet)):
        featData[i] = featDataSet[i][startFeatureLocation:endFeatureLocation]
    if self.separateTargetData != None:
      separateTime = self.separateTargetData
      separateTime = separateTime.split("|")
      startTime = separateTime[0]
      endTime = separateTime[1]
      if startTime >= endTime:
        self.raiseAnError(IOError, "Target start time", startTime, "is equal or larger than end time", endTime)
      if self.separateTargetType == "ratio":
        targetStartTime = pivotTargetData[0]+float(startTime)*(pivotTargetData[len(pivotTargetData)-1]-pivotTargetData[0])
        targetEndTime = pivotTargetData[0]+float(endTime)*(pivotTargetData[len(pivotTargetData)-1]-pivotTargetData[0])
      elif self.separateTargetType == "raw_values":
        targetStartTime = float(startTime)
        targetEndTime = float(endTime)
      else:
        self.raiseAnError(IOError, 'separateTargetData attribute "type" must be either "ratio" or "raw_values"')
      targetStartCount = 0
      startTargetLocation = None
      endTargetLocation = None
      for j in range(len(pivotTargetData)):
        if pivotTargetData[j] >= targetStartTime and targetStartCount == 0:
          startTargetLocation = j
          targetStartCount += 1
        if pivotTargetData[j] > targetEndTime:
          endTargetLocation = j-1
          break
        elif pivotTargetData[j] == targetEndTime:
          endTargetLocation = j
      pivotTarget = pivotTargetData[startTargetLocation:endTargetLocation]
      targData = np.zeros((np.shape(targDataSet)[0],endTargetLocation-startTargetLocation))
      for i in range(len(targDataSet)):
        targData[i] = targDataSet[i][startTargetLocation:endTargetLocation]
    else:
      pivotFeature = pivotFeatureData
      pivotTarget = pivotTargetData
      featData = featDataSet
      targData = targDataSet

    pivotFeatureSize = pivotFeature.shape[0]
    pivotTargetSize = pivotTarget.shape[0]
    if pivotFeatureSize >= pivotTargetSize:
      pivotSize = pivotTargetSize
    else:
      pivotSize = pivotFeatureSize

    if pivotFeatureSize == pivotSize:
      yCount = featData.shape[0]
      zCount = featData.shape[1]
    else:
      yCount = targData.shape[0]
      zCount = targData.shape[1]
    featureD = np.zeros((yCount,zCount))
    featureProcessTimeNorm = np.zeros((yCount,zCount))
    featureOmegaNorm = np.zeros((yCount,zCount))
    featureBeta = np.zeros((yCount,zCount))
    naNCount = np.zeros((yCount,zCount))
    #
    feature = nameFeat[1]
    for cnt2 in range(yCount):
      if pivotFeatureSize == pivotSize:
        featureBeta[cnt2] = featData[cnt2]
        interpGrid = pivotFeature
      else:
        interpFunction = interp1d(pivotFeature,featData[cnt2],kind='linear',fill_value='extrapolate')
        interpGrid = timeScalingRatio*pivotTarget
        featureBeta[cnt2] = interpFunction(interpGrid)
      featureOmega = np.gradient(featureBeta[cnt2],interpGrid)
      featureProcessTime = featureBeta[cnt2]/featureOmega
      featureDiffOmega = np.gradient(featureOmega,interpGrid)
      featureD[cnt2] = -featureBeta[cnt2]/featureOmega**2*featureDiffOmega
      for cnt3 in range(zCount):
        if np.isnan(featureD[cnt2][cnt3]) == True:
          naNCount[cnt2][cnt3] = 1
        elif np.isinf(featureD[cnt2][cnt3]) == True:
          naNCount[cnt2][cnt3] = 1
      featureInt = featureD[cnt2]+1
      # Excluding NaN type data and exclude corresponding time in grid in
      # preperation for numpy simpson integration
      count=0
      for i in range(len(featureD[cnt2])):
        if np.isnan(featureD[cnt2][i])==False and np.isinf(featureD[cnt2][i])==False:
          count += 1
      if count > 0:
        featureIntNew = np.zeros(count)
        interpGridNew = np.zeros(count)
        trackCount = 0
        for i in range(len(featureD[cnt2])):
          if np.isnan(featureD[cnt2][i])==False and np.isinf(featureD[cnt2][i])==False:
            interpGridNew[trackCount] = interpGrid[i]
            featureIntNew[trackCount] = featureInt[i]
            trackCount += 1
          else:
            featureD[cnt2][i] = 0
      #
      featureProcessAction = simps(featureIntNew, interpGridNew)
      featureProcessTimeNorm[cnt2] = featureProcessTime/featureProcessAction
      featureOmegaNorm[cnt2] = featureProcessAction*featureOmega
    #
    targetD = np.zeros((yCount,zCount))
    targetProcessTimeNorm = np.zeros((yCount,zCount))
    targetOmegaNorm = np.zeros((yCount,zCount))
    targetBeta = np.zeros((yCount,zCount))
    target = nameTarg[1]
    for cnt2 in range(yCount):
      if pivotTargetSize == pivotSize:
        targetBeta[cnt2] = targData[cnt2]
        interpGrid = pivotTarget
      else:
        interpFunction = interp1d(pivotTarget,targData[cnt2],kind='linear',fill_value='extrapolate')
        interpGrid = 1/timeScalingRatio*pivotFeature
        targetBeta[cnt2] = interpFunction(interpGrid)
      targetOmega = np.gradient(targetBeta[cnt2],interpGrid)
      #print("targetOmega:",targetOmega)
      targetProcessTime = targetBeta[cnt2]/targetOmega
      targetDiffOmega = np.gradient(targetOmega,interpGrid)
      targetD[cnt2] = -targetBeta[cnt2]/targetOmega**2*targetDiffOmega
      for cnt3 in range(zCount):
        if np.isnan(targetD[cnt2][cnt3]) == True:
          naNCount[cnt2][cnt3] = 1
        elif np.isinf(targetD[cnt2][cnt3]) == True:
          naNCount[cnt2][cnt3] = 1
      targetInt = targetD[cnt2]+1
      # Excluding NaN type data and exclude corresponding time in grid in
      # preperation for numpy simpson integration
      count=0
      for i in range(len(targetD[cnt2])):
        if np.isnan(targetD[cnt2][i])==False and np.isinf(targetD[cnt2][i])==False:
          count += 1
      if count > 0:
        targetIntNew = np.zeros(count)
        interpGridNew = np.zeros(count)
        trackCount = 0
        for i in range(len(targetD[cnt2])):
          if np.isnan(targetD[cnt2][i])==False and np.isinf(targetD[cnt2][i])==False:
            interpGridNew[trackCount] = interpGrid[i]
            targetIntNew[trackCount] = targetInt[i]
            trackCount += 1
          else:
            targetD[cnt2][i] = 0
      #
      targetProcessAction = simps(targetIntNew, interpGridNew)
      targetProcessTimeNorm[cnt2] = targetProcessTime/targetProcessAction
      targetOmegaNorm[cnt2] = targetProcessAction*targetOmega
    #
    featureProcessTimeNormScaled = np.zeros((yCount,zCount))
    featureOmegaNormScaled = np.zeros((yCount,zCount))
    for cnt3 in range(yCount):
      featureProcessTimeNormScaled[cnt3] = featureProcessTimeNorm[cnt3]/timeScalingRatio
      featureOmegaNormScaled[cnt3] = featureOmegaNorm[cnt3]/scaleRatioBeta
    newfeatureData = np.asarray([featureOmegaNormScaled,featureProcessTimeNormScaled,featureBeta])
    newtargetData = np.asarray([targetOmegaNorm,targetD,targetBeta])
    #------------------------------------------------------------------------------------------
    if pivotTargetSize == pivotSize:
      timeParameter = pivotTarget
    else:
      timeParameter = pivotFeature
    outputDict = {}
    distanceTotal = np.zeros((yCount,zCount))
    sigma = np.zeros((yCount,zCount))
    for metric in self.metrics:
      name = "{}_{}_{}".format(metric.estimator.name, targ.split("|")[-1], feat.split("|")[-1])
    output = metric.evaluate((newfeatureData,newtargetData), multiOutput='raw_values')
    for cnt2 in range(yCount):
      distanceSum = abs(np.sum(output[cnt2]))
      sigmaSum = 0
      for cnt3 in range(zCount):
        distanceTotal[cnt2][cnt3] = distanceSum
        sigmaSum += output[cnt2][cnt3]**2
      for cnt3 in range(zCount):
        sigma[cnt2][cnt3] = (1/(zCount-np.sum(naNCount[cnt2]))*sigmaSum)**0.5
    rlz = []
    for cnt in range(yCount):
      outputDict = {}
      outputDict[name] = abs(np.atleast_1d(output[cnt]))
      outputDict['pivot_parameter'] = timeParameter
      outputDict['total_distance_'+nameTarg[1]+'_'+nameFeat[1]] = distanceTotal[cnt]
      outputDict['feature_beta_'+nameTarg[1]+'_'+nameFeat[1]] = featureBeta[cnt]
      outputDict['target_beta_'+nameTarg[1]+'_'+nameFeat[1]] = targetBeta[cnt]
      outputDict['feature_omega_'+nameTarg[1]+'_'+nameFeat[1]] = featureOmegaNormScaled[cnt]
      outputDict['target_omega_'+nameTarg[1]+'_'+nameFeat[1]] = targetOmegaNorm[cnt]
      outputDict['feature_D_'+nameTarg[1]+'_'+nameFeat[1]] = featureD[cnt]
      outputDict['target_D_'+nameTarg[1]+'_'+nameFeat[1]] = targetD[cnt]
      outputDict['process_time_'+nameTarg[1]+'_'+nameFeat[1]] = newfeatureData[1][cnt]
      outputDict['standard_error_'+nameTarg[1]+'_'+nameFeat[1]] = sigma[cnt]
      rlz.append(outputDict)
    realizationArray.append(rlz)
    #---------------
    for cnt in range(len(realizationArray[0])):
      out = {}
      for cnt2 in range(len(realizationArray)):
        for key, val in realizationArray[cnt2][cnt].items():
          out[key] = val
      realizations.append(out)
    return realizations

  def _getDataFromDatasets(self, datasets, var, names=None):
    """
      Utility function to retrieve the data from dataDict
      @ In, datasets, list, list of datasets (data1,data2,etc.) to search from.
      @ In, names, list, optional, list of datasets names (data1,data2,etc.). If not present, the search will be done on the full list.
      @ In, var, str, the variable to find (either in fromat dataobject|var or simply var)
      @ Out, data, tuple(numpy.ndarray, xarray.DataArray or None), the retrived data (data, probability weights (None if not present))
    """
    data = None
    pw = None
    dat = None
    if "|" in var and names is not None:
      do, feat =  var.split("|")
      doindex = names.index(do)
      dat = datasets[doindex][feat]
    else:
      for doindex, ds in enumerate(datasets):
        if var in ds:
          dat = ds[var]
          break
    if 'ProbabilityWeight-{}'.format(feat) in datasets[names.index(do)]:
      pw = datasets[doindex]['ProbabilityWeight-{}'.format(feat)].values
    elif 'ProbabilityWeight' in datasets[names.index(do)]:
      pw = datasets[doindex]['ProbabilityWeight'].values
    dim = len(dat.shape)
    # (numRealizations,  numHistorySteps) for MetricDistributor
    dat = dat.values
    if dim == 1:
      #  the following reshaping does not require a copy
      dat.shape = (dat.shape[0], 1)
    data = dat, pw
    return data

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, DataObject.DataObject, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    realizations = evaluation[1]
    for rlz in realizations:
      output.addRealization(rlz)
