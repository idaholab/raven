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
import DataObjects
from ..Validation import Validation
#import Distributions
#import MetricDistributor
#from .ValidationBase import ValidationBase
#from Models.PostProcessors import PostProcessor
#Internal Modules End--------------------------------------------------------------------------------

class PPDSS(Validation):
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
    #featuresInput = InputData.parameterInputFactory("Features", contentType=InputTypes.StringListType)
    #featuresInput.addParam("type", InputTypes.StringType)
    #inputSpecification.addSub(featuresInput)
    #targetsInput = InputData.parameterInputFactory("Targets", contentType=InputTypes.StringListType)
    #targetsInput.addParam("type", InputTypes.StringType)
    #inputSpecification.addSub(targetsInput)
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
    #metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    #metricInput.addParam("class", InputTypes.StringType, True)
    #metricInput.addParam("type", InputTypes.StringType, True)
    #inputSpecification.addSub(metricInput)
    scaleTypeInput = InputData.parameterInputFactory("scale", contentType=InputTypes.StringType)
    inputSpecification.addSub(scaleTypeInput)
    scaleRatioBetaInput = InputData.parameterInputFactory("scaleBeta", contentType=InputTypes.FloatListType)
    inputSpecification.addSub(scaleRatioBetaInput)
    scaleRatioOmegaInput = InputData.parameterInputFactory("scaleOmega", contentType=InputTypes.FloatListType)
    inputSpecification.addSub(scaleRatioOmegaInput)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'POSTPROCESSOR DSS Scaling and Metrics'
    self.name = 'PPDSS'
    self.dynamic               = True  # is it time-dependent?
    self.dynamicType = ['dynamic']
    self.features              = None  # list of feature variables
    self.targets               = None  # list of target variables
    #self.metricsDict           = {}    # dictionary of metrics that are going to be assembled
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
    #self.addAssemblerObject('Metric', InputData.Quantity.one_to_infinity)

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
        #self.featuresType = child.parameterValues['type']
      elif child.getName() == 'Targets':
        self.targets = child.value
        #self.TargetsType = child.parameterValues['type']
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
    #if not self.features:
    #  self.raiseAnError(IOError, "XML node 'Features' is required but not provided")
    #elif len(self.features) != len(self.targets):
    #  self.raiseAnError(IOError, 'The number of variables found in XML node "Features" is not equal the number of variables found in XML node "Targets"')

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, list, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    # assert
    assert(isinstance(inputIn["Data"], list))
    assert(isinstance(inputIn["Data"][0][-1], xr.Dataset) and isinstance(inputI["Data"][1][-1], xr.Dataset))
    # the input can be either be a list of dataobjects or a list of datasets (xarray)
    #datasets = [inp if isinstance(inp, xr.Dataset) else inp.asDataset() for inp in inputIn]
    datasets = [data for _, _, data in inputIn['Data']]
    #print("datasets:",datasets)
    names = []
    pivotParameterTarget = self.pivotParameterTarget
    pivotParameterFeature = self.pivotParameterFeature
    names = [inp[-1].attrs['name'] for inp in inputIn['Data']]
    #print("names:",names)
    #print("inputIn:",inputIn)
    #print("inputIn['Data'][0][2].indexes:",inputIn['Data'][0][2].indexes)
    #print("inputIn['Data'][0][-1].indexes:",inputIn['Data'][0][-1].indexes)
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
      if len(datasets[0][pivotParameterFeature]) != len(list(evaluation[0].values())[0]) and len(datasets[1][pivotParameterTarget]) != len(list(evaluation[1].values())[0]):
        self.raiseAnError(RuntimeError, "The pivotParameterFeature value '{}' has size '{}' and validation output has size '{}' The pivotParameterTarget value '{}' has size '{}' and validation output has size '{}'.".format( len(datasets[0][self.pivotParameterFeature]), len(evaluation.values()[0])))
      if pivotParameterFeature not in evaluation and pivotParameterTarget not in evaluation:
        for i in range(len(evaluation)):
          if len(datasets[0][pivotParameterFeature]) < len(datasets[1][pivotParameterTarget]):
            evaluation[i]['pivot_parameter'] = datasets[0][pivotParameterFeature]
          else:
            evaluation[i]['pivot_parameter'] = datasets[1][pivotParameterTarget]
    return evaluation

  def _evaluate(self, datasets, **kwargs):
    """
      Main method to "do what you do".
      @ In, datasets, list, list of datasets (data1,data2,etc.) to used.
      @ In, kwargs, dict, keyword arguments
      @ Out, outputDict, dict, dictionary containing the results {"feat"_"target"_"metric_name":value}
    """
    #print("datasets:",datasets)
    realizations = []
    realization_array = []
    for feat, targ, scaleRatioBeta, scaleRatioOmega in zip(self.features, self.targets, self.scaleRatioBeta, self.scaleRatioOmega):
      nameFeat = feat.split("|")
      nameTarg = targ.split("|")
      names = [nameFeat[0],nameTarg[0]]
      featData = self._getDataFromDatasets(datasets, feat, names)[0]
      targData = self._getDataFromDatasets(datasets, targ, names)[0]
      if (isinstance(scaleRatioBeta,int) or isinstance(scaleRatioBeta,float)) and (isinstance(scaleRatioOmega,int) or isinstance(scaleRatioOmega,float)) is True:
        if self.scaleType == 'DataSynthesis':
          timeScalingRatio = scaleRatioBeta/scaleRatioOmega
        elif self.scaleType == '2_2_Affine':
          timeScalingRatio = 1
          if abs(1-scaleRatioBeta/scaleRatioOmega) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",scaleRatioBeta,"and",scaleRatioOmega,"are not nearly equivalent")
        elif self.scaleType == 'Dilation':
          timeScalingRatio = scaleRatioBeta
          if abs(1-scaleRatioOmega) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",scaleRatioOmega,"must be 1")
        elif self.scaleType == 'omega_strain':
          timeScalingRatio = 1/scaleRatioOmega
          if abs(1-scaleRatioOmega) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",scaleRatioBeta,"must be 1")
        elif self.scaleType == 'identity':
          timeScalingRatio = 1
          if abs(1-scaleRatioBeta) and abs(1-scaleRatioOmega) > 10**(-4):
            self.raiseAnError(IOError, "Scaling ratio",scaleRatioBeta,"must be 1")
        else:
          self.raiseAnError(IOError, "Scaling Type",self.scaleType, "is not provided")
      else:
        self.raiseAnError(IOError, scaleRatioBeta,"or",scaleRatioOmega,"is not a numerical number")
      
      pivotFeature = self._getDataFromDatasets(datasets, names[0]+"|"+self.pivotParameterFeature, names)[0]
      pivotFeature = np.transpose(pivotFeature)[0]
      pivotTarget = self._getDataFromDatasets(datasets, names[1]+"|"+self.pivotParameterTarget, names)[0]
      pivotTarget = np.transpose(pivotTarget)[0]
      pivotFeatureSize = pivotFeature.shape[0]
      pivotTargetSize = pivotTarget.shape[0]
      if pivotFeatureSize >= pivotTargetSize:
        pivotSize = pivotTargetSize
      else:
        pivotSize = pivotFeatureSize
      
      if pivotFeatureSize == pivotSize:
        y_count = featData.shape[0]
        z_count = featData.shape[1]
      else:
        y_count = targData.shape[0]
        z_count = targData.shape[1]
      featureProcessTimeNorm = np.zeros((y_count,z_count))
      featureOmegaNorm = np.zeros((y_count,z_count))
      featureBeta = np.zeros((y_count,z_count))
      #
      feature = nameFeat[1]
      for cnt2 in range(y_count):
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
        featureD = -featureBeta[cnt2]/featureOmega**2*featureDiffOmega
        featureInt = featureD+1
        featureProcessAction = simps(featureInt, interpGrid)
        featureProcessTimeNorm[cnt2] = featureProcessTime/featureProcessAction
        featureOmegaNorm[cnt2] = featureProcessAction*featureOmega
      #
      targetD = np.zeros((y_count,z_count))
      targetProcessTimeNorm = np.zeros((y_count,z_count))
      targetOmegaNorm = np.zeros((y_count,z_count))
      targetBeta = np.zeros((y_count,z_count))
      target = nameTarg[1]
      for cnt2 in range(y_count):
        if pivotTargetSize == pivotSize:
          targetBeta[cnt2] = targData[cnt2]
          interpGrid = pivotTarget
        else:
          interpFunction = interp1d(pivotTarget,targData[cnt2],kind='linear',fill_value='extrapolate')
          interpGrid = 1/timeScalingRatio*pivotFeature
          targetBeta[cnt2] = interpFunction(interpGrid)
        targetOmega = np.gradient(targetBeta[cnt2],interpGrid)
        targetProcessTime = targetBeta[cnt2]/targetOmega
        targetDiffOmega = np.gradient(targetOmega,interpGrid)
        targetD[cnt2] = -targetBeta[cnt2]/targetOmega**2*targetDiffOmega
        targetInt = targetD[cnt2]+1
        targetProcessAction = simps(targetInt, interpGrid)
        targetProcessTimeNorm[cnt2] = targetProcessTime/targetProcessAction
        targetOmegaNorm[cnt2] = targetProcessAction*targetOmega
      #
      featureProcessTimeNormScaled = np.zeros((y_count,z_count))
      featureOmegaNormScaled = np.zeros((y_count,z_count))
      for cnt3 in range(y_count):
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
      distanceTotal = np.zeros((y_count,z_count))
      sigma = np.zeros((y_count,z_count))
      for metric in self.metrics:
        name = "{}_{}_{}".format(feat.split("|")[-1], targ.split("|")[-1], metric.estimator.name)
      output = metric.evaluate((newfeatureData,newtargetData), multiOutput='raw_values')
      for cnt2 in range(y_count):
          distanceSum = abs(np.sum(output[cnt2]))
          sigmaSum = 0
          for cnt3 in range(z_count):
            distanceTotal[cnt2][cnt3] = distanceSum
            sigmaSum += output[cnt2][cnt3]**2
          for cnt3 in range(z_count):
            sigma[cnt2][cnt3] = (1/z_count*sigmaSum)**0.5
      rlz = []
      for cnt in range(y_count):
        outputDict = {}
        outputDict[name] = np.atleast_1d(output[cnt])
        outputDict['pivot_parameter'] = timeParameter
        outputDict[nameFeat[1]+'_'+nameTarg[1]+'_total_distance'] = distanceTotal[cnt]
        outputDict[nameFeat[1]+'_'+nameTarg[1]+'_process_time'] = newfeatureData[1][cnt]
        outputDict[nameFeat[1]+'_'+nameTarg[1]+'_standard_deviation'] = sigma[cnt]
        rlz.append(outputDict)
      realization_array.append(rlz)
    #---------------
    for cnt in range(len(realization_array[0])):
      out = {}
      for cnt2 in range(len(realization_array)):
        for key, val in realization_array[cnt2][cnt].items():
          out[key] = val
      realizations.append(out)
    #return outputDict
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
