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
Created on July 10, 2013

@author: alfoa, wangc, dgarrett622
"""
#External Modules---------------------------------------------------------------
import numpy as np
import os
import copy
from collections import OrderedDict, defaultdict
import six
import xarray as xr
import scipy.stats as stats
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import utils
from ...utils import InputData, InputTypes
from ...utils import mathUtils
from ... import Files
#Internal Modules End-----------------------------------------------------------

class BasicStatistics(PostProcessorInterface):
  """
    BasicStatistics filter class. It computes all the most popular statistics
  """

  scalarVals = ['expectedValue',
                'minimum',
                'maximum',
                'median',
                'variance',
                'sigma',
                'percentile',
                'variationCoefficient',
                'skewness',
                'kurtosis',
                'samples',
                'higherPartialVariance',   # Statistic metric not available yet
                'higherPartialSigma',      # Statistic metric not available yet
                'lowerPartialSigma',       # Statistic metric not available yet
                'lowerPartialVariance'     # Statistic metric not available yet
                ]
  vectorVals = ['sensitivity',
                'covariance',
                'pearson',
                'spearman',
                'NormalizedSensitivity',
                'VarianceDependentSensitivity']
  # quantities that the standard error can be computed
  steVals    = ['expectedValue_ste',
                'median_ste',
                'variance_ste',
                'sigma_ste',
                'skewness_ste',
                'kurtosis_ste',
                'percentile_ste']

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inputSpecification = super().getInputSpecification()

    for scalar in cls.scalarVals:
      scalarSpecification = InputData.parameterInputFactory(scalar, contentType=InputTypes.StringListType)
      if scalar == 'percentile':
        #percent is a string type because otherwise we can't tell 95.0 from 95
        # which matters because the number is used in output.
        scalarSpecification.addParam("percent", InputTypes.StringListType)
        # percentile has additional "interpolation" parameter
        scalarSpecification.addParam("interpolation",
                                     param_type=InputTypes.makeEnumType("interpolation",
                                                                        "interpolationType",
                                                                        ["linear", "midpoint"]),
                                     default="linear",
                                     descr="""Interpolation method for percentile calculation.
                                              'linear' uses linear interpolation between nearest
                                              data points while 'midpoint' uses the average of the
                                              nearest data points.""")
      if scalar == 'median':
        # median has additional "interpolation" parameter
        scalarSpecification.addParam("interpolation",
                                     param_type=InputTypes.makeEnumType("interpolation",
                                                                        "interpolationType",
                                                                        ["linear", "midpoint"]),
                                     default="linear",
                                     descr="""Interpolation method for median calculation. 'linear'
                                              uses linear interpolation between nearest data points
                                              while 'midpoint' uses the average of the nearest data
                                              points.""")
      scalarSpecification.addParam("prefix", InputTypes.StringType)
      inputSpecification.addSub(scalarSpecification)

    for vector in cls.vectorVals:
      vectorSpecification = InputData.parameterInputFactory(vector)
      vectorSpecification.addParam("prefix", InputTypes.StringType)
      features = InputData.parameterInputFactory('features',
                                contentType=InputTypes.StringListType)
      vectorSpecification.addSub(features)
      targets = InputData.parameterInputFactory('targets',
                                contentType=InputTypes.StringListType)
      vectorSpecification.addSub(targets)
      inputSpecification.addSub(vectorSpecification)

    pivotParameterInput = InputData.parameterInputFactory('pivotParameter', contentType=InputTypes.StringType)
    inputSpecification.addSub(pivotParameterInput)

    datasetInput = InputData.parameterInputFactory('dataset', contentType=InputTypes.BoolType)
    inputSpecification.addSub(datasetInput)

    methodsToRunInput = InputData.parameterInputFactory("methodsToRun", contentType=InputTypes.StringType)
    inputSpecification.addSub(methodsToRunInput)

    biasedInput = InputData.parameterInputFactory("biased", contentType=InputTypes.BoolType)
    inputSpecification.addSub(biasedInput)

    multipleFeaturesInput = InputData.parameterInputFactory("multipleFeatures", contentType=InputTypes.BoolType)
    inputSpecification.addSub(multipleFeaturesInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.parameters = {}  # parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.acceptedCalcParam = self.scalarVals + self.vectorVals
    self.what = self.acceptedCalcParam  # what needs to be computed... default...all
    self.methodsToRun = []  # if a function is present, its outcome name is here stored... if it matches one of the known outcomes, the pp is going to use the function to compute it
    self.printTag = 'PostProcessor BASIC STATISTIC'
    self.biased = False # biased statistics?
    self.pivotParameter = None # time-dependent statistics pivot parameter
    self.pivotValue = None # time-dependent statistics pivot parameter values
    self.dynamic        = False # is it time-dependent?
    self.sampleTag      = None  # Tag used to track samples
    self.pbPresent      = False # True if the ProbabilityWeight is available
    self.realizationWeight = None # The joint probabilities
    self.steMetaIndex   = 'targets' # when Dataset is requested as output, the default index of ste metadata is ['targets', self.pivotParameter]
    self.multipleFeatures = True # True if multiple features are employed in linear regression as feature inputs
    self.sampleSize     = None # number of sample size
    self.calculations   = {}
    self.validDataType  = ['PointSet', 'HistorySet', 'DataSet'] # The list of accepted types of DataObject

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, (inputDataset, pbWeights), tuple, the dataset of inputs and the corresponding variable probability weight
    """
    # The BasicStatistics postprocessor only accept DataObjects
    self.dynamic = False
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0:
      self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")

    pbWeights = None
    if type(currentInput).__name__ == 'tuple':
      return currentInput
    # TODO: convert dict to dataset, I think this will be removed when DataSet is used by other entities that
    # are currently using this Basic Statisitics PostProcessor.
    if type(currentInput).__name__ == 'dict':
      if 'targets' not in currentInput.keys():
        self.raiseAnError(IOError, 'Did not find targets in the input dictionary')
      inputDataset = xr.Dataset()
      for var, val in currentInput['targets'].items():
        inputDataset[var] = val
      if 'metadata' in currentInput.keys():
        metadata = currentInput['metadata']
        self.pbPresent = True if 'ProbabilityWeight' in metadata else False
        if self.pbPresent:
          pbWeights = xr.Dataset()
          self.realizationWeight = xr.Dataset()
          self.realizationWeight['ProbabilityWeight'] = metadata['ProbabilityWeight']/metadata['ProbabilityWeight'].sum()
          for target in self.parameters['targets']:
            pbName = 'ProbabilityWeight-' + target
            if pbName in metadata:
              pbWeights[target] = metadata[pbName]/metadata[pbName].sum()
            elif self.pbPresent:
              pbWeights[target] = self.realizationWeight['ProbabilityWeight']
        else:
          self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
      else:
        self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
      if 'RAVEN_sample_ID' not in inputDataset.sizes.keys():
        self.raiseAWarning('BasicStatisitics postprocessor did not detect RAVEN_sample_ID! Assuming the first dimension of given data...')
        self.sampleTag = utils.first(inputDataset.sizes.keys())
      return inputDataset, pbWeights

    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'BasicStatistics postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)

    # extract all required data from input DataObjects, an input dataset is constructed
    dataSet = currentInput.asDataset()
    try:
      inputDataset = dataSet[self.parameters['targets']]
    except KeyError:
      missing = [var for var in self.parameters['targets'] if var not in dataSet]
      self.raiseAnError(KeyError, "Variables: '{}' missing from dataset '{}'!".format(", ".join(missing),currentInput.name))
    self.sampleTag = currentInput.sampleTag

    if currentInput.type == 'HistorySet':
      dims = inputDataset.sizes.keys()
      if self.pivotParameter is None:
        if len(dims) > 1:
          self.raiseAnError(IOError, self, 'Time-dependent statistics is requested (HistorySet) but no pivotParameter \
                got inputted!')
      elif self.pivotParameter not in dims:
        self.raiseAnError(IOError, self, 'Pivot parameter', self.pivotParameter, 'is not the associated index for \
                requested variables', ','.join(self.parameters['targets']))
      else:
        self.dynamic = True
        if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
          self.raiseAnError(IOError, "The data provided by the data objects", currentInput.name, "is not synchronized!")
        self.pivotValue = inputDataset[self.pivotParameter].values
        if self.pivotValue.size != len(inputDataset.groupby(self.pivotParameter)):
          msg = "Duplicated values were identified in pivot parameter, please use the 'HistorySetSync'" + \
          " PostProcessor to syncronize your data before running 'BasicStatistics' PostProcessor."
          self.raiseAnError(IOError, msg)
    # extract all required meta data
    metaVars = currentInput.getVars('meta')
    self.pbPresent = True if 'ProbabilityWeight' in metaVars else False
    if self.pbPresent:
      pbWeights = xr.Dataset()
      self.realizationWeight = dataSet[['ProbabilityWeight']]/dataSet[['ProbabilityWeight']].sum()
      for target in self.parameters['targets']:
        pbName = 'ProbabilityWeight-' + target
        if pbName in metaVars:
          pbWeights[target] = dataSet[pbName]/dataSet[pbName].sum()
        elif self.pbPresent:
          pbWeights[target] = self.realizationWeight['ProbabilityWeight']
    else:
      self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')

    return inputDataset, pbWeights

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the BasicStatistic pp. In here the working dir is
      grepped.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    #construct a list of all the parameters that have requested values into self.allUsedParams
    self.allUsedParams = set()
    for metricName in self.scalarVals + self.vectorVals:
      if metricName in self.toDo.keys():
        for entry in self.toDo[metricName]:
          self.allUsedParams.update(entry['targets'])
          try:
            self.allUsedParams.update(entry['features'])
          except KeyError:
            pass

    #for backward compatibility, compile the full list of parameters used in Basic Statistics calculations
    self.parameters['targets'] = list(self.allUsedParams)
    super().initialize(runInfo, inputs, initDict)
    inputObj = inputs[-1] if type(inputs) == list else inputs
    if inputObj.type == 'HistorySet':
      self.dynamic = True
    inputMetaKeys = []
    outputMetaKeys = []
    for metric, infos in self.toDo.items():
      if metric in self.scalarVals + self.vectorVals:
        steMetric = metric + '_ste'
        if steMetric in self.steVals:
          for info in infos:
            prefix = info['prefix']
            for target in info['targets']:
              if metric == 'percentile':
                for strPercent in info['strPercent']:
                  metaVar = prefix + '_' + strPercent + '_ste_' + target if not self.outputDataset else metric + '_ste'
                  metaDim = inputObj.getDimensions(target)
                  if len(metaDim[target]) == 0:
                    inputMetaKeys.append(metaVar)
                  else:
                    outputMetaKeys.append(metaVar)
              else:
                metaVar = prefix + '_ste_' + target if not self.outputDataset else metric + '_ste'
                metaDim = inputObj.getDimensions(target)
                if len(metaDim[target]) == 0:
                  inputMetaKeys.append(metaVar)
                else:
                  outputMetaKeys.append(metaVar)
    metaParams = {}
    if not self.outputDataset:
      if len(outputMetaKeys) > 0:
        metaParams = {key:[self.pivotParameter] for key in outputMetaKeys}
    else:
      if len(outputMetaKeys) > 0:
        params = {}
        for key in outputMetaKeys + inputMetaKeys:
          # percentile standard error has additional index
          if key == 'percentile_ste':
            params[key] = [self.pivotParameter, self.steMetaIndex, 'percent']
          else:
            params[key] = [self.pivotParameter, self.steMetaIndex]
        metaParams.update(params)
      elif len(inputMetaKeys) > 0:
        params = {}
        for key in inputMetaKeys:
          # percentile standard error has additional index
          if key == 'percentile_ste':
            params[key] = [self.steMetaIndex, 'percent']
          else:
            params[key] = [self.steMetaIndex]
        metaParams.update(params)
    metaKeys = inputMetaKeys + outputMetaKeys
    self.addMetaKeys(metaKeys,metaParams)

  def _handleInput(self, paramInput, childVals=None):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ In, childVals, list, optional, quantities requested from child statistical object
      @ Out, None
    """
    if childVals is None:
      childVals = []
    self.toDo = {}
    for child in paramInput.subparts:
      tag = child.getName()
      #because percentile is strange (has attached parameters), we address it first
      if tag in self.scalarVals + self.vectorVals:
        if 'prefix' not in child.parameterValues:
          self.raiseAnError(IOError, "No prefix is provided for node: ", tag)
        #get the prefix
        prefix = child.parameterValues['prefix']
      if tag == 'percentile':
        #get targets
        targets = set(child.value)
        #what if user didn't give any targets?
        if len(targets)<1:
          self.raiseAWarning('No targets were specified in text of <'+tag+'>!  Skipping metric...')
          continue
        #prepare storage dictionary, keys are percentiles, values are set(targets)
        if tag not in self.toDo.keys():
          self.toDo[tag] = [] # list of {'targets':(), 'prefix':str, 'percent':str, 'interpolation':str}
        if 'percent' not in child.parameterValues:
          reqPercent = [0.05, 0.95]
          strPercent = ['5','95']
        else:
          reqPercent = set(utils.floatConversion(percent)/100. for percent in child.parameterValues['percent'])
          strPercent = set(percent for percent in child.parameterValues['percent'])
        if 'interpolation' not in child.parameterValues:
          interpolation = 'linear'
        else:
          interpolation = child.parameterValues['interpolation']
        self.toDo[tag].append({'targets':set(targets),
                               'prefix':prefix,
                               'percent':reqPercent,
                               'strPercent':strPercent,
                               'interpolation':interpolation})
      # median also has an attached parameter
      elif tag == 'median':
        if tag not in self.toDo.keys():
          self.toDo[tag] = [] # list of {'targets':{}, 'prefix':str, 'interpolation':str}
        if 'interpolation' not in child.parameterValues:
          interpolation = 'linear'
        else:
          interpolation = child.parameterValues['interpolation']
        self.toDo[tag].append({'targets':set(child.value),
                               'prefix':prefix,
                               'interpolation':interpolation})
      elif tag in self.scalarVals:
        if tag not in self.toDo.keys():
          self.toDo[tag] = [] # list of {'targets':(), 'prefix':str}
        self.toDo[tag].append({'targets':set(child.value),
                               'prefix':prefix})
      elif tag in self.vectorVals:
        if tag not in self.toDo.keys():
          self.toDo[tag] = [] # list of {'targets':(),'features':(), 'prefix':str}
        tnode = child.findFirst('targets')
        if tnode is None:
          self.raiseAnError('Request for vector value <'+tag+'> requires a "targets" node, and none was found!')
        fnode = child.findFirst('features')
        if fnode is None:
          self.raiseAnError('Request for vector value <'+tag+'> requires a "features" node, and none was found!')
        # we're storing toDo[tag] as a list of dictionaries.  This is because the user might specify multiple
        #   nodes with the same metric (tag), but with different targets and features.  For instance, the user might
        #   want the sensitivity of A and B to X and Y, and the sensitivity of C to W and Z, but not the sensitivity
        #   of A to W.  If we didn't keep them separate, we could potentially waste a fair number of calculations.
        self.toDo[tag].append({'targets':set(tnode.value),
                               'features':set(fnode.value),
                               'prefix':prefix})
      elif tag == "biased":
        self.biased = child.value
      elif tag == "pivotParameter":
        self.pivotParameter = child.value
      elif tag == "dataset":
        self.outputDataset = child.value
      elif tag == "multipleFeatures":
        self.multipleFeatures = child.value
      else:
        if tag not in childVals:
          self.raiseAWarning('Unrecognized node in BasicStatistics "',tag,'" has been ignored!')

    assert (len(self.toDo)>0), self.raiseAnError(IOError, 'BasicStatistics needs parameters to work on! Please check input for PP: ' + self.name)

  def _computePower(self, p, dataset):
    """
      Compute the p-th power of weights
      @ In, p, int, the power
      @ In, dataset, xarray.Dataset, probability weights of all input variables
      @ Out, pw, xarray.Dataset, the p-th power of weights
    """
    pw = {}
    coords = dataset.coords
    for target, targValue in dataset.variables.items():
      ##remove index variable
      if target in coords:
        continue
      pw[target] = np.power(targValue,p)
    pw = xr.Dataset(data_vars=pw,coords=coords)
    return pw

  def __computeVp(self,p,weights):
    """
      Compute the sum of p-th power of weights
      @ In, p, int, the power
      @ In, weights, xarray.Dataset, probability weights of all input variables
      @ Out, vp, xarray.Dataset, the sum of p-th power of weights
    """
    vp = self._computePower(p,weights)
    vp = vp.sum()
    return vp

  def __computeEquivalentSampleSize(self,weights):
    """
      Compute the equivalent sample size for given probability weights
      @ In, weights, xarray.Dataset, probability weights of all input variables
      @ Out, equivalentSize, xarray.Dataset, the equivalent sample size
    """
    # The equivalent sample size for given samples, i.e. (sum of weights) squared / sum of the squared weights
    # The definition of this quantity can be found:
    # R. F. Potthoff, M. A. Woodbury and K. G. Manton, "'Equivalent SampleSize' and 'Equivalent Degrees of Freedom'
    # Refinements for Inference Using Survey Weights Under Superpopulation Models", Journal of the American Statistical
    # Association, Vol. 87, No. 418 (1992)
    v1Square = self.__computeVp(1,weights)**2
    v2 = self.__computeVp(2,weights)
    equivalentSize = v1Square/v2
    return equivalentSize

  def __computeUnbiasedCorrection(self,order,weightsOrN):
    """
      Compute unbiased correction given weights and momement order
      Reference paper:
      Lorenzo Rimoldini, "Weighted skewness and kurtosis unbiased by sample size", http://arxiv.org/pdf/1304.6564.pdf
      @ In, order, int, moment order
      @ In, weightsOrN, xarray.Dataset or int, if xarray.Dataset -> weights else -> number of samples
      @ Out, corrFactor, xarray.Dataset or int, xarray.Dataset (order <=3) or tuple of xarray.Dataset (order ==4),
        the unbiased correction factor if weightsOrN is xarray.Dataset else integer
    """
    if order > 4:
      self.raiseAnError(RuntimeError,"computeUnbiasedCorrection is implemented for order <=4 only!")
    if type(weightsOrN).__name__ not in ['int','int8','int16','int64','int32']:
      if order == 2:
        V1, v1Square, V2 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**2.0, self.__computeVp(2, weightsOrN)
        corrFactor   = v1Square/(v1Square-V2)
      elif order == 3:
        V1, v1Cubic, V2, V3 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**3.0, self.__computeVp(2, weightsOrN), self.__computeVp(3, weightsOrN)
        corrFactor   =  v1Cubic/(v1Cubic-3.0*V2*V1+2.0*V3)
      elif order == 4:
        V1, v1Square, V2, V3, V4 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**2.0, self.__computeVp(2, weightsOrN), self.__computeVp(3, weightsOrN), self.__computeVp(4, weightsOrN)
        numer1 = v1Square*(v1Square**2.0-3.0*v1Square*V2+2.0*V1*V3+3.0*V2**2.0-3.0*V4)
        numer2 = 3.0*v1Square*(2.0*v1Square*V2-2.0*V1*V3-3.0*V2**2.0+3.0*V4)
        denom = (v1Square-V2)*(v1Square**2.0-6.0*v1Square*V2+8.0*V1*V3+3.0*V2**2.0-6.0*V4)
        corrFactor = numer1/denom ,numer2/denom
    else:
      if   order == 2:
        corrFactor   = float(weightsOrN)/(float(weightsOrN)-1.0)
      elif order == 3:
        corrFactor   = (float(weightsOrN)**2.0)/((float(weightsOrN)-1)*(float(weightsOrN)-2))
      elif order == 4:
        corrFactor = (float(weightsOrN)*(float(weightsOrN)**2.0-2.0*float(weightsOrN)+3.0))/((float(weightsOrN)-1)*(float(weightsOrN)-2)*(float(weightsOrN)-3)),(3.0*float(weightsOrN)*(2.0*float(weightsOrN)-3.0))/((float(weightsOrN)-1)*(float(weightsOrN)-2)*(float(weightsOrN)-3))
    return corrFactor

  def _computeKurtosis(self, arrayIn, expValue, variance, pbWeight=None, dim=None):
    """
      Method to compute the Kurtosis (fisher) of an array of observations
      @ In, arrayIn, xarray.Dataset, the dataset from which the Kurtosis needs to be estimated
      @ In, expValue, xarray.Dataset, expected value of arrayIn
      @ In, variance, xarray.Dataset, variance of arrayIn
      @ In, pbWeight, xarray.DataSet, optional, the reliability weights that correspond to the values in 'array'.
        If not present, an unweighted approach is used
      @ Out, result, xarray.Dataset, the Kurtosis of the dataset arrayIn.
    """
    if dim is None:
      dim = self.sampleTag
    vr = self._computePower(2.0, variance)
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(4,pbWeight) if not self.biased else 1.0
      vp = 1.0/self.__computeVp(1,pbWeight)
      p4 = ((arrayIn - expValue)**4.0 * pbWeight).sum(dim=dim)
      if not self.biased:
        p2 = ((arrayIn - expValue)**2.0 * pbWeight).sum(dim=dim)
        result = -3.0 + (p4*unbiasCorr[0]*vp - (p2*vp)**2.0 * unbiasCorr[1]) / vr
      else:
        result = -3.0 + (p4 * vp * unbiasCorr) / vr
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(4,int(arrayIn.sizes[dim])) if not self.biased else 1.0
      vp = 1.0 / arrayIn.sizes[dim]
      p4 = ((arrayIn - expValue)**4.0).sum(dim=dim)
      if not self.biased:
        p2 = (arrayIn - expValue).var(dim=dim)
        result = -3.0 + (p4*unbiasCorr[0]*vp-p2**2.0*unbiasCorr[1]) / vr
      else:
        result = -3.0 + (p4*unbiasCorr*vp) / vr
    return result

  def _computeSkewness(self, arrayIn, expValue, variance, pbWeight=None, dim=None):
    """
      Method to compute the skewness of an array of observations
      @ In, arrayIn, xarray.Dataset, the dataset from which the skewness needs to be estimated
      @ In, expValue, xarray.Dataset, expected value of arrayIn
      @ In, variance, xarray.Dataset, variance value of arrayIn
      @ In, pbWeight, xarray.Dataset, optional, the reliability weights that correspond to dataset arrayIn.
        If not present, an unweighted approach is used
      @ Out, result, xarray.Dataset, the skewness of the dataset arrayIn
    """
    if dim is None:
      dim = self.sampleTag
    vr = self._computePower(1.5, variance)
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(3,pbWeight) if not self.biased else 1.0
      vp = 1.0/self.__computeVp(1,pbWeight)
      result = ((arrayIn - expValue)**3 * pbWeight).sum(dim=dim) * vp * unbiasCorr / vr
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(3,int(arrayIn.sizes[dim])) if not self.biased else 1.0
      vp = 1.0 / arrayIn.sizes[dim]
      result = ((arrayIn - expValue)**3).sum(dim=dim) * vp * unbiasCorr / vr
    return result

  def _computeVariance(self, arrayIn, expValue, pbWeight=None, dim = None):
    """
      Method to compute the Variance (fisher) of an array of observations
      @ In, arrayIn, xarray.Dataset, the dataset from which the Variance needs to be estimated
      @ In, expValue, xarray.Dataset, expected value of arrayIn
      @ In, pbWeight, xarray.Dataset, optional, the reliability weights that correspond to dataset arrayIn.
        If not present, an unweighted approach is used
      @ Out, result, xarray.Dataset, the Variance of the dataset arrayIn
    """
    if dim is None:
      dim = self.sampleTag
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(2,pbWeight) if not self.biased else 1.0
      vp = 1.0/self.__computeVp(1,pbWeight)
      result = ((arrayIn-expValue)**2 * pbWeight).sum(dim=dim) * vp * unbiasCorr
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(2,int(arrayIn.sizes[dim])) if not self.biased else 1.0
      result =  (arrayIn-expValue).var(dim=dim) * unbiasCorr
    return result

  def _computeLowerPartialVariance(self, arrayIn, medValue, pbWeight=None, dim = None):
    """
      Method to compute the lower partial variance of an array of observations
      @ In, arrayIn, xarray.Dataset, the dataset from which the Variance needs to be estimated
      @ In, medValue, xarray.Dataset, median value of arrayIn
      @ In, pbWeight, xarray.Dataset, optional, the reliability weights that correspond to dataset arrayIn.
        If not present, an unweighted approach is used
      @ Out, result, xarray.Dataset, the lower partial variance of the dataset arrayIn
    """
    if dim is None:
      dim = self.sampleTag
    diff = (medValue-arrayIn).clip(min=0)
    if pbWeight is not None:
      vp = 1.0/self.__computeVp(1,pbWeight)
      result = ((diff)**2 * pbWeight).sum(dim=dim) * vp
    else:
      result = diff.var(dim=dim)
    return result

  def _computeHigherPartialVariance(self, arrayIn, medValue, pbWeight=None, dim = None):
    """
      Method to compute the higher partial variance of an array of observations
      @ In, arrayIn, xarray.Dataset, the dataset from which the Variance needs to be estimated
      @ In, medValue, xarray.Dataset, median value of arrayIn
      @ In, pbWeight, xarray.Dataset, optional, the reliability weights that correspond to dataset arrayIn.
        If not present, an unweighted approach is used
      @ Out, result, xarray.Dataset, the higher partial variance of the dataset arrayIn
    """
    if dim is None:
      dim = self.sampleTag
    diff = (arrayIn-medValue).clip(min=0)
    if pbWeight is not None:
      vp = 1.0/self.__computeVp(1,pbWeight)
      result = ((diff)**2 * pbWeight).sum(dim=dim) * vp
    else:
      result = diff.var(dim=dim)
    return result

  def _computeSigma(self,arrayIn,variance,pbWeight=None):
    """
      Method to compute the sigma of an array of observations
      @ In, arrayIn, xarray.Dataset, the dataset from which the standard deviation needs to be estimated
      @ In, variance, xarray.Dataset, variance of arrayIn
      @ In, pbWeight, xarray.Dataset, optional, the reliability weights that correspond to dataset arrayIn.
        If not present, an unweighted approach is used
      @ Out, sigma, xarray.Dataset, the sigma of the dataset of arrayIn
    """
    return np.sqrt(variance)

  def _computeWeightedPercentile(self,arrayIn,pbWeight,interpolation='linear',percent=[0.5]):
    """
      Method to compute the weighted percentile in a array of data
      @ In, arrayIn, list/numpy.array, the array of values from which the percentile needs to be estimated
      @ In, pbWeight, list/numpy.array, the reliability weights that correspond to the values in 'array'
      @ In, interpolation, str, 'linear' or 'midpoint'
      @ In, percent, list/numpy.array, the percentile(s) that needs to be computed (between 0.01 and 1.0)
      @ Out, result, list, the percentile(s)
    """

    # only do the argsort once for all requested percentiles
    idxs                   = np.argsort(np.asarray(list(zip(pbWeight,arrayIn)))[:,1])
    # Inserting [0.0,arrayIn[idxs[0]]] is needed when few samples are generated and
    # a percentile that is < that the first pb weight is requested. Otherwise the median
    # is returned.
    sortedWeightsAndPoints = np.insert(np.asarray(list(zip(pbWeight[idxs],arrayIn[idxs]))),0,[0.0,arrayIn[idxs[0]]],axis=0)
    weightsCDF             = np.cumsum(sortedWeightsAndPoints[:,0])
    weightsCDF            /= weightsCDF[-1]
    if interpolation == 'linear':
      result = np.interp(percent, weightsCDF, sortedWeightsAndPoints[:, 1]).tolist()
    elif interpolation == 'midpoint':
      result = [self._computeSingleWeightedPercentile(pct, weightsCDF, sortedWeightsAndPoints) for pct in percent]

    return result

  def _computeSingleWeightedPercentile(self, pct, weightsCDF, sortedWeightsAndPoints):
    """
      Method to compute a single percentile
      @ In, pct, float, the percentile
      @ In, weightsCDF, numpy.array, the cumulative sum of weights (CDF)
      @ In, sortedWeightsAndPoints, numpy.array, array of weights and data points
      @ Out, result, float, the percentile
    """

    # This step returns the index of the array which is < than the percentile, because
    # the insertion create another entry, this index should shift to the bigger side
    indexL = utils.first(np.asarray(weightsCDF >= pct).nonzero())[0]
    # This step returns the indices (list of index) of the array which is > than the percentile
    indexH = utils.first(np.asarray(weightsCDF > pct).nonzero())
    try:
      # if the indices exists that means the desired percentile lies between two data points
      # with index as indexL and indexH[0]. Calculate the midpoint of these two points
      result = 0.5*(sortedWeightsAndPoints[indexL,1]+sortedWeightsAndPoints[indexH[0],1])
    except IndexError:
      result = sortedWeightsAndPoints[indexL,1]

    return result

  def __runLocal(self, inputData):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In, inputData, tuple,  (inputDataset, pbWeights), tuple, the dataset of inputs and the corresponding
        variable probability weight
      @ Out, outputSet or outputDict, xarray.Dataset or dict, dataset or dictionary containing the results
    """
    inputDataset, pbWeights = inputData[0], inputData[1]
    #storage dictionary for skipped metrics
    self.skipped = {}
    #construct a dict of required computations
    needed = dict((metric,{'targets':set(),'percent':set(),'interpolation':''}) for metric in self.scalarVals)
    needed.update(dict((metric,{'targets':set(),'features':set()}) for metric in self.vectorVals))
    for metric, params in self.toDo.items():
      if metric in self.scalarVals + self.vectorVals:
        for entry in params:
          needed[metric]['targets'].update(entry['targets'])
          try:
            needed[metric]['features'].update(entry['features'])
          except KeyError:
            pass
          try:
            needed[metric]['percent'].update(entry['percent'])
          except KeyError:
            pass
          try:
            needed[metric]['interpolation'] = entry['interpolation']
          except KeyError:
            pass
    # variable                     | needs                  | needed for
    # --------------------------------------------------------------------
    # skewness needs               | expectedValue,variance |
    # kurtosis needs               | expectedValue,variance |
    # median needs                 |                        | lowerPartialVariance, higherPartialVariance
    # percentile needs             | expectedValue,sigma    |
    # maximum needs                |                        |
    # minimum needs                |                        |
    # covariance needs             |                        | pearson,VarianceDependentSensitivity,NormalizedSensitivity
    # NormalizedSensitivity        | covariance,VarDepSens  |
    # VarianceDependentSensitivity | covariance             | NormalizedSensitivity
    # sensitivity needs            |                        |
    # pearson needs                | covariance             |
    # sigma needs                  | variance               | variationCoefficient
    # variance                     | expectedValue          | sigma, skewness, kurtosis
    # expectedValue                |                        | variance, variationCoefficient, skewness, kurtosis,
    #                              |                        | lowerPartialVariance, higherPartialVariance
    # lowerPartialVariance needs   | expectedValue,median   | lowerPartialSigma
    # lowerPartialSigma needs      | lowerPartialVariance   |
    # higherPartialVariance needs  | expectedValue,median   | higherPartialSigma
    # higherPartialSigma needs     | higherPartialVariance  |

    # update needed dictionary when standard errors are requested
    needed['expectedValue']['targets'].update(needed['sigma']['targets'])
    needed['expectedValue']['targets'].update(needed['variationCoefficient']['targets'])
    needed['expectedValue']['targets'].update(needed['variance']['targets'])
    needed['expectedValue']['targets'].update(needed['median']['targets'])
    needed['expectedValue']['targets'].update(needed['skewness']['targets'])
    needed['expectedValue']['targets'].update(needed['kurtosis']['targets'])
    needed['expectedValue']['targets'].update(needed['NormalizedSensitivity']['targets'])
    needed['expectedValue']['targets'].update(needed['NormalizedSensitivity']['features'])
    needed['expectedValue']['targets'].update(needed['percentile']['targets'])
    needed['sigma']['targets'].update(needed['expectedValue']['targets'])
    needed['sigma']['targets'].update(needed['percentile']['targets'])
    needed['variance']['targets'].update(needed['sigma']['targets'])
    needed['lowerPartialVariance']['targets'].update(needed['lowerPartialSigma']['targets'])
    needed['higherPartialVariance']['targets'].update(needed['higherPartialSigma']['targets'])
    needed['median']['targets'].update(needed['lowerPartialVariance']['targets'])
    needed['median']['targets'].update(needed['higherPartialVariance']['targets'])
    needed['covariance']['targets'].update(needed['NormalizedSensitivity']['targets'])
    needed['covariance']['features'].update(needed['NormalizedSensitivity']['features'])
    needed['VarianceDependentSensitivity']['targets'].update(needed['NormalizedSensitivity']['targets'])
    needed['VarianceDependentSensitivity']['features'].update(needed['NormalizedSensitivity']['features'])
    needed['covariance']['targets'].update(needed['pearson']['targets'])
    needed['covariance']['features'].update(needed['pearson']['features'])
    needed['covariance']['targets'].update(needed['VarianceDependentSensitivity']['targets'])
    needed['covariance']['features'].update(needed['VarianceDependentSensitivity']['features'])

    for metric, params in needed.items():
      needed[metric]['targets'] = list(params['targets'])
      try:
        needed[metric]['features'] = list(params['features'])
      except KeyError:
        pass

    #
    # BEGIN actual calculations
    #

    calculations = {}

    #################
    # SCALAR VALUES #
    #################
    #
    # samples
    #
    self.sampleSize = inputDataset.sizes[self.sampleTag]
    metric = 'samples'
    if len(needed[metric]['targets']) > 0:
      self.raiseADebug('Starting "'+metric+'"...')
      if self.dynamic:
        nt = inputDataset.sizes[self.pivotParameter]
        sampleMat = np.zeros((len(self.parameters['targets']), len(self.pivotValue)))
        sampleMat.fill(self.sampleSize)
        samplesDA = xr.DataArray(sampleMat,dims=('targets', self.pivotParameter), coords={'targets':self.parameters['targets'], self.pivotParameter:self.pivotValue})
      else:
        sampleMat = np.zeros(len(self.parameters['targets']))
        sampleMat.fill(self.sampleSize)
        samplesDA = xr.DataArray(sampleMat,dims=('targets'), coords={'targets':self.parameters['targets']})
      self.calculations[metric] = samplesDA
      calculations[metric] = samplesDA
    #
    # expected value
    #
    metric = 'expectedValue'
    if len(needed[metric]['targets']) > 0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      if self.pbPresent:
        relWeight = pbWeights[list(needed[metric]['targets'])]
        equivalentSize = self.__computeEquivalentSampleSize(relWeight)
        dataSet = dataSet * relWeight
        expectedValueDS = dataSet.sum(dim = self.sampleTag)
        calculations['equivalentSamples'] = equivalentSize
      else:
        expectedValueDS = dataSet.mean(dim = self.sampleTag)
      self.calculations[metric] = expectedValueDS
      calculations[metric] = expectedValueDS
    #
    # variance
    #
    metric = 'variance'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      meanSet = calculations['expectedValue'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      varianceDS = self._computeVariance(dataSet,meanSet,pbWeight=relWeight,dim=self.sampleTag)
      calculations[metric] = varianceDS
    #
    # sigma
    #
    metric = 'sigma'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      sigmaDS = self._computePower(0.5,calculations['variance'][list(needed[metric]['targets'])])
      self.calculations[metric] = sigmaDS
      calculations[metric] = sigmaDS
    #
    # coeff of variation (sigma/mu)
    #
    metric = 'variationCoefficient'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      calculations[metric] = calculations['sigma'][needed[metric]['targets']] / calculations['expectedValue'][needed[metric]['targets']]
    #
    # skewness
    #
    metric = 'skewness'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      meanSet = calculations['expectedValue'][list(needed[metric]['targets'])]
      varianceSet = calculations['variance'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      calculations[metric] = self._computeSkewness(dataSet,meanSet,varianceSet,pbWeight=relWeight,dim=self.sampleTag)
    #
    # kurtosis
    #
    metric = 'kurtosis'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      meanSet = calculations['expectedValue'][list(needed[metric]['targets'])]
      varianceSet = calculations['variance'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      calculations[metric] = self._computeKurtosis(dataSet,meanSet,varianceSet,pbWeight=relWeight,dim=self.sampleTag)
    #
    # median
    #
    metric = 'median'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      if self.pbPresent:
        # if all weights are the same, calculate percentile with xarray, no need for _computeWeightedPercentile
        allSameWeight = True
        for target in needed[metric]['targets']:
          targWeight = relWeight[target].values
          if targWeight.min() != targWeight.max():
            allSameWeight = False
        if allSameWeight:
          medianSet = dataSet.median(dim=self.sampleTag)
        else:
          medianSet = xr.Dataset()
          relWeight = pbWeights[list(needed[metric]['targets'])]
          for target in needed[metric]['targets']:
            targWeight = relWeight[target].values
            targDa = dataSet[target]
            if self.pivotParameter in targDa.sizes.keys():
              quantile = [self._computeWeightedPercentile(group.values,targWeight,needed[metric]['interpolation'],percent=[0.5])[0] for label,group in targDa.groupby(self.pivotParameter)]
            else:
              quantile = self._computeWeightedPercentile(targDa.values,targWeight,needed[metric]['interpolation'],percent=[0.5])[0]
            if self.pivotParameter in targDa.sizes.keys():
              da = xr.DataArray(quantile,dims=(self.pivotParameter),coords={self.pivotParameter:self.pivotValue})
            else:
              da = xr.DataArray(quantile)
            medianSet[target] = da
      else:
        medianSet = dataSet.median(dim=self.sampleTag)
      self.calculations[metric] = medianSet
      calculations[metric] = medianSet
    #
    # lowerPartialVariance
    #
    metric = 'lowerPartialVariance'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      medianSet = calculations['median'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      lowerPartialVarianceDS = self._computeLowerPartialVariance(dataSet,medianSet,pbWeight=relWeight,dim=self.sampleTag)

      calculations[metric] = lowerPartialVarianceDS
    #
    # lowerPartialSigma
    #
    metric = 'lowerPartialSigma'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      lpsDS = self._computePower(0.5,calculations['lowerPartialVariance'][list(needed[metric]['targets'])])
      calculations[metric] = lpsDS
    #
    # higherPartialVariance
    #
    metric = 'higherPartialVariance'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      medianSet = calculations['median'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      higherPartialVarianceDS = self._computeHigherPartialVariance(dataSet,medianSet,pbWeight=relWeight,dim=self.sampleTag)

      calculations[metric] = lowerPartialVarianceDS
    #
    # higherPartialSigma
    #
    metric = 'higherPartialSigma'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      hpsDS = self._computePower(0.5,calculations['higherPartialVariance'][list(needed[metric]['targets'])])
      calculations[metric] = hpsDS

    ############################################################
    # Begin Standard Error Calculations
    #
    # Reference for standard error calculations (including percentile):
    # B. Harding, C. Tremblay and D. Cousineau, "Standard errors: A review and evaluation of
    # standard error estimators using Monte Carlo simulations", The Quantitative Methods of
    # Psychology, Vol. 10, No. 2 (2014)
    ############################################################
    metric = 'expectedValue'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting calculate standard error on"'+metric+'"...')
      if self.pbPresent:
        factor = self._computePower(0.5,calculations['equivalentSamples'])
      else:
        factor = np.sqrt(self.sampleSize)
      calculations[metric+'_ste'] = calculations['sigma'][list(needed[metric]['targets'])]/factor

    metric = 'variance'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting calculate standard error on "'+metric+'"...')
      varList = list(needed[metric]['targets'])
      if self.pbPresent:
        en = calculations['equivalentSamples'][varList]
        factor = 2.0 /(en - 1.0)
        factor = self._computePower(0.5,factor)
      else:
        factor = np.sqrt(2.0/(float(self.sampleSize) - 1.0))
      calculations[metric+'_ste'] = calculations['sigma'][varList]**2 * factor

    metric = 'sigma'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting calculate standard error on "'+metric+'"...')
      varList = list(needed[metric]['targets'])
      if self.pbPresent:
        en = calculations['equivalentSamples'][varList]
        factor = 2.0 * (en - 1.0)
        factor = self._computePower(0.5,factor)
      else:
        factor = np.sqrt(2.0 * (float(self.sampleSize) - 1.0))
      calculations[metric+'_ste'] = calculations['sigma'][varList] / factor

    metric = 'median'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting calculate standard error on "'+metric+'"...')
      varList = list(needed[metric]['targets'])
      calculations[metric+'_ste'] = calculations['expectedValue_ste'][varList] * np.sqrt(np.pi/2.0)

    metric = 'skewness'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting calculate standard error on "'+metric+'"...')
      varList = list(needed[metric]['targets'])
      if self.pbPresent:
        en = calculations['equivalentSamples'][varList]
        factor = 6.*en*(en-1.)/((en-2.)*(en+1.)*(en+3.))
        factor = self._computePower(0.5,factor)
        calculations[metric+'_ste'] = xr.full_like(calculations[metric],1.0) * factor
      else:
        en = float(self.sampleSize)
        factor = np.sqrt(6.*en*(en-1.)/((en-2.)*(en+1.)*(en+3.)))
        calculations[metric+'_ste'] = xr.full_like(calculations[metric],factor)

    metric = 'kurtosis'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting calculate standard error on "'+metric+'"...')
      varList = list(needed[metric]['targets'])
      if self.pbPresent:
        en = calculations['equivalentSamples'][varList]
        factor1 = self._computePower(0.5,6.*en*(en-1.)/((en-2.)*(en+1.)*(en+3.)))
        factor2 = self._computePower(0.5,(en**2-1.)/((en-3.0)*(en+5.0)))
        factor = 2.0 * factor1 * factor2
        calculations[metric+'_ste'] = xr.full_like(calculations[metric],1.0) * factor
      else:
        en = float(self.sampleSize)
        factor = 2.0 * np.sqrt(6.*en*(en-1.)/((en-2.)*(en+1.)*(en+3.)))*np.sqrt((en**2-1.)/((en-3.0)*(en+5.0)))
        calculations[metric+'_ste'] = xr.full_like(calculations[metric],factor)
    ############################################################
    # End of Standard Error Calculations
    ############################################################
    #
    # maximum
    #
    metric = 'maximum'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      calculations[metric] = dataSet.max(dim=self.sampleTag)
    #
    # minimum
    #
    metric = 'minimum'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      calculations[metric] = dataSet.min(dim=self.sampleTag)
    #
    # percentile, this metric is handled differently
    #
    metric = 'percentile'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      percent = list(needed[metric]['percent'])
      # are there probability weights associated with the data?
      if self.pbPresent:
        relWeight = pbWeights[list(needed[metric]['targets'])]
        # if all weights are the same, calculate percentile with xarray, no need for _computeWeightedPercentile
        allSameWeight = True
        for target in needed[metric]['targets']:
          targWeight = relWeight[target].values
          if targWeight.min() != targWeight.max():
            allSameWeight = False
        if allSameWeight:
          # all weights are the same, percentile can be calculated with xarray.DataSet
          percentileSet = dataSet.quantile(percent,dim=self.sampleTag,interpolation=needed[metric]['interpolation'])
          percentileSet = percentileSet.rename({'quantile': 'percent'})
        else:
          # probability weights are not all the same
          # xarray does not have capability to calculate weighted quantiles at present
          # implement our own solution
          percentileSet = xr.Dataset()
          for target in needed[metric]['targets']:
            targWeight = relWeight[target].values
            targDa = dataSet[target]
            if self.pivotParameter in targDa.sizes.keys():
              quantile = []
              for label, group in targDa.groupby(self.pivotParameter):
                qtl = self._computeWeightedPercentile(group.values, targWeight, needed[metric]['interpolation'], percent=percent)
                quantile.append(qtl)
              da = xr.DataArray(quantile, dims=(self.pivotParameter, 'percent'), coords={'percent': percent, self.pivotParameter: self.pivotValue})
            else:
              quantile = self._computeWeightedPercentile(targDa.values, targWeight, needed[metric]['interpolation'], percent=percent)
              da = xr.DataArray(quantile, dims=('percent'), coords={'percent': percent})

            percentileSet[target] = da

        # TODO: remove when complete
        # interpolation: {'linear', 'lower', 'higher','midpoint','nearest'}, do not try to use 'linear' or 'midpoint'
        # The xarray.Dataset.where() will not return the corrrect solution
        # 'lower' is used for consistent
        # using xarray.Dataset.sel(**{'quantile':reqPercent}) to retrieve the quantile values
        #dataSetWeighted = dataSet * relWeight
        #percentileSet = dataSet.where(dataSetWeighted==dataSetWeighted.quantile(percent,dim=self.sampleTag,interpolation='lower')).mean(self.sampleTag)
      else:
        percentileSet = dataSet.quantile(percent,dim=self.sampleTag,interpolation=needed[metric]['interpolation'])
        percentileSet = percentileSet.rename({'quantile':'percent'})
      calculations[metric] = percentileSet

      # because percentile is different, calculate standard error here
      # standard error calculation uses the standard normal formulation for speed
      self.raiseADebug('Starting calculate standard error on "'+metric+'"...')
      norm = stats.norm
      factor = np.sqrt(np.asarray(percent)*(1.0 - np.asarray(percent)))/norm.pdf(norm.ppf(percent))
      sigmaAdjusted = calculations['sigma'][list(needed[metric]['targets'])]/np.sqrt(calculations['equivalentSamples'][list(needed[metric]['targets'])])
      sigmaAdjusted = sigmaAdjusted.expand_dims(dim={'percent': percent})
      factor = xr.DataArray(data=factor, dims='percent', coords={'percent': percent})
      calculations[metric + '_ste'] = sigmaAdjusted*factor

      # # TODO: this is the KDE method, it is a more accurate method of calculating standard error
      # # for percentile, but the computation time is too long. IF this computation can be sped up,
      # # implement it here:
      # percentileSteSet = xr.Dataset()
      # calculatedPercentiles = calculations[metric]
      # relWeight = pbWeights[list(needed[metric]['targets'])]
      # for target in needed[metric]['targets']:
      #   targWeight = relWeight[target].values
      #   en = calculations['equivalentSamples'][target].values
      #   targDa = dataSet[target]
      #   if self.pivotParameter in targDa.sizes.keys():
      #     percentileSte = np.zeros((len(self.pivotValue), len(percent))) # array
      #     for i, (label, group) in enumerate(targDa.groupby(self.pivotParameter)): # array
      #       if group.values.min() == group.values.max():
      #         subPercentileSte = np.array([0.0]*len(percent))
      #       else:
      #         # get KDE
      #         kde = stats.gaussian_kde(group.values, weights=targWeight)
      #         vals = calculatedPercentiles[target].sel(**{'percent': percent, self.pivotParameter: label}).values
      #         factor = np.sqrt(np.asarray(percent)*(1.0 - np.asarray(percent))/en)
      #         subPercentileSte = factor/kde(vals)
      #       percentileSte[i, :] = subPercentileSte
      #     da = xr.DataArray(percentileSte, dims=(self.pivotParameter, 'percent'), coords={self.pivotParameter: self.pivotValue, 'percent': percent})
      #     percentileSteSet[target] = da
      #   else:
      #     calcPercentiles = calculatedPercentiles[target]
      #     if targDa.values.min() == targDa.values.max():
      #       # distribution is a delta function, so no KDE construction
      #       percentileSte = list(np.zeros(calcPercentiles.shape))
      #     else:
      #       # get KDE
      #       kde = stats.gaussian_kde(targDa.values, weights=targWeight)
      #       factor = np.sqrt(np.array(percent)*(1.0 - np.array(percent))/en)
      #       percentileSte = list(factor/kde(calcPercentiles.values))
      #     da = xr.DataArray(percentileSte, dims=('percent'), coords={'percent': percent})
      #     percentileSteSet[target] = da
      # calculations[metric+'_ste'] = percentileSteSet

    def startVector(metric):
      """
        Common method among all metrics for establishing parameters
        @ In, metric, string, the name of the statistics metric to calculate
        @ Out, targets, list(str), list of target parameter names (evaluate metrics for these)
        @ Out, features, list(str), list of feature parameter names (evaluate with respect to these)
        @ Out, skip, bool, if True it means either features or parameters were missing, so don't calculate anything
      """
      # default to skipping, change that if we find criteria
      targets = []
      features = []
      skip = True
      if len(needed[metric]['targets'])>0:
        self.raiseADebug('Starting "'+metric+'"...')
        targets = list(needed[metric]['targets'])
        features = list(needed[metric]['features'])
        skip = False #True only if we don't have targets and features
      if skip:
        if metric not in self.skipped.keys():
          self.skipped[metric] = True
      return targets,features,skip

    #################
    # VECTOR VALUES #
    #################
    #
    # sensitivity matrix
    #
    metric = 'sensitivity'
    targets,features,skip = startVector(metric)
    #NOTE sklearn expects the transpose of what we usually do in RAVEN, so #samples by #features
    if not skip:
      #for sensitivity matrix, we don't use numpy/scipy methods to calculate matrix operations,
      #so we loop over targets and features
      params = list(set(targets).union(set(features)))
      dataSet = inputDataset[params]
      relWeight = pbWeights[params] if self.pbPresent else None
      intersectionSet = set(targets) & set(features)
      if self.pivotParameter in dataSet.sizes.keys():
        dataSet = dataSet.to_array().transpose(self.pivotParameter,self.sampleTag,'variable')
        featSet = dataSet.sel(**{'variable':features}).values
        targSet = dataSet.sel(**{'variable':targets}).values
        pivotVals = dataSet.coords[self.pivotParameter].values
        da = None
        for i in range(len(pivotVals)):
          ds = self.sensitivityCalculation(features,targets,featSet[i,:,:],targSet[i,:,:],intersectionSet)
          da = ds if da is None else xr.concat([da,ds], dim=self.pivotParameter)
        da.coords[self.pivotParameter] = pivotVals
      else:
        # construct target and feature matrices
        dataSet = dataSet.to_array().transpose(self.sampleTag,'variable')
        featSet = dataSet.sel(**{'variable':features}).values
        targSet = dataSet.sel(**{'variable':targets}).values
        da = self.sensitivityCalculation(features,targets,featSet,targSet,intersectionSet)
      calculations[metric] = da
    #
    # covariance matrix
    #
    metric = 'covariance'
    targets,features,skip = startVector(metric)
    if not skip:
      # because the C implementation is much faster than picking out individual values,
      #   we do the full covariance matrix with all the targets and features.
      # FIXME adding an alternative for users to choose pick OR do all, defaulting to something smart
      #   dependent on the percentage of the full matrix desired, would be better.
      # IF this is fixed, make sure all the features and targets are requested for all the metrics
      #   dependent on this metric
      params = list(set(targets).union(set(features)))
      dataSet = inputDataset[params]
      relWeight = pbWeights[params] if self.pbPresent else None
      if self.pbPresent:
        fact = (self.__computeUnbiasedCorrection(2, self.realizationWeight)).to_array().values if not self.biased else 1.0
        meanSet = (dataSet * relWeight).sum(dim = self.sampleTag)
      else:
        meanSet = dataSet.mean(dim = self.sampleTag)
        fact = 1.0 / (float(dataSet.sizes[self.sampleTag]) - 1.0) if not self.biased else 1.0 / float(dataSet.sizes[self.sampleTag])
      targVars = list(dataSet.data_vars)
      varianceSet = self._computeVariance(dataSet,meanSet,pbWeight=relWeight,dim=self.sampleTag)
      dataSet = dataSet - meanSet
      if self.pivotParameter in dataSet.sizes.keys():
        ds = None
        paramDA = dataSet.to_array().transpose(self.pivotParameter,'variable',self.sampleTag).values
        varianceDA = varianceSet[targVars].to_array().transpose(self.pivotParameter,'variable').values
        pivotVals = dataSet.coords[self.pivotParameter].values
        for i in range(len(pivotVals)):
          # construct target and feature matrices
          paramSamples = paramDA[i,...]
          da = self.covarianceCalculation(paramDA[i,...],fact,varianceDA[i,:],targVars)
          ds = da if ds is None else xr.concat([ds,da], dim=self.pivotParameter)
        ds.coords[self.pivotParameter] = pivotVals
        calculations[metric] = ds
      else:
        # construct target and feature matrices
        paramSamples = dataSet.to_array().transpose('variable',self.sampleTag).values
        varianceDA = varianceSet[targVars].to_array().values
        da = self.covarianceCalculation(paramSamples,fact,varianceDA,targVars)
        calculations[metric] = da

    def getCovarianceSubset(desired):
      """
        @ In, desired, list(str), list of parameters to extract from covariance matrix
        @ Out, reducedCov, xarray.DataArray, reduced covariance matrix
      """
      if self.pivotParameter in desired:
        self.raiseAnError(RuntimeError, 'The pivotParameter "{}" is among the parameters requested for performing statistics. Please remove!'.format(self.pivotParameter))
      reducedCov = calculations['covariance'].sel(**{'targets':desired,'features':desired})
      return reducedCov
    #
    # pearson matrix
    #
    # see comments in covariance for notes on C implementation
    metric = 'pearson'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar = getCovarianceSubset(params)
      targCoords = reducedCovar.coords['targets'].values
      if self.pivotParameter in reducedCovar.sizes.keys():
        pivotCoords = reducedCovar.coords[self.pivotParameter].values
        ds = None
        for label, group in reducedCovar.groupby(self.pivotParameter):
          corrMatrix = self.corrCoeff(group.values)
          da = xr.DataArray(corrMatrix, dims=('targets','features'), coords={'targets':targCoords,'features':targCoords})
          ds = da if ds is None else xr.concat([ds,da], dim=self.pivotParameter)
        ds.coords[self.pivotParameter] = pivotCoords
        calculations[metric] = ds
      else:
        corrMatrix = self.corrCoeff(reducedCovar.values)
        da = xr.DataArray(corrMatrix, dims=('targets','features'), coords={'targets':targCoords,'features':targCoords})
        calculations[metric] = da
    #
    # spearman matrix
    #
    # see RAVEN theory manual for a detailed explaination
    # of the formulation used here
    #
    metric = 'spearman'
    targets,features,skip = startVector(metric)
    #NOTE sklearn expects the transpose of what we usually do in RAVEN, so #samples by #features
    if not skip:
      #for spearman matrix, we don't use numpy/scipy methods to calculate matrix operations,
      #so we loop over targets and features
      params = list(set(targets).union(set(features)))
      dataSet = inputDataset[params]
      relWeight = pbWeights[params] if self.pbPresent else None
      if self.pivotParameter in dataSet.sizes.keys():
        dataSet = dataSet.to_array().transpose(self.pivotParameter,self.sampleTag,'variable')
        featSet = dataSet.sel(**{'variable':features}).values
        targSet = dataSet.sel(**{'variable':targets}).values
        pivotVals = dataSet.coords[self.pivotParameter].values
        da = None
        for i in range(len(pivotVals)):
          ds = self.spearmanCorrelation(features,targets,featSet[i,:,:],targSet[i,:,:],relWeight)
          da = ds if da is None else xr.concat([da,ds], dim=self.pivotParameter)
        da.coords[self.pivotParameter] = pivotVals
      else:
        # construct target and feature matrices
        dataSet = dataSet.to_array().transpose(self.sampleTag,'variable')
        featSet = dataSet.sel(**{'variable':features}).values
        targSet = dataSet.sel(**{'variable':targets}).values
        da = self.spearmanCorrelation(features,targets,featSet,targSet,relWeight)
      calculations[metric] = da
    #
    # VarianceDependentSensitivity matrix
    # The formula for this calculation is coming from: http://www.math.uah.edu/stat/expect/Matrices.html
    # The best linear predictor: L(Y|X) = expectedValue(Y) + cov(Y,X) * [vc(X)]^(-1) * [X-expectedValue(X)]
    # where Y is a vector of outputs, and X is a vector of inputs, cov(Y,X) is the covariance matrix of Y and X,
    # vc(X) is the covariance matrix of X with itself.
    # The variance dependent sensitivity matrix is defined as: cov(Y,X) * [vc(X)]^(-1)
    metric = 'VarianceDependentSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar = getCovarianceSubset(params)
      targCoords = reducedCovar.coords['targets'].values
      if self.pivotParameter in reducedCovar.sizes.keys():
        pivotCoords = reducedCovar.coords[self.pivotParameter].values
        ds = None
        for label, group in reducedCovar.groupby(self.pivotParameter):
          da = self.varianceDepSenCalculation(targCoords,group.values)
          ds = da if ds is None else xr.concat([ds,da], dim=self.pivotParameter)
        ds.coords[self.pivotParameter] = pivotCoords
        calculations[metric] = ds
      else:
        da = self.varianceDepSenCalculation(targCoords,reducedCovar.values)
        calculations[metric] = da

    #
    # Normalized variance dependent sensitivity matrix
    # variance dependent sensitivity  normalized by the mean (% change of output)/(% change of input)
    #
    metric = 'NormalizedSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedSen = calculations['VarianceDependentSensitivity'].sel(**{'targets':params,'features':params})
      meanDA = calculations['expectedValue'][params].to_array()
      meanDA = meanDA.rename({'variable':'targets'})
      reducedSen /= meanDA
      meanDA = meanDA.rename({'targets':'features'})
      reducedSen *= meanDA
      calculations[metric] = reducedSen

    for metric, ds in calculations.items():
      if metric in self.scalarVals + self.steVals +['equivalentSamples'] and metric !='samples':
        calculations[metric] = ds.to_array().rename({'variable':'targets'})
    # in here we fill the NaN with "nan". In this way, we are sure that even if
    # there might be NaN in any raw for a certain timestep we do not drop the variable
    # In the past, in a condition such as:
    # time, A, B, C
    #    0, 1, NaN, 1
    #    1, 1, 0.5, 1
    #    2, 1, 2.0, 2
    # the variable B would have been dropped (in the printing stage)
    # with this modification, this should not happen anymore
    outputSet = xr.Dataset(data_vars=calculations).fillna("nan")

    if self.outputDataset:
      # Add 'RAVEN_sample_ID' to output dataset for consistence
      if 'RAVEN_sample_ID' not in outputSet.sizes.keys():
        outputSet = outputSet.expand_dims('RAVEN_sample_ID')
        outputSet['RAVEN_sample_ID'] = [0]
      return outputSet
    else:
      outputDict = {}
      for metric, requestList  in self.toDo.items():
        for targetDict in requestList:
          prefix = targetDict['prefix'].strip()
          for target in targetDict['targets']:
            if metric in self.scalarVals and metric != 'percentile':
              varName = prefix + '_' + target
              outputDict[varName] = np.atleast_1d(outputSet[metric].sel(**{'targets':target}))
              steMetric = metric + '_ste'
              if steMetric in self.steVals:
                metaVar = prefix + '_ste_' + target
                outputDict[metaVar] = np.atleast_1d(outputSet[steMetric].sel(**{'targets':target}))
            elif metric == 'percentile':
              for percent in targetDict['strPercent']:
                varName = '_'.join([prefix,percent,target])
                percentVal = float(percent)/100.
                outputDict[varName] = np.atleast_1d(outputSet[metric].sel(**{'targets':target,'percent':percentVal}))
                steMetric = metric + '_ste'
                if steMetric in self.steVals:
                  metaVar = '_'.join([prefix,percent,'ste',target])
                  outputDict[metaVar] = np.atleast_1d(outputSet[steMetric].sel(**{'targets':target,'percent':percentVal}))
            else:
              #check if it was skipped for some reason
              skip = self.skipped.get(metric, None)
              if skip is not None:
                self.raiseADebug('Metric',metric,'was skipped for parameters',targetDict,'!  See warnings for details.  Ignoring...')
                continue
              if metric in self.vectorVals:
                for feature in targetDict['features']:
                  varName = '_'.join([prefix,target,feature])
                  outputDict[varName] = np.atleast_1d(outputSet[metric].sel(**{'targets':target,'features':feature}))
      if self.pivotParameter in outputSet.sizes.keys():
        outputDict[self.pivotParameter] = np.atleast_1d(self.pivotValue)

      return outputDict

  def corrCoeff(self, covM):
    """
      This method calculates the correlation coefficient Matrix (pearson) for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calcuated depending on the selection of the inputs.
      @ In,  covM, numpy.array, [#targets,#targets] covariance matrix
      @ Out, covM, numpy.array, [#targets,#targets] correlation matrix
    """
    try:
      d = np.diag(covM)
    except ValueError:
      # scalar covariance
      # nan if incorrect value (nan, inf, 0), 1 otherwise
      return covM / covM
    stdDev = np.sqrt(d)
    covM /= stdDev[:,None]
    covM /= stdDev[None,:]
    return covM

  def sensitivityCalculation(self,featVars, targVars, featSamples, targSamples, intersectionSet):
    """
      This method computes the sensitivity coefficients based on the SciKitLearn LinearRegression method
      @ In, featVars, list, list of feature variables
      @ In, targVars, list, list of target variables
      @ In, featSamples, numpy.ndarray, [#samples, #features] array of features
      @ In, targSamples, numpy.ndarray, [#samples, #targets] array of targets
      @ In, intersectionSet, boolean, True if some target variables are in the list of features
      @ Out, da, xarray.DataArray, contains the calculations of sensitivity coefficients
    """
    from sklearn.linear_model import LinearRegression
    if self.multipleFeatures:
      # intersectionSet is flag that used to check the relationship between the features and targets.
      # If True, part of the target variables are listed in teh feature set, then multivariate linear
      # regression should not be used, and a loop over the target set is required.
      # If False, which means there is no overlap between the target set and feature set.
      # mutivariate linear regression can be used. However, for both cases, co-linearity check should be
      # added for the feature set. ~ wangc

      if not intersectionSet:
        condNumber = np.linalg.cond(featSamples)
        if condNumber > 30.:
          self.raiseAWarning("Condition Number: {:10.4f} > 30.0. Detected SEVERE multicollinearity problem. Sensitivity might be incorrect!".format(condNumber))
        senMatrix = LinearRegression().fit(featSamples,targSamples).coef_
      else:
        # Target variables are in feature variables list, multi-target linear regression can not be used
        # Since the 'multi-colinearity' exists, we need to loop over target variables
        # TODO: Some general methods need to be implemented in order to handle the 'multi-colinearity' -- wangc
        senMatrix = np.zeros((len(targVars), len(featVars)))
        for p, targ in enumerate(targVars):
          ind = list(featVars).index(targ) if targ in featVars else None
          if ind is not None:
            featMat = np.delete(featSamples,ind,axis=1)
          else:
            featMat = featSamples
          regCoeff = LinearRegression().fit(featMat, targSamples[:,p]).coef_
          condNumber = np.linalg.cond(featMat)
          if condNumber > 30.:
            self.raiseAWarning("Condition Number: {:10.4f} > 30.0. Detected SEVERE multicollinearity problem. Sensitivity might be incorrect!".format(condNumber))
          if ind is not None:
            regCoeff = np.insert(regCoeff,ind,1.0)
          senMatrix[p,:] = regCoeff
    else:
      senMatrix = np.zeros((len(targVars), len(featVars)))
      for p, feat in enumerate(featVars):
        regCoeff = LinearRegression().fit(featSamples[:,p].reshape(-1,1),targSamples).coef_
        senMatrix[:,p] = regCoeff[:,0]
    da = xr.DataArray(senMatrix, dims=('targets','features'), coords={'targets':targVars,'features':featVars})

    return da

  def covarianceCalculation(self,paramSamples,fact,variance,targVars):
    """
      This method computes the covariance of given sample matrix
      @ In, paramSamples, numpy.ndarray, [#parameters, #samples], array of parameters
      @ In, fact, float, the unbiase correction factor
      @ In, variance, numpy.ndarray, [#parameters], variance of parameters
      @ In, targVars, list, the list of parameters
      @ Out, da, xarray.DataArray, contains the calculations of covariance
    """
    if self.pbPresent:
      paramSamplesT = (paramSamples*self.realizationWeight['ProbabilityWeight'].values).T
    else:
      paramSamplesT = paramSamples.T
    cov = np.dot(paramSamples, paramSamplesT.conj())
    cov *= fact
    np.fill_diagonal(cov,variance)
    da = xr.DataArray(cov, dims=('targets','features'), coords={'targets':targVars,'features':targVars})
    return da

  def varianceDepSenCalculation(self,targCoords, cov):
    """
      This method computes the covariance of given sample matrix
      @ In, targCoords, list, the list of parameters
      @ In, cov, numpy.ndarray, the covariance of parameters
      @ Out, da, xarray.DataArray, contains the calculations of variance dependent sensitivities
    """
    senMatrix = np.zeros((len(targCoords), len(targCoords)))
    if self.multipleFeatures:
      for p, param in enumerate(targCoords):
        covX = np.delete(cov,p,axis=0)
        covX = np.delete(covX,p,axis=1)
        covYX = np.delete(cov[p,:],p)
        sensCoef = np.dot(covYX,np.linalg.pinv(covX))
        sensCoef = np.insert(sensCoef,p,1.0)
        senMatrix[p,:] = sensCoef
    else:
      for p, param in enumerate(targCoords):
        covX = cov[p,p]
        covYX = cov[:,p]
        sensCoef = covYX / covX
        senMatrix[:,p] = sensCoef
    da = xr.DataArray(senMatrix, dims=('targets','features'), coords={'targets':targCoords,'features':targCoords})
    return da

  def spearmanCorrelation(self, featVars, targVars, featSamples, targSamples, pbWeights):
    """
      This method computes the spearman correlation coefficients
      @ In, featVars, list, list of feature variables
      @ In, targVars, list, list of target variables
      @ In, featSamples, numpy.ndarray, [#samples, #features] array of features
      @ In, targSamples, numpy.ndarray, [#samples, #targets] array of targets
      @ In, pbWeights, dataset, probability weights
      @ Out, da, xarray.DataArray, contains the calculations of spearman coefficients
    """
    spearmanMat = np.zeros((len(targVars), len(featVars)))
    wf, wt = None, None
    # compute unbiased factor
    if self.pbPresent:
      fact = (self.__computeUnbiasedCorrection(2, self.realizationWeight)).to_array().values if not self.biased else 1.0
      vp = self.__computeVp(1,self.realizationWeight)['ProbabilityWeight'].values
      varianceFactor = fact*(1.0/vp)
    else:
      fact = 1.0 / (float(featSamples.shape[0]) - 1.0) if not self.biased else 1.0 / float(featSamples.shape[0])
      varianceFactor = fact

    for tidx, target in enumerate(targVars):
      for fidx, feat in enumerate(featVars):
        if self.pbPresent:
          wf, wt = np.asarray(pbWeights[feat]), np.asarray(pbWeights[target])
        rankFeature, rankTarget = mathUtils.rankData(featSamples[:,fidx],wf),  mathUtils.rankData(targSamples[:,tidx],wt)
        # compute covariance of the ranked features
        cov  = np.cov(rankFeature, y=rankTarget, aweights=wt)
        covF = np.cov(rankFeature,y=rankFeature, aweights=wf)
        covT = np.cov(rankTarget,y=rankTarget, aweights=wf)
        # apply correction factor (for biased or unbiased) (off diagonal)
        cov[~np.eye(2,dtype=bool)] *= fact
        covF[~np.eye(2,dtype=bool)] *= fact
        covT[~np.eye(2,dtype=bool)] *= fact
        # apply correction factor (for biased or unbiased) (diagonal)
        cov[np.eye(2,dtype=bool)] *= varianceFactor
        covF[~np.eye(2,dtype=bool)] *= varianceFactor
        covT[~np.eye(2,dtype=bool)] *= varianceFactor
        # now we can compute the pearson of such pairs
        spearman = (cov / np.sqrt(covF * covT))[-1,0]
        spearmanMat[tidx,fidx] = spearman

    da = xr.DataArray(spearmanMat, dims=('targets','features'), coords={'targets':targVars,'features':featVars})
    return da

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputSet, xarray.Dataset or dictionary, dataset or dictionary containing the results
    """
    inputData = self.inputToInternal(inputIn)
    outputSet = self.__runLocal(inputData)
    return outputSet

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    super().collectOutput(finishedJob, output)
