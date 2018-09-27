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

@author: alfoa, wangc
"""
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules---------------------------------------------------------------
import numpy as np
import os
import copy
from collections import OrderedDict, defaultdict
from sklearn.linear_model import LinearRegression
import six
import xarray as xr
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData
from utils import mathUtils
import Files
import Runners
#Internal Modules End-----------------------------------------------------------

class BasicStatistics(PostProcessor):
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
                'samples']
  vectorVals = ['sensitivity',
                'covariance',
                'pearson',
                'NormalizedSensitivity',
                'VarianceDependentSensitivity']

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
    inputSpecification = super(BasicStatistics, cls).getInputSpecification()

    for scalar in cls.scalarVals:
      scalarSpecification = InputData.parameterInputFactory(scalar, contentType=InputData.StringListType)
      if scalar == 'percentile':
        #percent is a string type because otherwise we can't tell 95.0 from 95
        # which matters because the number is used in output.
        scalarSpecification.addParam("percent", InputData.StringListType)
      scalarSpecification.addParam("prefix", InputData.StringType)
      inputSpecification.addSub(scalarSpecification)

    for vector in cls.vectorVals:
      vectorSpecification = InputData.parameterInputFactory(vector)
      vectorSpecification.addParam("prefix", InputData.StringType)
      features = InputData.parameterInputFactory('features',
                                contentType=InputData.StringListType)
      vectorSpecification.addSub(features)
      targets = InputData.parameterInputFactory('targets',
                                contentType=InputData.StringListType)
      vectorSpecification.addSub(targets)
      inputSpecification.addSub(vectorSpecification)

    pivotParameterInput = InputData.parameterInputFactory('pivotParameter', contentType=InputData.StringType)
    inputSpecification.addSub(pivotParameterInput)

    datasetInput = InputData.parameterInputFactory('dataset', contentType=InputData.StringType)
    inputSpecification.addSub(datasetInput)

    methodsToRunInput = InputData.parameterInputFactory("methodsToRun", contentType=InputData.StringType)
    inputSpecification.addSub(methodsToRunInput)

    biasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType)
    inputSpecification.addSub(biasedInput)

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.parameters = {}  # parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.acceptedCalcParam = self.scalarVals + self.vectorVals
    self.what = self.acceptedCalcParam  # what needs to be computed... default...all
    self.methodsToRun = []  # if a function is present, its outcome name is here stored... if it matches one of the known outcomes, the pp is going to use the function to compute it
    self.externalFunction = []
    self.printTag = 'POSTPROCESSOR BASIC STATISTIC'
    self.addAssemblerObject('Function','-1', True)
    self.biased = False # biased statistics?
    self.pivotParameter = None # time-dependent statistics pivot parameter
    self.pivotValue = None # time-dependent statistics pivot parameter values
    self.dynamic        = False # is it time-dependent?
    self.sampleTag      = None  # Tag used to track samples
    self.pbPresent      = False # True if the ProbabilityWeight is available
    self.realizationWeight = None # The joint probabilities
    self.outputDataset  = False # True if the user wants to dump the outputs to dataset

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
        self.sampleTag = inputDataset.sizes.keys()[0]
      return inputDataset, pbWeights

    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'BasicStatistics postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)

    # extract all required data from input DataObjects, an input dataset is constructed
    dataSet = currentInput.asDataset()
    inputDataset = dataSet[self.parameters['targets']]
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
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    self.toDo = {}
    for child in paramInput.subparts:
      tag = child.getName()
      #because percentile is strange (has an attached parameter), we address it first
      if tag in ['percentile'] + self.scalarVals + self.vectorVals:
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
          self.toDo[tag] = [] # list of {'targets':(), 'prefix':str, 'percent':str}
        if 'percent' not in child.parameterValues:
          reqPercent = [0.05, 0.95]
          strPercent = ['5','95']
        else:
          reqPercent = set(utils.floatConversion(percent)/100. for percent in child.parameterValues['percent'])
          strPercent = set(percent for percent in child.parameterValues['percent'])
        self.toDo[tag].append({'targets':set(targets),
                               'prefix':prefix,
                               'percent':reqPercent,
                               'strPercent':strPercent})
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
        if child.value.lower() in utils.stringsThatMeanTrue():
          self.biased = True
      elif tag == "pivotParameter":
        self.pivotParameter = child.value
      elif tag == "dataset":
        if child.value.lower() in utils.stringsThatMeanTrue():
          self.outputDataset = True
      else:
        self.raiseAWarning('Unrecognized node in BasicStatistics "',tag,'" has been ignored!')
    assert (len(self.toDo)>0), self.raiseAnError(IOError, 'BasicStatistics needs parameters to work on! Please check input for PP: ' + self.name)

  def __computePower(self, p, dataset):
    """
      Compute the p-th power of weights
      @ In, p, int, the power
      @ In, dataset, xarray.Dataset, probability weights of all input variables
      @ Out, pw, xarray.Dataset, the p-th power of weights
    """
    pw = xr.Dataset()
    for target, targValue in dataset.data_vars.items():
      pw[target] = np.power(targValue,p)
    return pw

  def __computeVp(self,p,weights):
    """
      Compute the sum of p-th power of weights
      @ In, p, int, the power
      @ In, weights, xarray.Dataset, probability weights of all input variables
      @ Out, vp, xarray.Dataset, the sum of p-th power of weights
    """
    vp = self.__computePower(p,weights)
    vp = vp.sum()
    return vp

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
    vr = self.__computePower(2.0, variance)
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
      unbiasCorr = self.__computeUnbiasedCorrection(4,arrayIn.sizes[dim]) if not self.biased else 1.0
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
    vr = self.__computePower(1.5, variance)
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(3,pbWeight) if not self.biased else 1.0
      vp = 1.0/self.__computeVp(1,pbWeight)
      result = ((arrayIn - expValue)**3 * pbWeight).sum(dim=dim) * vp * unbiasCorr / vr
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(3,arrayIn.sizes[dim]) if not self.biased else 1.0
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
      unbiasCorr = self.__computeUnbiasedCorrection(2,arrayIn.sizes[dim]) if not self.biased else 1.0
      result =  (arrayIn-expValue).var(dim=dim) * unbiasCorr
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

  def _computeWeightedPercentile(self,arrayIn,pbWeight,percent=0.5):
    """
      Method to compute the weighted percentile in a array of data
      @ In, arrayIn, list/numpy.array, the array of values from which the percentile needs to be estimated
      @ In, pbWeight, list/numpy.array, the reliability weights that correspond to the values in 'array'
      @ In, percent, float, the percentile that needs to be computed (between 0.01 and 1.0)
      @ Out, result, float, the percentile
    """
    idxs                   = np.argsort(np.asarray(zip(pbWeight,arrayIn))[:,1])
    # Inserting [0.0,arrayIn[idxs[0]]] is needed when few samples are generated and
    # a percentile that is < that the first pb weight is requested. Otherwise the median
    # is returned (that is wrong).
    sortedWeightsAndPoints = np.insert(np.asarray(zip(pbWeight[idxs],arrayIn[idxs])),0,[0.0,arrayIn[idxs[0]]],axis=0)
    weightsCDF             = np.cumsum(sortedWeightsAndPoints[:,0])
    try:
      index = utils.find_le_index(weightsCDF,percent)
      result = sortedWeightsAndPoints[index,1]
    except ValueError:
      result = np.percentile(arrayIn,percent,interpolation='lower')
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
    needed = dict((metric,{'targets':set(),'percent':set()}) for metric in self.scalarVals)
    needed.update(dict((metric,{'targets':set(),'features':set()}) for metric in self.vectorVals))
    for metric, params in self.toDo.items():
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

    # variable                     | needs                  | needed for
    # --------------------------------------------------------------------
    # skewness needs               | expectedValue,variance |
    # kurtosis needs               | expectedValue,variance |
    # median needs                 |                        |
    # percentile needs             |                        |
    # maximum needs                |                        |
    # minimum needs                |                        |
    # covariance needs             |                        | pearson,VarianceDependentSensitivity,NormalizedSensitivity
    # NormalizedSensitivity        | covariance,VarDepSens  |
    # VarianceDependentSensitivity | covariance             | NormalizedSensitivity
    # sensitivity needs            |                        |
    # pearson needs                | covariance             |
    # sigma needs                  | variance               | variationCoefficient
    # variance                     | expectedValue          | sigma, skewness, kurtosis
    # expectedValue                |                        | variance, variationCoefficient, skewness, kurtosis
    needed['sigma']['targets'].update(needed['variationCoefficient']['targets'])
    needed['variance']['targets'].update(needed['sigma']['targets'])
    needed['expectedValue']['targets'].update(needed['sigma']['targets'])
    needed['expectedValue']['targets'].update(needed['variationCoefficient']['targets'])
    needed['expectedValue']['targets'].update(needed['variance']['targets'])
    needed['expectedValue']['targets'].update(needed['skewness']['targets'])
    needed['expectedValue']['targets'].update(needed['kurtosis']['targets'])
    needed['expectedValue']['targets'].update(needed['NormalizedSensitivity']['targets'])
    needed['expectedValue']['targets'].update(needed['NormalizedSensitivity']['features'])
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
    metric = 'samples'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      numRlz = inputDataset.sizes[self.sampleTag]
      if self.dynamic:
        nt = inputDataset.sizes[self.pivotParameter]
        sampleMat = np.zeros((len(self.parameters['targets']),len(self.pivotValue)))
        sampleMat.fill(numRlz)
        samplesDA = xr.DataArray(sampleMat,dims=('targets',self.pivotParameter),coords={'targets':self.parameters['targets'],self.pivotParameter:self.pivotValue})
      else:
        sampleMat = np.zeros(len(self.parameters['targets']))
        sampleMat.fill(numRlz)
        samplesDA = xr.DataArray(sampleMat,dims=('targets'),coords={'targets':self.parameters['targets']})

      calculations[metric] = samplesDA
    #
    # expected value
    #
    metric = 'expectedValue'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      if self.pbPresent:
        relWeight = pbWeights[list(needed[metric]['targets'])]
        dataSet = dataSet * relWeight
        expectedValueDS = dataSet.sum(dim = self.sampleTag)
      else:
        expectedValueDS = dataSet.mean(dim = self.sampleTag)
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
      sigmaDS = self.__computePower(0.5,calculations['variance'][list(needed[metric]['targets'])])
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
        medianSet = xr.Dataset()
        relWeight = pbWeights[list(needed[metric]['targets'])]
        for target in needed[metric]['targets']:
          targWeight = relWeight[target].values
          targDa = dataSet[target]
          if self.pivotParameter in targDa.sizes.keys():
            quantile = [self._computeWeightedPercentile(group.values,targWeight,percent=0.5) for label,group in targDa.groupby(self.pivotParameter)]
          else:
            quantile = self._computeWeightedPercentile(targDa.values,targWeight,percent=0.5)
          if self.pivotParameter in targDa.sizes.keys():
            da = xr.DataArray(quantile,dims=(self.pivotParameter),coords={self.pivotParameter:self.pivotValue})
          else:
            da = xr.DataArray(quantile)
          medianSet[target] = da

        #TODO: remove when complete
        # interpolation: {'linear', 'lower', 'higher','midpoint','nearest'}, do not try to use 'linear' or 'midpoint'
        # The xarray.Dataset.where() will not return the corrrect solution
        # 'lower' is used for consistent
        #dataSetWeighted = dataSet * relWeight
        #medianSet = dataSet.where(dataSetWeighted==dataSetWeighted.quantile(0.5,dim=self.sampleTag,interpolation='lower')).sum(self.sampleTag)
      else:
        medianSet = dataSet.median(dim=self.sampleTag)
      calculations[metric] = medianSet
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
      if self.pbPresent:
        percentileSet = xr.Dataset()
        relWeight = pbWeights[list(needed[metric]['targets'])]
        for target in needed[metric]['targets']:
          targWeight = relWeight[target].values
          targDa = dataSet[target]
          quantile = []
          for pct in percent:
            if self.pivotParameter in targDa.sizes.keys():
              qtl = [self._computeWeightedPercentile(group.values,targWeight,percent=pct) for label,group in targDa.groupby(self.pivotParameter)]
            else:
              qtl = self._computeWeightedPercentile(targDa.values,targWeight,percent=pct)
            quantile.append(qtl)
          if self.pivotParameter in targDa.sizes.keys():
            da = xr.DataArray(quantile,dims=('percent',self.pivotParameter),coords={'percent':percent,self.pivotParameter:self.pivotValue})
          else:
            da = xr.DataArray(quantile,dims=('percent'),coords={'percent':percent})
          percentileSet[target] = da

        # TODO: remove when complete
        # interpolation: {'linear', 'lower', 'higher','midpoint','nearest'}, do not try to use 'linear' or 'midpoint'
        # The xarray.Dataset.where() will not return the corrrect solution
        # 'lower' is used for consistent
        # using xarray.Dataset.sel(**{'quantile':reqPercent}) to retrieve the quantile values
        #dataSetWeighted = dataSet * relWeight
        #percentileSet = dataSet.where(dataSetWeighted==dataSetWeighted.quantile(percent,dim=self.sampleTag,interpolation='lower')).mean(self.sampleTag)
      else:
        percentileSet = dataSet.quantile(percent,dim=self.sampleTag,interpolation='lower')
        percentileSet = percentileSet.rename({'quantile':'percent'})
      calculations[metric] = percentileSet

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
      if metric in self.scalarVals and metric !='samples':
        calculations[metric] = ds.to_array().rename({'variable':'targets'})
    outputSet = xr.Dataset(data_vars=calculations)

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
            elif metric == 'percentile':
              for percent in targetDict['strPercent']:
                varName = '_'.join([prefix,percent,target])
                percentVal = float(percent)/100.
                outputDict[varName] = np.atleast_1d(outputSet[metric].sel(**{'targets':target,'percent':percentVal}))
            else:
              #check if it was skipped for some reason
              skip = self.skipped.get(metric, None)
              if skip is not None:
                self.raiseADebug('Metric',metric,'was skipped for parameters',targetDict,'!  See warnings for details.  Ignoring...')
                continue
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
    if not intersectionSet:
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
        regCoeff = LinearRegression().fit(featMat, targSamples[:,p]).coef_
        if ind is not None:
          regCoeff = np.insert(regCoeff,p,1.0)
        senMatrix[p,:] = regCoeff
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
    for p,param in enumerate(targCoords):
      covX = np.delete(cov,p,axis=0)
      covX = np.delete(covX,p,axis=1)
      covYX = np.delete(cov[p,:],p)
      sensCoef = np.dot(covYX,np.linalg.pinv(covX))
      sensCoef = np.insert(sensCoef,p,1.0)
      senMatrix[p,:] = sensCoef
    da = xr.DataArray(senMatrix, dims=('targets','features'), coords={'targets':targCoords,'features':targCoords})
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
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")

    outputRealization = evaluation[1]
    if output.type in ['PointSet','HistorySet']:
      if self.outputDataset:
        self.raiseAnError(IOError, "DataSet output is required, but the provided type of DataObject is",output.type)
      self.raiseADebug('Dumping output in data object named ' + output.name)
      output.addRealization(outputRealization)
    elif output.type in ['DataSet']:
      self.raiseADebug('Dumping output in DataSet named ' + output.name)
      output.load(outputRealization,style='dataset')
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')
