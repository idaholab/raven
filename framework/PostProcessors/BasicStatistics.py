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

@author: alfoa
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


#class BasicStatisticsInput(InputData.ParameterInput):
#  """
#    Class for reading the Basic Statistics block
#  """

#BasicStatisticsInput.createClass("PostProcessor", False, baseNode=ModelInput)
#BasicStatisticsInput.addSub(WhatInput)
#BiasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType) #bool
#BasicStatisticsInput.addSub(BiasedInput)
#ParameterInput = InputData.parameterInputFactory("parameters", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(ParameterInput)
#MethodsToRunInput = InputData.parameterInputFactory("methodsToRun", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(MethodsToRunInput)
#FunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(FunctionInput)
#PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(PivotParameterInput)

#
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
        scalarSpecification.addParam("percent", InputData.FloatListType)
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

    methodsToRunInput = InputData.parameterInputFactory("methodsToRun", contentType=InputData.StringType)
    inputSpecification.addSub(methodsToRunInput)

    biasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType) #bool
    inputSpecification.addSub(biasedInput)

    ## TODO: Fill this in with the appropriate tags

    # inputSpecification.addSub(WhatInput)
    # BiasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType) #bool
    # inputSpecification.addSub(BiasedInput)
    # ParameterInput = InputData.parameterInputFactory("parameters", contentType=InputData.StringType)
    # inputSpecification.addSub(ParameterInput)
    # FunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
    # inputSpecification.addSub(FunctionInput)
    # PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
    # inputSpecification.addSub(PivotParameterInput)

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
    self.realizationWeight = None

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, inputDict, dict, dictionary of the converted data
    """
    # The BasicStatistics postprocessor only accept DataObjects
    self.dynamic = False
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0:
      self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")

    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'BasicStatistics postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)

    # extract all required data from input DataObjects, an input dataset is constructed
    dataSet = currentInput.asDataset()
    inputDataset = dataSet[self.parameters['targets']]
    self.sampleTag = currentInput.sampleTag

    if currentInput.type == 'HistorySet':
      dims = inputDataset.dims.keys()
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
    pbWeights = None
    self.pbPresent = True if 'ProbabilityWeight' in metaVars else False
    if self.pbPresent:
      pbWeights = xr.Dataset()
      self.realizationWeight = dataSet['ProbabilityWeight']/dataSet['ProbabilityWeight'].sum()
      for target in self.parameters['targets']:
        pbName = 'ProbabilityWeight-' + target
        if pbName in metaVars:
          pbWeights[target] = dataSet[pbName]/dataSet[pbName].sum()
        elif self.pbPresent:
          pbWeights[target] = self.realizationWeight
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
          reqPercent = [5, 95]
        else:
          reqPercent = child.parameterValues['percent']
        self.toDo[tag].append({'targets':set(targets),
                               'prefix':prefix,
                               'percent':reqPercent})
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
      else:
        self.raiseAWarning('Unrecognized node in BasicStatistics "',tag,'" has been ignored!')
    assert (len(self.toDo)>0), self.raiseAnError(IOError, 'BasicStatistics needs parameters to work on! Please check input for PP: ' + self.name)

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
      self.raiseADebug('Dumping output in data object named ' + output.name)
      #output.addRealization(outputRealization)
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def __computePower(self, p, dataset):
    """
      Compute the p-th power of weights
      @ In, p, int, the power
      @ In, dataset, xarray.Dataset, weights
      @ Out, pw, xarray.Dataset, the p-th power of weights
    """
    pw = xr.Dataset()
    for target, targValue in weights.data_vars.items():
      pw[target] = np.power(weights,p)
    return pw

  def __computeVp(self,p,weights):
    """
      Compute the sum of p-th power of weights
      @ In, p, int, the power
      @ In, weights, xarray.Dataset, weights
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
      @ In, weightsOrN, list/numpy.array or int, if list/numpy.array -> weights else -> number of samples
      @ Out, corrFactor, float (order <=3) or tuple of floats (order ==4), the unbiased correction factor
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
      @ In, arrayIn, list/numpy.array, the array of values from which the Kurtosis needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Kurtosis of the array of data
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
      @ In, arrayIn, list/numpy.array, the array of values from which the skewness needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the skewness of the array of data
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
      @ In, arrayIn, list/numpy.array, the array of values from which the Variance needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Variance of the array of data
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
      @ In, arrayIn, list/numpy.array, the array of values from which the sigma needs to be estimated
      @ In, variance, float, variance of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, sigma, float, the sigma of the array of data
    """
    return np.sqrt(variance)

  def __runLocal(self, inputData):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In, inputDict, dict, dictionary containing the input, output, and metadata
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputDataset, pbWeights = inputData[0], inputData[1]
    #storage dictionary for skipped metrics
    self.skipped = {}
    #construct a dict of required computations
    needed = dict((metric,{'targets':set()}) for metric in self.scalarVals)
    needed.update(dict((metric,{'targets':set(),'features':set()}) for metric in self.vectorVals))
    for metric, params in self.toDo.items():
      for entry in params:
        needed[metric]['targets'].update(entry['targets'])
        try:
          needed[metric]['features'].update(entry['features'])
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

    #
    # BEGIN actual calculations
    #
    # do things in order to preserve prereqs
    # TODO many of these could be sped up through vectorization
    # TODO additionally, this could be done with less code duplication, probably
    # Store the calculation results into a data set
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
      # TODO: remove latter
      #samples = xr.DataArray([numRlz]*len(self.parameters['targets']),dims=('targets'),coords={'targets':self.parameters['targets']})
      calculations[metric] = numRlz
    #
    # expected value
    #
    metric = 'expectedValue'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      if self.pbPresent:
        relWeight = pbWeights[list(needed[metric]['targets'])]
        dataSet = dataSet * relWeights
        expectedValue = dataSet.sum(dim = self.sampleTag)
      else:
        expectedValue = dataSet.mean(dim = self.sampleTag)
      calculations[metric] = expectedValue
    #
    # variance
    #
    metric = 'variance'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      dataSet = inputDataset[list(needed[metric]['targets'])]
      meanSet = calculations['expectedValue'][list(needed[metric]['targets'])]
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      calculations[metric] = self._computeVariance(dataSet,meanSet,pbWeight=relWeight,dim=self.sampleTag)
    #
    # sigma
    #
    metric = 'sigma'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      calculations[metric] = self.__computePower(0.5,calculations['variance'][list(needed[metric]['targets'])])
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
      relWeight = pbWeights[list(needed[metric]['targets'])] if self.pbPresent else None
      if self.pbPresent:
        dataSetWeighted = dataSet * relWeight
        # interpolation: {'linear', 'lower', 'higher','midpoint','nearest'}, do not try to use 'linear' or 'midpoint'
        # The xarray.Dataset.where() will not return the corrrect solution
        # 'lower' is used for consistent
        medianSet = dataSet.where(dataSetWeighted==dataSetWeighted.quantile(0.5,dim=self.sampleTag,interpolation='lower')).sum(self.sampleTag)
        calculations[metric] = medianSet
      else:
        calculations[metric] = dataSet.median(dim=self.sampleTag)
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
      calculations[metric] = []
      for targetDict in self.toDo[metric]:
        percent = targetDict['percent']
        dataSet = inputDataset[list(targetDict['targets'])]
        if self.pbPresent:
          relWeight = pbWeights[list(targetDict['targets'])]
          dataSetWeighted = dataSet * relWeight
          # interpolation: {'linear', 'lower', 'higher','midpoint','nearest'}, do not try to use 'linear' or 'midpoint'
          # The xarray.Dataset.where() will not return the corrrect solution
          # 'lower' is used for consistent
          # using xarray.Dataset.sel(**{'quantile':reqPercent}) to retrieve the quantile values
          percentileSet = dataSet.where(dataSetWeighted==dataSetWeighted.quantile(percent,dim=self.sampleTag,interpolation='lower')).sum(self.sampleTag)
          calculations[metric].append(percentileSet)
        else:
          percentileSet = dataSet.quantile(percent,dim=self.sampleTag,interpolation='lower')
          calculations[metric].append(percentileSet)

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
        targets = needed[metric]['targets']
        features = needed[metric]['features']
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
      calculations[metric] = []
      for paramDict in self.toDo[metric]:
        targList = paramDict['targets']
        featList = paramDict['features']
        dataSet = inputDataset[targList + featList]
        if self.pivotParameter in dataSet.dims.keys():
          ds = None
          pivotVals = []
          for label, group in dataSet.groupby(self.pivotParameter):
            # construct target and feature matrices
            featSamples = group[featList].to_array().transpose(self.sampleTag,'variable')
            targSamples = group[targList].to_array().transpose(self.sampleTag,'variable')
            featVars = featSamples.coords['variable'].values
            targVars = targSamples.coords['variable'].values
            regCoeff = LinearRegression().fit(featSamples.values,targSamples.values).coef_
            pivotVals.append(label)
            da = xr.DataArray(regCoeff, dims=('targets','features'), coords={'targets':targVars,'features':featVars})
            ds = da if ds is None else xr.concat([ds,da], dim=self.pivotParameter)
          ds.coords[self.pivotParameter] = pivotVals
          calculations[metric].append(ds)
        else:
          # construct target and feature matrices
          featSamples = dataSet[featList].to_array().transpose(self.sampleTag,'variable')
          targSamples = dataSet[targList].to_array().transpose(self.sampleTag,'variable')
          featVars = featSamples.coords['variable'].values
          targVars = targSamples.coords['variable'].values
          regCoeff = LinearRegression().fit(featSamples.values,targSamples.values).coef_
          da = xr.DataArray(regCoeff, dims=('targets','features'), coords={'targets':targVars,'features':featVars})
          calculations[metric].append(da)
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
      calculations[metric] = {}
      params = list(set(targets).union(set(features)))
      dataSet = inputDataset[params]
      realWeight = pbWeight[params] if self.pbPresent else None
      if self.pbPresent:
        fact = self.__computeUnbiasedCorrection(2, self.realizationWeight) if not self.biased else 1.0
        meanSet = (dataSet * relWeights).sum(dim = self.sampleTag)
      else:
        meanSet = dataSet.mean(dim = self.sampleTag)
        fact = 1.0 / (float(dataSet.sizes[self.sampleTag]) - 1.0) if not self.biased else 1.0 / float(dataSet.sizes[self.sampleTag])
      varianceSet = self._computeVariance(dataSet,meanSet,pbWeight=relWeight,dim=self.sampleTag)
      dataSet = dataSet - meanSet
      if self.pivotParameter in dataSet.dims.keys():
        ds = None
        pivotVals = []
        for label, group in dataSet.groupby(self.pivotParameter):
          # construct target and feature matrices
          paramSamples = group.to_array().transpose('variable',self.sampleTag)
          targVars = paramSamples.coords['variable'].values
          paramSamples = paramSamples.values
          if self.pbPresent:
            paramSamplesT = (paramSamples*self.realizationWeight.values).T
          else:
            paramSamplesT = paramSamples.T
          cov = np.dot(paramSamples, paramSamplesT.conj())
          cov *= fact
          variance = varianceSet[targVars].sel(**{self.pivotParameter:label}).to_array().values
          np.fill_diagonal(cov,variance)
          pivotVals.append(label)
          da = xr.DataArray(cov, dims=('targets','features'), coords={'targets':targVars,'features':targVars})
          ds = da if ds is None else xr.concat([ds,da], dim=self.pivotParameter)
        ds.coords[self.pivotParameter] = pivotVals
        calculations[metric] = ds
      else:
        # construct target and feature matrices
        paramSamples = group.to_array().transpose('variable',self.sampleTag)
        targVars = paramSamples.coords['variable'].values
        paramSamples = paramSamples.values
        if self.pbPresent:
          paramSamplesT = (paramSamples*self.realizationWeight.values).T
        else:
          paramSamplesT = paramSamples.T
        cov = np.dot(paramSamples, paramSamplesT.conj())
        cov *= fact
        variance = varianceSet[targVars].to_array().values
        np.fill_diagonal(cov,variance)
        da = xr.DataArray(cov, dims=('targets','features'), coords={'targets':targVars,'features':targVars})
        calculations[metric] = da

    def getCovarianceSubset(desired):
      """
        @ In, desired, list(str), list of parameters to extract from covariance matrix
        @ Out, reducedSecond, np.array, reduced covariance matrix
        @ Out, wantedParams, list(str), parameter labels for reduced covar matrix
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
      if self.pivotParameter in reducedCovar.dims.keys():
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
      if self.pivotParameter in reducedCovar.dims.keys():
        pivotCoords = reducedCovar.coords[self.pivotParameter].values
        ds = None
        for label, group in reducedCovar.groupby(self.pivotParameter):
          senMatrix = np.zeros(len(targCoords), len(targCoords))
          covMatrix = group.values
          for p,param in enumerate(targCoords):
            reduceTargs = list(r for r in reducedParams if r!=param)
            covX = np.delete(covMatrix,p,axis=0)
            covX = np.delete(covX,p,axis=1)
            covYX = np.delete(covMatrix[p,:],p)
            sensCoef = np.dot(covYX,np.linalg.pinv(covX))
            np.insert(sensCoef,p,1.0)
            senMatrix[p,:] = sensCoef
          da = xr.DataArray(senMatrix, dims=('targets','features'), coords={'targets':targCoords,'features':targCoords})
          ds = da if ds is None else xr.concat([ds,da], dim=self.pivotParameter)
        ds.coords[self.pivotParameter] = pivotCoords
        calculations[metric] = ds
      else:
        senMatrix = np.zeros(len(targCoords), len(targCoords))
        covMatrix = reducedCovar.values
        for p,param in enumerate(targCoords):
          reduceTargs = list(r for r in reducedParams if r!=param)
          covX = np.delete(covMatrix,p,axis=0)
          covX = np.delete(covX,p,axis=1)
          covYX = np.delete(covMatrix[p,:],p)
          sensCoef = np.dot(covYX,np.linalg.pinv(covX))
          np.insert(sensCoef,p,1.0)
          senMatrix[p,:] = sensCoef
        da = xr.DataArray(senMatrix, dims=('targets','features'), coords={'targets':targCoords,'features':targCoords})
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
      meanDA.rename({'variable':'targets'})
      reducedSen /= meanDA
      meanDA.rename({'targets':'features'})
      reducedSen *= meanDA
      calculations[metric] = reducedSen
    """
    #collect only the requested calculations except percentile, since it has been already collected
    #in the outputDict.
    for metric, requestList  in self.toDo.items():
      if metric == 'percentile':
        continue
      for targetDict in requestList:
        prefix = targetDict['prefix'].strip()
        if metric in self.scalarVals:
          for targetP in targetDict['targets']:
            varName = prefix + '_' + targetP
            outputDict[varName] = np.atleast_1d(calculations[metric][targetP])
        #TODO someday we might need to expand the "skipped" check to include scalars, but for now
        #   the only reason to skip is if an invalid matrix is requested
        #if matrix block, extract desired values
        else:
          #check if it was skipped for some reason
          skip = self.skipped.get(metric, False)
          if skip:
            continue
          if metric in ['pearson', 'covariance']:
            for targetP in targetDict['targets']:
              targetIndex = calculations[metric]['params'].index(targetP)
              for feature in targetDict['features']:
                varName = prefix + '_' + targetP + '_' + feature
                featureIndex = calculations[metric]['params'].index(feature)
                outputDict[varName] = np.atleast_1d(calculations[metric]['matrix'][targetIndex,featureIndex])
          #if matrix but stored in dictionaries, just grab the values
          elif metric in ['sensitivity','NormalizedSensitivity','VarianceDependentSensitivity']:
            for targetP in targetDict['targets']:
              for feature in targetDict['features']:
                varName = prefix + '_' + targetP + '_' + feature
                outputDict[varName] = np.atleast_1d(calculations[metric][targetP][feature])
    """

    return calculations

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputData = self.inputToInternal(inputIn)
    outputDict = self.__runLocal(inputData)

    return outputDict

  def corrCoeff(self, covM):
    """
      This method calculates the correlation coefficient Matrix (pearson) for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calcuated depending on the selection of the inputs.
      @ In,  covM, list/numpy.array, [#targets,#targets] covariance matrix
      @ Out, corrMatrix, list/numpy.array, [#targets,#targets] the correlation matrix
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

