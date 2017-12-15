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
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import utils
from utils import InputData
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
        scalarSpecification.addParam("percent", InputData.StringType)
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

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, inputDict, dict, dictionary of the converted data
    """
    # each post processor knows how to handle the coming inputs. The BasicStatistics postprocessor accept all the input type (files (csv only), hdf5 and datas
    self.dynamic = False
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0:
      self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")

    if type(currentInput).__name__ =='dict':
      if 'targets' not in currentInput.keys() and 'timeDepData' not in currentInput.keys():
        self.raiseAnError(IOError, 'Did not find targets or timeDepData in input dictionary')
      return [currentInput]
    if currentInput.type not in ['PointSet','HistorySet']:
      self.raiseAnError(IOError, self, 'BasicStatistics postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)

    metadata =  currentInput.getMeta(pointwise=True)
    inputList = []
    if currentInput.type == 'PointSet':
      inputDict = {}
      #FIXME: the following operation is slow, and we should operate on the data
      # directly without transforming it into dicts first.
      inputDict['targets'] = currentInput.getVarValues(self.parameters['targets'])
      inputDict['metadata'] =  metadata
      inputList.append(inputDict)
    elif currentInput.type == 'HistorySet':
      if self.pivotParameter is None:
        self.raiseAnError(IOError, self, 'Time-dependent statistics is requested (HistorySet) but no pivotParameter got inputted!')
      self.dynamic = True
      self.pivotValue = currentInput.asDataset()[self.pivotParameter].values
      if not currentInput.checkIndexAlignment(indexesToCheck=self.pivotParameter):
        self.raiseAnError(IOError, "The data provided by the data objects", currentInput.name, "is not synchronized!")
      slices = currentInput.sliceByIndex(self.pivotParameter)
      for sliceData in slices:
        inputDict = {}
        inputDict['metadata'] = metadata
        inputDict['targets'] = dict((target, sliceData[target]) for target in self.parameters['targets'])
        inputList.append(inputDict)

    self.raiseAMessage("Recasting performed")

    return inputList

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
        if 'percentile' not in self.toDo.keys():
          self.toDo[tag] = [] # list of {'targets':(), 'prefix':str, 'percent':str}
        if 'percent' not in child.parameterValues:
          strPercent = ['5', '95']
        else:
          strPercent = [child.parameterValues['percent']]
        for reqPercent in strPercent:
          self.toDo[tag].append({'targets':set(targets),
                                 'prefix':prefix,
                                 'percent':reqPercent})
      elif tag in self.scalarVals:
        self.toDo[tag] = [] # list of {'targets':(), 'prefix':str}
        self.toDo[tag].append({'targets':set(child.value),
                               'prefix':prefix})
      elif tag in self.vectorVals:
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
      output.addRealization(outputRealization)
    else:
      self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def __computeVp(self,p,weights):
    """
      Compute the sum of p-th power of weights
      @ In, p, int, the power
      @ In, weights, list or numpy.array, weights
      @ Out, vp, float, the sum of p-th power of weights
    """
    vp = np.sum(np.power(weights,p))
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

  def _computeKurtosis(self,arrayIn,expValue,variance,pbWeight=None):
    """
      Method to compute the Kurtosis (fisher) of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the Kurtosis needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Kurtosis of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(4,pbWeight) if not self.biased else 1.0
      if not self.biased:
        result = -3.0 + ((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,4.0),pbWeight))*unbiasCorr[0]-unbiasCorr[1]*np.power(((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,2.0),pbWeight))),2.0))/np.power(variance,2.0)
      else:
        result = -3.0 + ((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,4.0),pbWeight))*unbiasCorr)/np.power(variance,2.0)
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(4,len(arrayIn)) if not self.biased else 1.0
      if not self.biased:
        result = -3.0 + ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**4)*unbiasCorr[0]-unbiasCorr[1]*(np.average((arrayIn - expValue)**2))**2.0)/(variance)**2.0
      else:
        result = -3.0 + ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**4)*unbiasCorr)/(variance)**2.0
    return result

  def _computeSkewness(self,arrayIn,expValue,variance,pbWeight=None):
    """
      Method to compute the skewness of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the skewness needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the skewness of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(3,pbWeight) if not self.biased else 1.0
      result = (1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,3.0),pbWeight))*unbiasCorr/np.power(variance,1.5)
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(3,len(arrayIn)) if not self.biased else 1.0
      result = ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**3)*unbiasCorr)/np.power(variance,1.5)
    return result

  def _computeVariance(self,arrayIn,expValue,pbWeight=None):
    """
      Method to compute the Variance (fisher) of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the Variance needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Variance of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(2,pbWeight) if not self.biased else 1.0
      result = (1.0/self.__computeVp(1,pbWeight))*np.average((arrayIn - expValue)**2,weights= pbWeight)*unbiasCorr
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(2,len(arrayIn)) if not self.biased else 1.0
      result = np.average((arrayIn - expValue)**2)*unbiasCorr
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
      result = np.median(arrayIn)
    return result

  def __runLocal(self, inputDict):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In, inputDict, dict, dictionary containing the input, output, and metadata
      @ Out, outputDict, dict, Dictionary containing the results
    """
    pbWeights, pbPresent  = {'realization':None}, False
    # setting some convenience values
    parameterSet = list(self.allUsedParams)
    if 'metadata' in inputDict.keys():
      pbPresent = 'ProbabilityWeight' in inputDict['metadata'].keys() if 'metadata' in inputDict.keys() else False
    if not pbPresent:
      pbWeights['realization'] = None
      if 'metadata' in inputDict.keys():
        if 'SamplerType' in inputDict['metadata'].keys():
          if inputDict['metadata']['SamplerType'].values[0] != 'MonteCarlo' :
            self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
        else:
          self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights. Assuming unit weights instead...')
    else:
      pbWeights['realization'] = inputDict['metadata']['ProbabilityWeight'].values/np.sum(inputDict['metadata']['ProbabilityWeight'].values)
    #This section should take the probability weight for each sampling variable
    pbWeights['SampledVarsPbWeight'] = {'SampledVarsPbWeight':{}}
    if 'metadata' in inputDict.keys():
      for target in parameterSet:
        if 'ProbabilityWeight-'+target in inputDict['metadata'].keys():
          pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target] = np.asarray(inputDict['metadata']['ProbabilityWeight-'+target].values)
          pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target][:] = pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target][:]/np.sum(pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target])

    #establish a dict of indices to parameters and vice versa
    parameter2index = dict((param,p) for p,param in enumerate(inputDict['targets'].keys()))

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
    calculations = {}
    # do things in order to preserve prereqs
    # TODO many of these could be sped up through vectorization
    # TODO additionally, this could be done with less code duplication, probably
    #################
    # SCALAR VALUES #
    #################
    def startMetric(metric):
      """
        Common starting for each metric calculation.
        @ In, metric, string, name of metric
        @ Out, None
      """
      if len(needed[metric]['targets'])>0:
        self.raiseADebug('Starting "'+metric+'"...')
        calculations[metric]={}
    #
    # samples
    #
    metric = 'samples'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      calculations[metric][targetP] = len(inputDict['targets'][targetP].values)
    #
    # expected value
    #
    metric = 'expectedValue'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[metric][targetP] = np.average(inputDict['targets'][targetP].values, weights = relWeight)
      else:
        relWeight  = None
        calculations[metric][targetP] = np.mean(inputDict['targets'][targetP].values)
    #
    # variance
    #
    metric = 'variance'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeVariance(inputDict['targets'][targetP].values,calculations['expectedValue'][targetP],pbWeight=relWeight)
      #sanity check
      if (calculations[metric][targetP] == 0):
        self.raiseAWarning('The variable: ' + targetP + ' has zero variance! Please check your input in PP: ' + self.name)
    #
    # sigma
    #
    metric = 'sigma'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      if calculations['variance'][targetP] == 0:
        #np.Infinity:
        self.raiseAWarning('The variable: ' + targetP + ' has zero sigma! Please check your input in PP: ' + self.name)
        calculations[metric][targetP] = 0.0
      else:
        calculations[metric][targetP] = self._computeSigma(inputDict['targets'][targetP].values,calculations['variance'][targetP])
    #
    # coeff of variation (sigma/mu)
    #
    metric = 'variationCoefficient'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      if calculations['expectedValue'][targetP] == 0:
        self.raiseAWarning('Expected Value for ' + targetP + ' is zero! Variation Coefficient cannot be calculated, so setting as infinite.')
        calculations[metric][targetP] = np.Infinity
      else:
        calculations[metric][targetP] = calculations['sigma'][targetP]/calculations['expectedValue'][targetP]
    #
    # skewness
    #
    metric = 'skewness'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeSkewness(inputDict['targets'][targetP].values,calculations['expectedValue'][targetP],calculations['variance'][targetP],pbWeight=relWeight)
    #
    # kurtosis
    #
    metric = 'kurtosis'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeKurtosis(inputDict['targets'][targetP].values,calculations['expectedValue'][targetP],calculations['variance'][targetP],pbWeight=relWeight)
    #
    # median
    #
    metric = 'median'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[metric][targetP] = self._computeWeightedPercentile(inputDict['targets'][targetP].values,relWeight,percent=0.5)
      else:
        calculations[metric][targetP] = np.median(inputDict['targets'][targetP].values)
    #
    # maximum
    #
    metric = 'maximum'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      calculations[metric][targetP] = np.amax(inputDict['targets'][targetP].values)
    #
    # minimum
    #
    metric = 'minimum'
    startMetric(metric)
    for targetP in needed[metric]['targets']:
      calculations[metric][targetP] = np.amin(inputDict['targets'][targetP].values)

    #################
    # VECTOR VALUES #
    #################
    #
    # sensitivity matrix
    #
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
      allParams = set(needed[metric]['targets'])
      allParams.update(set(needed[metric]['features']))
      if len(needed[metric]['targets'])>0 and len(allParams)>=2:
        self.raiseADebug('Starting "'+metric+'"...')
        calculations[metric]={}
        targets = needed[metric]['targets']
        features = needed[metric]['features']
        skip = False #True only if we don't have targets and features
        if len(features)<1:
          self.raiseAWarning('No features specified for <'+metric+'>!  Please specify features in a <features> node (see the manual).  Skipping...')
          skip = True
      elif len(needed[metric]['targets']) == 0:
        #unrequested, no message needed
        pass
      elif len(allParams) < 2:
        #insufficient target/feature combinations (usually when only 1 target and 1 feature, and they are the same)
        self.raiseAWarning('A total of',len(allParams),'were provided for metric',metric,'but at least 2 are required!  Skipping...')
      if skip:
        if metric not in self.skipped.keys():
          self.skipped[metric] = {}
        self.skipped[metric].update(needed[metric])
      return targets,features,skip

    metric = 'sensitivity'
    targets,features,skip = startVector(metric)
    #NOTE sklearn expects the transpose of what we usually do in RAVEN, so #samples by #features
    if not skip:
      #for sensitivity matrix, we don't use numpy/scipy methods to calculate matrix operations,
      #so we loop over targets and features
      for t,target in enumerate(targets):
        calculations[metric][target] = {}
        targetVals = inputDict['targets'][target].values
        #don't do self-sensitivity
        inpSamples = np.atleast_2d(np.asarray(list(inputDict['targets'][f].values for f in features if f!=target))).T
        useFeatures = list(f for f in features if f != target)
        #use regressor coefficients as sensitivity
        regressDict = dict(zip(useFeatures, LinearRegression().fit(inpSamples,targetVals).coef_))
        for f,feature in enumerate(features):
          calculations[metric][target][feature] = 1.0 if feature==target else regressDict[feature]
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
      paramSamples = np.zeros((len(params), inputDict['targets'][params[0]].values.size))
      pbWeightsList = [None]*len(inputDict['targets'].keys())
      for p,param in enumerate(params):
        dataIndex = parameter2index[param]
        paramSamples[p,:] = inputDict['targets'][param].values[:]
        pbWeightsList[p] = pbWeights['realization'] if param not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][param]
      pbWeightsList.append(pbWeights['realization'])
      #Note: this is basically "None in pbWeightsList", but
      # using "is None" instead of "== None", which is more reliable
      if True in [x is None for x in pbWeightsList]:
        covar = self.covariance(paramSamples)
      else:
        covar = self.covariance(paramSamples, weights = pbWeightsList)
      calculations[metric]['matrix'] = covar
      calculations[metric]['params'] = params

    def getCovarianceSubset(desired):
      """
        @ In, desired, list(str), list of parameters to extract from covariance matrix
        @ Out, reducedSecond, np.array, reduced covariance matrix
        @ Out, wantedParams, list(str), parameter labels for reduced covar matrix
      """
      wantedIndices = list(calculations['covariance']['params'].index(d) for d in desired)
      wantedParams = list(calculations['covariance']['params'][i] for i in wantedIndices)
      #retain rows, colums
      reducedFirst = calculations['covariance']['matrix'][wantedIndices]
      reducedSecond = reducedFirst[:,wantedIndices]
      return reducedSecond, wantedParams
    #
    # pearson matrix
    #
    # see comments in covariance for notes on C implementation
    metric = 'pearson'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar,reducedParams = getCovarianceSubset(params)
      calculations[metric]['matrix'] = self.corrCoeff(reducedCovar)
      calculations[metric]['params'] = reducedParams
    #
    # VarianceDependentSensitivity matrix
    # The formula for this calculation is coming from: http://www.math.uah.edu/stat/expect/Matrices.html
    # The best linear predictor: L(Y|X) = expectedValue(Y) + cov(Y,X) * [vc(X)]^(-1) * [X-expectedValue(X)]
    # where Y is a vector of outputs, and X is a vector of inputs, cov(Y,X) is the covariance matrix of Y and X,
    # vc(X) is the covariance matrix of X with itself.
    # The variance dependent sensitivity matrix is defined as: cov(Y,X) * [vc(X)]^(-1)
    #
    metric = 'VarianceDependentSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar,reducedParams = getCovarianceSubset(params)
      inputSamples = np.zeros((len(params),inputDict['targets'][params[0]].values.size))
      pbWeightsList = [None]*len(params)
      for p,param in enumerate(reducedParams):
        inputSamples[p,:] = inputDict['targets'][param].values[:]
        pbWeightsList[p] = pbWeights['realization'] if param not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][param]
      pbWeightsList.append(pbWeights['realization'])
      for p,param in enumerate(reducedParams):
        calculations[metric][param] = {}
        targCoefs = list(r for r in reducedParams if r!=param)
        inpParams = np.delete(inputSamples,p,axis=0)
        inpCovMatrix = np.delete(reducedCovar,p,axis=0)
        inpCovMatrix = np.delete(inpCovMatrix,p,axis=1)
        outInpCov = np.delete(reducedCovar[p,:],p)
        sensCoefDict = dict(zip(targCoefs,np.dot(outInpCov,np.linalg.pinv(inpCovMatrix))))
        for f,feature in enumerate(reducedParams):
          if param == feature:
            calculations[metric][param][feature] = 1.0
          else:
            calculations[metric][param][feature] = sensCoefDict[feature]
    #
    # Normalized variance dependent sensitivity matrix
    # variance dependent sensitivity  normalized by the mean (% change of output)/(% change of input)
    #
    metric = 'NormalizedSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      reducedCovar,reducedParams = getCovarianceSubset(params)
      for p,param in enumerate(reducedParams):
        calculations[metric][param] = {}
        for f,feature in enumerate(reducedParams):
          expValueRatio = calculations['expectedValue'][feature]/calculations['expectedValue'][param]
          calculations[metric][param][feature] = calculations['VarianceDependentSensitivity'][param][feature]*expValueRatio

    # The following dict is used to collect outputs from calculations
    outputDict = {}
    #
    # percentile, this metric is handled differently
    #
    metric = 'percentile'
    if len(needed[metric]['targets'])>0:
      self.raiseADebug('Starting "'+metric+'"...')
      for targetDict in self.toDo[metric]:
        percent = float(targetDict['percent'])
        prefix = targetDict['prefix'].strip()
        for targetP in targetDict['targets']:
          varName = '_'.join([prefix, targetDict['percent'].strip(), targetP])
          if pbPresent:
            relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
            outputDict[varName] = np.atleast_1d(self._computeWeightedPercentile(inputDict['targets'][targetP].values,relWeight,percent=float(percent)/100.0))
          else:
            outputDict[varName] = np.percentile(inputDict['targets'][targetP].values, float(percent), interpolation='lower')

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
          skip = self.skipped.get(metric, None)
          if skip is not None:
            self.raiseADebug('Metric',metric,'was skipped for parameters',targetDict,'!  See warnings for details.  Ignoring...')
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

    return outputDict

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputAdapted = self.inputToInternal(inputIn)
    if not self.dynamic:
      outputDict = self.__runLocal(inputAdapted[0])
    else:
      # time dependent (actually pivot-dependent)
      self.raiseADebug('BasicStatistics Pivot-Dependent output:')
      outputList = []
      for inputDict in inputAdapted:
        outputList.append(self.__runLocal(inputDict))
      #FIXME: swith to defaultdict
      outputDict = dict((var,list()) for var in outputList[0].keys())
      for output in outputList:
        for var, value in output.items():
          outputDict[var] = np.append(outputDict[var], value)
      # add the pivot parameter and its values
      outputDict[self.pivotParameter] = np.atleast_1d(self.pivotValue)

    return outputDict

  def covariance(self, feature, weights = None, rowVar = 1):
    """
      This method calculates the covariance Matrix for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calculated depending on the selection of the inputs.
      @ In,  feature, list/numpy.array, [#targets,#samples]  features' samples
      @ In,  weights, list of list/numpy.array, optional, [#targets,#samples,realizationWeights]  reliability weights, and the last one in the list is the realization weights. Default is None
      @ In,  rowVar, int, optional, If rowVar is non-zero, then each row represents a variable,
                                    with samples in the columns. Otherwise, the relationship is transposed. Default=1
      @ Out, covMatrix, list/numpy.array, [#targets,#targets] the covariance matrix
    """
    X = np.array(feature, ndmin = 2, dtype = np.result_type(feature, np.float64))
    w = np.zeros(feature.shape, dtype = np.result_type(feature, np.float64))
    if X.shape[0] == 1:
      rowVar = 1
    if rowVar:
      N = X.shape[1]
      featuresNumber = X.shape[0]
      axis = 0
      for myIndex in range(featuresNumber):
        if weights is None:
          w[myIndex,:] = np.ones(N)/float(N)
        else:
          w[myIndex,:] = np.array(weights[myIndex],dtype=np.result_type(feature, np.float64))[:] if weights is not None else np.ones(len(w[myIndex,:]),dtype =np.result_type(feature, np.float64))[:]
    else:
      N = X.shape[0]
      featuresNumber = X.shape[1]
      axis = 1
      for myIndex in range(featuresNumber):
        if weights is None:
          w[myIndex,:] = np.ones(N)/float(N)
        else:
          w[:,myIndex] = np.array(weights[myIndex], dtype=np.result_type(feature, np.float64))[:] if weights is not None else np.ones(len(w[:,myIndex]),dtype=np.result_type(feature, np.float64))[:]
    realizationWeights = weights[-1] if weights is not None else np.ones(N)/float(N)
    if N <= 1:
      self.raiseAWarning("Degrees of freedom <= 0")
      return np.zeros((featuresNumber,featuresNumber), dtype = np.result_type(feature, np.float64))
    diff = X - np.atleast_2d(np.average(X, axis = 1 - axis, weights = w)).T
    covMatrix = np.ones((featuresNumber,featuresNumber), dtype = np.result_type(feature, np.float64))
    for myIndex in range(featuresNumber):
      for myIndexTwo in range(featuresNumber):
        # The weights that are used here should represent the joint probability (P(x,y)).
        # Since I have no way yet to compute the joint probability with weights only (eventually I can think to use an estimation of the P(x,y) computed through a 2D histogram construction and weighted a posteriori with the 1-D weights),
        # I decided to construct a weighting function that is defined as Wi = (2.0*Wi,x*Wi,y)/(Wi,x+Wi,y) that respects the constrains of the
        # covariance (symmetric and that the diagonal is == variance) but that is completely arbitrary and for that not used. As already mentioned, I need the joint probability to compute the E[XY] = integral[xy*p(x,y)dxdy]. Andrea
        # for now I just use the realization weights
        #jointWeights = (2.0*weights[myIndex][:]*weights[myIndexTwo][:])/(weights[myIndex][:]+weights[myIndexTwo][:])
        #jointWeights = jointWeights[:]/np.sum(jointWeights)
        if myIndex == myIndexTwo:
          jointWeights = w[myIndex]/np.sum(w[myIndex])
        else:
          jointWeights = realizationWeights/np.sum(realizationWeights)
        fact = self.__computeUnbiasedCorrection(2,jointWeights) if not self.biased else 1.0/np.sum(jointWeights)
        covMatrix[myIndex,myIndexTwo] = np.sum(diff[:,myIndex]*diff[:,myIndexTwo]*jointWeights[:]*fact) if not rowVar else np.sum(diff[myIndex,:]*diff[myIndexTwo,:]*jointWeights[:]*fact)
    return covMatrix

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
      corrMatrix = covM / np.sqrt(np.multiply.outer(d, d))
    except ValueError:
      # scalar covariance
      # nan if incorrect value (nan, inf, 0), 1 otherwise
      corrMatrix = covM / covM
    # to prevent numerical instability
    return corrMatrix
