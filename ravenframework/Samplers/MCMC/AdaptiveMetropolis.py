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
  Adaptive Metropolis Hastings Algorithm for Markov Chain Monte Carlo

  Created on Dec. 16, 2020
  @author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import abc
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .MCMC import MCMC
from ...utils import utils,randomUtils,InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class AdaptiveMetropolis(MCMC):
  """
    Adaptive Metropolis Hastings Sampler
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
    inputSpecification = super(AdaptiveMetropolis, cls).getInputSpecification()
    samplerInit = inputSpecification.getSub('samplerInit')
    adaptiveInterval = InputData.parameterInputFactory("adaptiveInterval", contentType=InputTypes.IntegerType,
        descr='The number of sample steps for each proposal parameters update')
    samplerInit.addSub(adaptiveInterval)
    inputSpecification.addSub(samplerInit)
    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    MCMC.__init__(self)
    self._optAlpha = 0.234 # optimum acceptance rate
    self._lambda = None # adaptive updated factor for covariance matrix
    self._gamma = None  # adaptive step size
    self._ensembleMean = None # The mean vector of ordered variables that need to be calibrated
    self._ensembleCov = None  # The covariance matrix of ordered variables
    self._orderedVars = OrderedDict() # ordered dict of variables that is used to construct proposal function
    self._orderedVarsList = [] # List of ordered variables
    self._adaptiveInterval = 20 # The number of sample steps for each adaptive update of scaling and covariance parameters

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    MCMC.handleInput(self, paramInput)
    init = paramInput.findFirst('samplerInit')
    if init is not None:
      adaptiveInterval = init.findFirst('adaptiveInterval')
      if adaptiveInterval is not None:
        self._adaptiveInterval = adaptiveInterval.value

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean Adaptive Metropolis is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    MCMC.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    ## retrieve proposal distribution
    if len(self._proposal) != 0:
      for var in self._updateValues:
        if var in self._proposal:
          self._proposal[var] = self.retrieveObjectFromAssemblerDict('proposal', self._proposal[var])
        else:
          std = self._stdProposalDefault
          if var in self.distDict:
            dist = self.distDict[var]
            untrStdDev = dist.untruncatedStdDev()
            std *= untrStdDev
          propDist = self._availProposal['normal'](0.0, std)
          propDist.initializeDistribution()
          self._proposal[var] = propDist
          self.raiseAWarning('"proposal" is not provided for variable "{}", default normal distribution with std={} is used!'.format(var, std))
    ## compute initial gamma and lambda
    totalNumVars = len(self._updateValues)
    self._lambda = 2.38**2/totalNumVars
    if totalNumVars != len(self.toBeCalibrated):
      self.raiseAnError(IOError, 'AdaptiveMetropolis can not handle "probabilityFunction" yet!',
                        'Please check your input and provide "distribution" instead of "probabilityFunction"!')
    ## construct ordered variable list
    ## construct ensemble covariance that will be used for proposal distribution
    ## ToDO: current structure only works for untruncated distribution
    self._ensembleCov = np.zeros((totalNumVars, totalNumVars))
    index = 0
    ## Construct proposal distribution
    if len(self._proposal) != 0:
      for propDistName, elementList in self._proposalDist.items():
        dim = elementList[0][1]
        if dim is None:
          for elem in elementList:
            key = elem[0]
            if key in self._updateValues.keys():
              self._orderedVarsList.append(key)
              dist = self._proposal[key]
              sigma = dist.untruncatedStdDev()
              self._ensembleCov[index, index] = sigma**2
              ## update index
              index += 1
              # self._orderedVars: {distName:[[varlist], []]}
              distName = self.variables2distributionsMapping[key]['name']
              totDim = max(self.distributions2variablesIndexList[distName])
              if totDim > 1:
                self.raiseAnError(IOError, 'Single dimension proposal distribution is provided for multivariate distribution variable "{}", this is not allowed!'.format(key))
              if distName not in self._orderedVars:
                self._orderedVars[distName] = [[key]]
              else:
                self._orderedVars[distName].append([key])
        else:
          orderedVars = [k for k, v in sorted(elementList, key=lambda item: item[1])]
          if len(orderedVars) != len(set(orderedVars)):
            self.raiseAnError(IOError, "Duplicated value of 'dim' is found for proposal distribution '{}'".format(distName))
          var = orderedVars[0]
          proposalDist = self._proposal[var]
          distName = self.variables2distributionsMapping[var]['name']
          totDim = max(self.distributions2variablesIndexList[distName])
          if totDim > 1:
            listElem = self.distributions2variablesMapping[distName]
            elemDict = {}
            for elem in listElem:
              elemDict.update(elem)
            checkVars = [k for k, v in sorted(elemDict.items(), key=lambda item: item[1])]
            if checkVars != orderedVars:
              self.raiseAnError(IOError, 'The "dim" provided by "distribution" and "proposal" is not consistent for variables:', ','.join(checkVars))
            self._orderedVars[distName] = [orderedVars]
          else:
            for key in orderedVars:
              distName   = self.variables2distributionsMapping[key]['name']
              totDim = max(self.distributions2variablesIndexList[distName])
              if totDim > 1:
                self.raiseAnError(IOError,'Multivariate proposal distribution is not allowed for mixed single and multivariate distribution!')
              if distName not in self._orderedVars:
                self._orderedVars[distName] = [[key]]
              else:
                self._orderedVars[distName].append([key])
          self._orderedVarsList.extend(orderedVars)
          if proposalDist.type != 'MultivariateNormal':
            self.raiseAnError(IOError, 'Only accept "MultivariateNormal" distribution, but got "{}"'.format(proposalDist.type))
          mean = proposalDist.mu
          cov = proposalDist.covariance
          totDim = len(mean)
          cov = np.asarray(cov).reshape((totDim, totDim))
          self._ensembleCov[index:index+totDim, index:index+totDim] = cov
          ## update index
          index += totDim
    else:
      for distName, elementList in self.distributions2variablesMapping.items():
        totDim = max(self.distributions2variablesIndexList[distName])
        orderedVars = []
        if totDim == 1:
          for elem in elementList:
            key = list(elem.keys())[0]
            if key in self._updateValues.keys():
              orderedVars.append([key])
              self._orderedVarsList.append(key)
              dist = self.distDict[key]
              sigma = dist.untruncatedStdDev()
              self._ensembleCov[index, index] = sigma**2
              ## update index
              index += 1
          # self._orderedVars: {distName:[[varlist], []]}
          self._orderedVars[distName] = orderedVars
        else:
          elemDict = {}
          for elem in elementList:
            elemDict.update(elem)
          orderedVars = [k for k, v in sorted(elemDict.items(), key=lambda item: item[1])]
          var = orderedVars[0]
          if var in self._updateValues.keys():
            self._orderedVars[distName] = [orderedVars]
            self._orderedVarsList.extend(orderedVars)
            dist = self.distDict[var]
            if dist.type != 'MultivariateNormal':
              self.raiseAnError(IOError, 'Only accept "MultivariateNormal" distribution, but got "{}"'.format(dist.type))
            mean = dist.mu
            cov = dist.covariance
            totDim = len(mean)
            cov = np.asarray(cov).reshape((totDim, totDim))
            self._ensembleCov[index:index+totDim, index:index+totDim] = cov
            ## update index
            index += totDim

    self._ensembleMean = np.asarray([self._updateValues[var] for var in self._orderedVarsList])
    size = len(self._ensembleMean)
    self._proposal = self.constructProposalDistribution(np.zeros(size), self._lambda*self._ensembleCov.ravel())

    ## initialize variables
    for distName, elementList in self.distributions2variablesMapping.items():
      totDim = max(self.distributions2variablesIndexList[distName])
      if totDim == 1:
        for elem in elementList:
          key = list(elem.keys())[0]
          if key in self._updateValues and self._updateValues[key] is None:
            dist = self.distDict[key]
            value = dist.rvs()
            self._updateValues[key] = value
      else:
        elemDict = {}
        for elem in elementList:
          elemDict.update(elem)
        orderedVars = [k for k, v in sorted(elemDict.items(), key=lambda item: item[1])]
        var = orderedVars[0]
        if var in self._updateValues.keys() and self._updateValues[var] is None:
          dist = self.distDict[var]
          value = dist.rvs()
          for i, var in enumerate(orderedVars):
            self._updateValues[var] = value[i]

  def constructProposalDistribution(self, mu, cov):
    """
      Methods to construct proposal distribution
      @ In, mu, list or 1-d numpy.array, the mean value
      @ In, cov, list or 1-d numpy.array, the covariance value
      @ Out, proposal, Distribution Object, the constructed distribution object.
    """
    proposal = self._availProposal['multivariateNormal']()
    proposal.mu = mu
    proposal.covariance = cov.ravel()
    proposal.dimension = len(mu)
    proposal.rank = len(mu)
    proposal.initializeDistribution()
    return proposal

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    self.values.update(self._updateValues)
    if self.counter > 1:
      self._localReady = False
      newVal = self._proposal.rvs()
      # update sampled value using proposal distribution
      for i, var in enumerate(self._orderedVarsList):
        ## scaling for the new generated inputs
        self.values[var] = self._updateValues[var] + newVal[i] * self._scaling
        ## check the lowerBound and upperBound
        lowerBound = self.distDict[var].lowerBound
        upperBound = self.distDict[var].upperBound
        distName = self.variables2distributionsMapping[var]['name']
        totDim = max(self.distributions2variablesIndexList[distName])
        if totDim > 1:
          dim = self.variables2distributionsMapping[var]['dim']
          lowerBound = lowerBound[dim-1]
          upperBound = upperBound[dim-1]
        if lowerBound is not None and self.values[var] < lowerBound:
          self.values[var] = lowerBound
        if upperBound is not None and self.values[var] > upperBound:
          self.values[var] = upperBound
    self._setProbabilities()
    self.inputInfo['LogPosterior'] = self.netLogPosterior
    self.inputInfo['AcceptRate'] = self._acceptRate

  def _setProbabilities(self):
    """
      Method to compute probability related information
      @ In, None
      @ Out, None
    """
    for distName, orderedVars in self._orderedVars.items():
      totDim = max(self.distributions2variablesIndexList[distName])
      dist = self.distDict[orderedVars[0][0]]
      if totDim == 1:
        for var in orderedVars:
          key = var[0]
          value = self.values[key]
          self.inputInfo['SampledVarsPb'][key] = dist.pdf(value)
          self.inputInfo['ProbabilityWeight-' + key] = 1.
      else:
        value = [self.values[var] for var in orderedVars[0]]
        for var in orderedVars[0]:
          self.inputInfo['SampledVarsPb'][var] = dist.pdf(value)
        self.inputInfo['ProbabilityWeight-' + distName] = 1.
    self.inputInfo['PointProbability'] = 1.0
    self.inputInfo['ProbabilityWeight' ] = 1.0
    self.inputInfo['SamplerType'] = 'Metropolis'

  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      General function (available to all samplers) that finalize the sampling
      calculation just ended. In this case, The function is aimed to check if
      all the batch calculations have been performed
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
      @ Out, None
    """
    MCMC.localFinalizeActualSampling(self, jobObject, model, myInput)
    if self.counter > 1:
      self._updateAdaptiveParams(self.netLogPosterior, self._currentRlz)

  def _useRealization(self, newRlz, currentRlz):
    """
      Used to feedback the collected runs within the sampler
      @ In, newRlz, dict, new generated realization
      @ In, currentRlz, dict, the current existing realization
      @ Out, netLogPosterior, float, the accepted probabilty
    """
    ## first compute acceptable probability vs. netLogLikelihood
    netLogPosterior = 0
    # compute net log prior
    for distName, orderedVars in self._orderedVars.items():
      totDim = max(self.distributions2variablesIndexList[distName])
      dist = self.distDict[orderedVars[0][0]]
      if totDim == 1:
        for var in orderedVars:
          key = var[0]
          netLogPosterior += dist.logPdf(newRlz[key]) - dist.logPdf(currentRlz[key])
      else:
        newVal = [newRlz[var] for var in orderedVars[0]]
        currVal = [currentRlz[var] for var in orderedVars[0]]
        netLogPosterior += dist.logPdf(newVal) - dist.logPdf(currVal)
    if not self._logLikelihood:
      netLogLikelihood = np.log(newRlz[self._likelihood]) - np.log(currentRlz[self._likelihood])
    else:
      netLogLikelihood = newRlz[self._likelihood] - currentRlz[self._likelihood]
    netLogPosterior += netLogLikelihood
    netLogPosterior = min(0.0, netLogPosterior)
    return netLogPosterior

  def _updateAdaptiveParams(self, alpha, rlz):
    """
      Used to feedback the collected runs within the sampler
      @ In, alpha, float, the accepted probabilty
      @ In, rlz, dict, the updated current existing realization
      @ Out, None
    """
    ### first use normal strategy (tuneScalingParam) to update scaling parameter until burnIn
    ### Reset scaling and then start to use adaptive approach to update scaling and cov parameters
    if self.counter == self._burnIn:
      self._lambda = self._scaling**2
      self._scaling = 1.
      self._tune = False
    elif self.counter > self._burnIn:
      orderedVarsVals = np.asarray([rlz[var] for var in self._orderedVarsList])
      ## update _lambda and _gamma
      self._gamma = 1.0/np.sqrt(self.counter-self._burnIn+1.0)
      self._lambda = self._lambda * np.exp(self._gamma * (np.exp(alpha) - self._optAlpha))
      if self.counter % self._adaptiveInterval == 0:
        diff = orderedVarsVals - self._ensembleMean
        self._ensembleMean += self._gamma * diff
        self._ensembleCov += self._gamma * (np.outer(diff, diff)-self._ensembleCov)
        ## update proposal distribution
        size = len(self._ensembleMean)
        self._proposal = self.constructProposalDistribution(np.zeros(size), self._lambda*self._ensembleCov.ravel())

  def localStillReady(self, ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    ready = self._localReady and MCMC.localStillReady(self, ready)
    return ready
