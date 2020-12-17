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
from utils import utils,randomUtils,InputData, InputTypes
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
    self._lambda = None
    self._gamma = None
    self._ensembleMean = None
    self._ensembleCov = None
    self._orderedVars = OrderedDict() # ordered dict of variables that is used to construct proposal function
    self._orderedVarsList = []

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    MCMC.handleInput(self, paramInput)

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean Adaptive Metropolis is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    MCMC.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    totalNumVars = len(self.variables2distributionsMapping)
    ## compute initial gamma and lambda
    self._lambda = 2.38**2/totalNumVars
    self._gamma = 1.0/(self.counter+1.0)
    if totalNumVars != len(self.toBeCalibrated):
      self.raiseAnError(IOError, 'AdaptiveMetropolis can not handle "probabilityFunction" yet!',
                        'Please check your input and provide "distribution" instead of "probabilityFunction"!')
    if self._proposal:
      self.raiseAWarning('In AdaptiveMetropolis, "proposal" will be automatic generated!',
                         'The user provided proposal will not be used!')
    ## construct ordered variable list
    ## construct ensemble mean and covariance that will be used for proposal distribution
    ## ToDO: current structure only works for untruncated distribution
    self._ensembleMean = np.zeros(totalNumVars)
    self._ensembleCov = np.zeros((totalNumVars, totalNumVars))
    index = 0
    for distName, elementDict in self.distributions2variablesMapping.items():
      orderedVars = [k for k, v in sorted(elementDict.items(), key=lambda item: item[1])]
      self._orderedVars[distName] = orderedVars
      self._orderedVarsList.extend(orderedVars)
      dist = self.distDict[orderedVars[0]]
      if len(elementDict) == 1:
        mean = dist.untruncatedMean()
        sigma = dist.untruncatedStdDev()
        self._ensembleMean[index] = mean
        self._ensembleCov[index, index] = sigma**2
        ## update initial value
        var = orderedVars[0]
        if self._updateValues[var] is None:
          value = dist.rvs()
          self._updateValues[var] = value
        ## update index
        index += 1
      else:
        if dist.type != 'MultivariateNormal':
          self.raiseAnError(IOError, 'Only accept "MultivariateNormal" distribution, but got "{}"'.format(dist.type))
        mean = dist.mu
        cov = dist.covariance
        totDim = len(mean)
        cov = np.asarray(cov).reshape((totDim, totDim))
        self._ensembleMean[index:index+totDim] = mean
        self._ensembleCov[index:index+totDim, index:index+totDim] = cov
        ## update initial value
        value = dist.rvs()
        for i, var in enumerate(orderedVars):
          if self._updateValues[var] is None:
            self._updateValues[var] = value[i]
        ## update index
        index += totDim
    ## construct the proposal distribution for given mean and covariance
    self._proposal = self.constructProposalDistribution(self._ensembleMean, self._ensembleCov.ravel())

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
        self.values[var] = self._updateValues[var] + newVal[i]
    self._setProbabilities()

  def _setProbabilities(self):
    """
      Method to compute probability related information
      @ In, None
      @ Out, None
    """
    for distName, orderedVars in self._orderedVars.items():
      dist = self.distDict[orderedVars[0]]
      if len(orderedVars) == 1:
        var = orderedVars[0]
        value = self.values[var]
        self.inputInfo['SampledVarsPb'][var] = dist.pdf(value)
        self.inputInfo['ProbabilityWeight-' + var] = 1.
      else:
        value = [self.values[var] for var in orderedVars]
        for var in orderedVars:
          self.inputInfo['SampledVarsPb'][var] = dist.pdf(value)
          self.inputInfo['ProbabilityWeight-' + var] = 1.
    self.inputInfo['PointProbability'] = 1.0
    self.inputInfo['ProbabilityWeight' ] = 1.0
    self.inputInfo['SamplerType'] = 'Metropolis'


  ## unchanged, can be moved to MCMC base class
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
    self._localReady = True
    MCMC.localFinalizeActualSampling(self, jobObject, model, myInput)
    prefix = jobObject.getMetadata()['prefix']
    _, full = self._targetEvaluation.realization(matchDict={'prefix': prefix})
    rlz = dict((var, full[var]) for var in (list(self.toBeCalibrated.keys()) + [self._likelihood] + list(self.dependentSample.keys())))
    rlz['traceID'] = self.counter
    if self.counter == 1:
      self._addToSolutionExport(rlz)
      self._currentRlz = rlz
    if self.counter > 1:
      acceptable = self._useRealization(rlz, self._currentRlz)
      if acceptable:
        self._currentRlz = rlz
        self._addToSolutionExport(rlz)
        self._updateValues = dict((var, rlz[var]) for var in self._updateValues)
      else:
        self._addToSolutionExport(self._currentRlz)
        self._updateValues = dict((var, self._currentRlz[var]) for var in self._updateValues)
      self._updateAdaptiveParams(self._currentRlz)

  def _useRealization(self, newRlz, currentRlz):
    """
      Used to feedback the collected runs within the sampler
      @ In, newRlz, dict, new generated realization
      @ In, currentRlz, dict, the current existing realization
      @ Out, acceptable, bool, True if we accept the new sampled point
    """
    ## first compute acceptable probability vs. netLogLikelihood
    netLogPosterior = 0
    # compute net log prior
    for distName, orderedVars in self._orderedVars.items():
      dist = self.distDict[orderedVars[0]]
      if len(orderedVars) == 1:
        var = orderedVars[0]
        netLogPrior = dist.logPdf(newRlz[var]) - dist.logPdf(currentRlz[var])
      else:
        newVal = [newRlz[var] for var in orderedVars]
        currVal = [currentRlz[var] for var in orderedVars]
        netLogPrior = dist.logPdf(newVal) - dist.logPdf(currVal)
      netLogPosterior += netLogPrior
    if not self._logLikelihood:
      netLogLikelihood = np.log(newRlz[self._likelihood]) - np.log(currentRlz[self._likelihood])
    else:
      netLogLikelihood = newRlz[self._likelihood] - currentRlz[self._likelihood]
    netLogPosterior += netLogLikelihood
    acceptValue = np.log(self._acceptDist.rvs())
    acceptable = netLogPosterior > acceptValue
    return acceptable

  def _updateAdaptiveParams(self, rlz):
    """
      Used to feedback the collected runs within the sampler
      @ In, rlz, dict, the updated current existing realization
      @ Out, None
    """
    orderedVarsVals = np.asarray([rlz[var] for var in self._orderedVarsList])
    ## update _lambda
    self._gamma = 1.0/(self.counter+1.0)
    self._lambda = self._lambda * np.exp(self._gamma * (netLogPosterior - self._optAlpha))
    diff = orderedVarsVals - self._ensembleMean
    self._ensembleMean += self._gamma * diff
    self._ensembleCov += self._gamma * (np.outer(diff, diff)-self._ensembleCov)
    ## update proposal distribution
    self._proposal = self.constructProposalDistribution(self._ensembleMean, self._ensembleCov.ravel())

  def localStillReady(self, ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    ready = self._localReady and MCMC.localStillReady(self, ready)
    return ready
