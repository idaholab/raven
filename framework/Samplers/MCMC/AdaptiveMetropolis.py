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
from collection import OrderedDict
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
    if self.proposal:
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
    self.proposal = self.constructProposalDistribution(self._ensembleMean, self._ensembleCov.ravel())

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
    MCMC.localGenerateInput(self, model, myInput)

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

  def localStillReady(self, ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    ready = self._localReady and MCMC.localStillReady(self, ready)
    return ready
