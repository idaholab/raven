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
  Metropolis Hastings Algorithm for Markov Chain Monte Carlo

  Created on June 26, 2020
  @author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import numpy as np
import copy
import abc
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .MCMC import MCMC
from ...utils import utils,randomUtils,InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class Metropolis(MCMC):
  """
    Metropolis Sampler
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
    inputSpecification = super(Metropolis, cls).getInputSpecification()
    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    MCMC.__init__(self)

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    MCMC.handleInput(self, paramInput)

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean MCMC is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    MCMC.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    if not self._correlated:
      for var in self._updateValues:
        if var in self.distDict:
          dist = self.distDict[var]
          dim = dist.getDimensionality()
          if dim != 1:
            self.raiseAnError(IOError, 'When "proposal" is used, only 1-dimensional probability distribution is allowed!',
                              'Please check your input for variable "{}".'.format(var),
                              'Please refer to adaptive Metropolis Sampler if the input variables are correlated!')
    else:
      self.raiseAnError(IOError, 'Multivariate case can not be handled by Metropolis, please consider adaptive Metropolis!')

    for var in self._updateValues:
      std = self._stdProposalDefault
      if var in self.distDict:
        dist = self.distDict[var]
        if var in self._proposal:
          self._proposal[var] = self.retrieveObjectFromAssemblerDict('proposal', self._proposal[var])
          distType = self._proposal[var].getDistType()
          dim = self._proposal[var].getDimensionality()
          if distType != 'Continuous':
            self.raiseAnError(IOError, 'variable "{}" requires continuous proposal distribution, but "{}" is provided!'.format(var, distType))
          if dim != 1:
            self.raiseAnError(IOError, 'When "proposal" is used, only 1-dimensional probability distribution is allowed!',
                              'Please check your input for variable "{}".'.format(var),
                              'Please refer to adaptive Metropolis Sampler if the input variables are correlated!')
        else:
          untrStdDev = dist.untruncatedStdDev()
          std *= untrStdDev
          propDist = self._availProposal['normal'](0.0, std)
          propDist.initializeDistribution()
          self._proposal[var] = propDist
          self.raiseAWarning('"proposal" is not provided for variable "{}", default normal distribution with std={} is used!'.format(var, std))
      else:
        if var in self._proposal:
          self._proposal[var] = self.retrieveObjectFromAssemblerDict('proposal', self._proposal[var])
        else:
          propDist = self._availProposal['normal'](0.0, std)
          propDist.initializeDistribution()
          self._proposal[var] = propDist
          self.raiseAWarning('"proposal" is not provided for variable "{}", default normal distribution with std={} is used!'.format(var, std))
      if self._updateValues[var] is None:
        value = dist.rvs()
        self._updateValues[var] = value

  def localGenerateInput(self, model, myInput):
    """
      Provides the next sample to take.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    if self.counter < 2:
      MCMC.localGenerateInput(self, model, myInput)
    else:
      self._localReady = False
      for key, value in self._updateValues.items():
        # update value based on proposal distribution
        newVal = value + self._proposal[key].rvs() * self._scaling
        self.values[key] = newVal
        if key in self.distDict:
          ## check the lowerBound and upperBound
          lowerBound = self.distDict[key].lowerBound
          upperBound = self.distDict[key].upperBound
          if lowerBound is not None and self.values[key] < lowerBound:
            self.values[key] = lowerBound
          if upperBound is not None and self.values[key] > upperBound:
            self.values[key] = upperBound
          self.inputInfo['SampledVarsPb'][key] = self.distDict[key].pdf(newVal)
        else:
          self.inputInfo['SampledVarsPb'][key] = self._priorFuns[key].evaluate("pdf", self.values)
        self.inputInfo['ProbabilityWeight-' + key] = 1.
    self.inputInfo['PointProbability'] = 1.0
    self.inputInfo['ProbabilityWeight' ] = 1.0
    self.inputInfo['SamplerType'] = 'Metropolis'
    self.inputInfo['LogPosterior'] = self.netLogPosterior
    self.inputInfo['AcceptRate'] = self._acceptRate

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

  def _useRealization(self, newRlz, currentRlz):
    """
      Used to feedback the collected runs within the sampler
      @ In, newRlz, dict, new generated realization
      @ In, currentRlz, dict, the current existing realization
      @ Out, netLogPosterior, float, the accepted probabilty
    """
    netLogPosterior = 0
    # compute net log prior
    for var in self._updateValues:
      newVal = newRlz[var]
      currVal = currentRlz[var]
      if var in self.distDict:
        dist = self.distDict[var]
        netLogPrior = dist.logPdf(newVal) - dist.logPdf(currVal)
      else:
        fun = self._priorFuns[var]
        netLogPrior = np.log(fun.evaluate("pdf", newRlz)) - np.log(fun.evaluate("pdf", currentRlz))
      netLogPosterior += netLogPrior
    if not self._logLikelihood:
      netLogLikelihood = np.log(newRlz[self._likelihood]) - np.log(currentRlz[self._likelihood])
    else:
      netLogLikelihood = newRlz[self._likelihood] - currentRlz[self._likelihood]
    netLogPosterior += netLogLikelihood
    netLogPosterior = min(0.0, netLogPosterior)
    return netLogPosterior

  def localStillReady(self, ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    ready = self._localReady and MCMC.localStillReady(self, ready)
    return ready
