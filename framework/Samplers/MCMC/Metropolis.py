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
  Markov Chain Monte Carlo
  This base class defines the principle methods required for MCMC

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
from utils import utils,randomUtils,InputData, InputTypes
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
    self._localReady = True # True if the submitted job finished
    self._currentRlz = None # dict stores the current realizations, i.e. {var: val}

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
        newVal = value + self._proposal[key].rvs()
        self.values[key] = newVal
        if key in self.distDict:
          self.inputInfo['SampledVarsPb'][key] = self.distDict[key].pdf(newVal)
        else:
          self.inputInfo['SampledVarsPb'][key] = self._priorFuns[key].evaluate("pdf", self.values)
        self.inputInfo['ProbabilityWeight-' + key] = 1.
    self.inputInfo['PointProbability'] = 1.0
    self.inputInfo['ProbabilityWeight' ] = 1.0
    self.inputInfo['SamplerType'] = 'Metroplis'

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

  def _useRealization(self, newRlz, currentRlz):
    """
      Used to feedback the collected runs within the sampler
      @ In, newRlz, dict, new generated realization
      @ In, currentRlz, dict, the current existing realization
      @ Out, acceptable, bool, True if we accept the new sampled point
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
    acceptValue = np.log(self._acceptDist.rvs())
    acceptable = netLogPosterior > acceptValue
    return acceptable

  def localStillReady(self, ready):
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    ready = self._localReady and MCMC.localStillReady(self, ready)
    return ready
