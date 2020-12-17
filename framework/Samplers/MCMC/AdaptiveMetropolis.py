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
    self._lambda = {'i': None, 'i+1': None, 'counter':0} # Grow or shrink factor for covariance for step i and i+1
    self._mean = {'i': None, 'i+1': None} # mean for step i and i+1
    self._cov = {'i': None, 'i+1': None} # covariance for step i and i+1
    self._gamma = {'i': None, 'i+1': None}


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

    if self.proposal:
      self.raiseAWarning('In AdaptiveMetropolis, "proposal" will be automatic generated!',
                         'The user provided proposal will not be used!')

    ## Define proposal distribution


    ## generate initial values for self._updateValues

    for var in self._updateValues:
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
        newStd = 2.38 * untrStdDev # see Andrieu-Thoms2008
        propDist = self._availProposal['normal'](0.0, newStd)
        propDist.initializeDistribution()
        self._proposal[var] = propDist

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
