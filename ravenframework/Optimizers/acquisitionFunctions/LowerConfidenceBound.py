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
  Class for implementing Lower Confidence Bound
  auth: Anthoney Griffith (@grifaa)
  date: June, 2023
"""

# External Modules
from scipy.special import ndtri
import numpy as np
# External Modules

# Internal Modules
import abc
from ...utils import utils, InputData, InputTypes
from .AcquisitionFunction import AcquisitionFunction
# Internal Modules

class LowerConfidenceBound(AcquisitionFunction):
  """
    Provides class for the Lower Confidence Bound (LCB) acquisition function
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, None
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(LowerConfidenceBound, cls).getInputSpecification()
    specs.description = r"""If this node is present within the acquisition node,
                        the lower confidence bound acqusition function is utilized.
                        This function is derived by applying optimistic decision making in the infinite-armed bandit problem.
                        The approach assumes the model is conservative and values optimism through the following equation (for minimization):
                        $LCB(x) = \mu - \beta \sigma$, where $\beta = \Phi^{-1}(\pi)"""
    specs.addSub(InputData.parameterInputFactory('pi', contentType=InputTypes.FloatType,
                                                 descr=r"""Parameter that determines the lower confidence bound. Must be
                                                 between 0 and 1.""", default=0.98))
    specs.addSub(InputData.parameterInputFactory('rho', contentType=InputTypes.FloatType,
                                                 descr=r"""Provides a 'time-constant' for the Exploit and Explore transient settings.
                                                 Provides the period for the oscillate transient settings.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory('transient', contentType=InputTypes.makeEnumType("transient", "transientType",
                                                 ['Constant', 'Exploit', 'Explore', 'Oscillate', 'DecayingOscillate']),
                                                 descr=r"""Determines how the threshold \tau changes as optimization progresses.
                                                 \begin{itemize}
                                                 \item Constant: \epsilon remains the provided value.
                                                 \item Exploit: \epsilon exponentially decays to 0 encouraging exploitation.
                                                 \item Explore: \epsilon exponentially grows from 0 to provided value.
                                                 \item Oscillate: \epsilon varies between 0 and provided value.
                                                 \item DecayingOscillate: \epsilon oscillates and decays, driving to exploitation.
                                                 \end{itemize}""", default='Constant'))
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    super().__init__()
    self._pi = 0.5               # Value of pi currently
    self._basePi = 0.5           # Base value of pi (no transients applied)
    self._rho = 1                # Time constant or periodicity
    self._transient = None       # Method by which epsilon varies throughout optimization

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    super().handleInput(specs)
    settings, notFound = specs.findNodesAndExtractValues(['pi', 'transient', 'rho'])
    assert(not notFound)
    self._pi = settings['pi']
    self._basePi = settings['pi']
    assert(self._pi >= 0)
    assert(self._pi <= 1)
    self._rho = settings['rho']
    if settings['transient'] == 'Constant':
      self._transient = lambda N: self._basePi
    elif settings['transient'] == 'Exploit':
      self._transient = self.exploit
    elif settings['transient'] == 'Explore':
      self._transient = self.explore
    elif settings['transient'] == 'Oscillate':
      self._transient = self.oscillate
    elif settings['transient'] == 'DecayingOscillate':
      self._transient = self.decayingOscillate

  ######################
  # Evaluation Methods #
  ######################
  def evaluate(self, var, bayesianOptimizer, vectorized=False):
    """
      Evaluates acquisition function using the current BO instance
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Expected Improvement at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ In, vectorized, bool, whether the evaluation should be vectorized or not (useful for differential evolution)
      @ Out, lcb, float, lower confidence bound function value
    """
    # Need to convert array input "x" into dict point
    featurePoint = bayesianOptimizer.arrayToFeaturePoint(var)

    # Evaluate posterior mean and standard deviation
    mu, s = bayesianOptimizer._evaluateRegressionModel(featurePoint)

    # Retrieve iteration and set pi for quantile function
    self._transient(bayesianOptimizer._iteration[0])
    beta = ndtri(self._pi)

    # Is this evaluation vectorized?
    if vectorized:
      lcb = np.add(-1*mu, beta*s)
    else:
      lcb = -mu + beta*s
    return lcb

  def gradient(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's gradient using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Expected Improvement gradient at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, lcbGrad, float/array, LCB gradient value
    """
    # NOTE assumes scikitlearn GPR currently
    meanGrad, stdGrad = bayesianOptimizer._model.supervisedContainer[0].evaluateGradients(var)

    # Retrieve iteration and set epsilon for threshhold
    self._transient(bayesianOptimizer._iteration[0])
    beta = ndtri(self._pi)

    # Gradient of LCB
    lcbGrad = -1*np.transpose(meanGrad) + beta*stdGrad

    return lcbGrad

  #####################
  # Transient Methods #
  #####################
  def exploit(self, iter):
    """
      Defines the transient method for exploit setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._pi = self._basePi * np.exp(-iter / self._rho)

  def explore(self, iter):
    """
      Defines the transient method for explore setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._pi = self._basePi * (1 - np.exp(-iter / self._rho))

  def oscillate(self, iter):
    """
      Defines the transient method for oscillate setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._pi = self._basePi * (np.sin((2 * np.pi * iter) / self._rho)**2)

  def decayingOscillate(self, iter):
    """
      Defines the transient method for oscillate setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._pi = self._basePi * (np.exp(-iter / (4 * self._rho))) * (np.sin((2 * np.pi * iter) / self._rho)**2)

  # convergence
  def _converged(self, bayesianOptimizer):
    """
      Lower Confidence bound has a different style of convergence
      @ In, bayesianOptimizer, BayesianOptimizer object, instance of BayesianOptimizer class
      @ Out, converged, bool, has the optimizer converged on acquisition?
    """
    if self._optValue is None:
      converged = False
      return converged
    optDiff = np.absolute(-1*self._optValue - bayesianOptimizer._optPointHistory[0][-1][0][bayesianOptimizer._objectiveVar])
    optDiff /= np.absolute(bayesianOptimizer._optPointHistory[0][-1][0][bayesianOptimizer._objectiveVar])
    if optDiff <= bayesianOptimizer._acquisitionConv:
      converged = True
    else:
      converged = False
    return converged

  ###################
  # Utility Methods #
  ###################
  def flush(self):
    """
      Reset Acquisition Function attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self._pi = self._basePi
