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
  Class for implementing Probability of Improvement
  auth: Anthoney Griffith (@grifaa)
  date: June, 2023
"""

# External Modules
import numpy as np
from scipy.stats import norm
# External Modules

# Internal Modules
import abc
from ...utils import utils, InputData, InputTypes
from .AcquisitionFunction import AcquisitionFunction
# Internal Modules

class ProbabilityOfImprovement(AcquisitionFunction):
  """
    Provides class for the Probability of Improvement (PoI) acquisition function
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
    specs = super(ProbabilityOfImprovement, cls).getInputSpecification()
    specs.description = r"""If this node is present within the acquisition node,
                        the probability of improvement acqusition function is utilized.
                        This function is derived by applying Bayesian optimal decision making (Bellman's Principle of Optimality)
                        with the probability of local reward utility function in conjunction with a one-step lookahead.
                        The approach weighs values the probability of improving the solution past some thresh-hold and has the
                        following expression (for minimization):
                        $PoI(x) = \frac{\tau - \mu}{\sigma}$"""
    specs.addSub(InputData.parameterInputFactory('epsilon', contentType=InputTypes.FloatType,
                                                 descr=r"""Defines the threshold for PoI via the equation:
                                                 $\tau = min \mu(\textbf{X}) - \epsilon$. The larger \epsilon is, the
                                                 more exploratory the algorithm and vice versa.""", default=1.0))
    specs.addSub(InputData.parameterInputFactory('rho', contentType=InputTypes.FloatType,
                                                 descr=r"""Provides a 'time-constant' for the Exploit and Explore transient settings.
                                                 Provides the period for the oscillate transient setting.""", default=1.0))
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
    self._epsilon = 1            # Value of epsilon currently
    self._baseEpsilon = 1        # Base value of epsilon (no transients applied)
    self._rho = 1                # Time constant or periodicity
    self._transient = None       # Method by which epsilon varies throughout optimization

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    super().handleInput(specs)
    settings, notFound = specs.findNodesAndExtractValues(['epsilon', 'transient', 'rho'])
    assert(not notFound)
    self._epsilon = settings['epsilon']
    self._baseEpsilon = settings['epsilon']
    self._rho = settings['rho']
    if settings['transient'] == 'Constant':
      self._transient = lambda N: self._baseEpsilon
    elif settings['transient'] == 'Exploit':
      self._transient = self.exploit
    elif settings['transient'] == 'Explore':
      self._transient = self.explore
    elif settings['transient'] == 'Oscillate':
      self._transient = self.oscillate
    elif settings['transient'] == 'DecayingOscillate':
      self._transient = self.decayingOscillate

  # PoI requires a correction in acquisition
  def conductAcquisition(self, bayesianOptimizer):
    """
      Selects new sample via optimizing the acquisition function
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, newPoint, dict, new point to sample the cost function at
    """
    newPoint = AcquisitionFunction.conductAcquisition(self, bayesianOptimizer)
    self._optValue = norm.cdf(self._optValue)
    return newPoint

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
      @ Out, poi, float, probability of improvement function value
    """
    # Need to retrieve current optimum point
    best = bayesianOptimizer._optPointHistory[0][-1][0]
    fopt = best[bayesianOptimizer._objectiveVar]

    # Need to convert array input "x" into dict point
    featurePoint = bayesianOptimizer.arrayToFeaturePoint(var)

    # Evaluate posterior mean and standard deviation
    mu, s = bayesianOptimizer._evaluateRegressionModel(featurePoint)

    # Retrieve iteration and set epsilon for threshhold
    self._transient(bayesianOptimizer._iteration[0])
    tau = fopt - self._epsilon

    # Is this evaluation vectorized?
    if vectorized:
      poi = np.subtract(tau, mu)
      poi = np.divide(poi, s)
    else:
      poi = (tau - mu) / s
    return poi

  def gradient(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's gradient using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Expected Improvement gradient at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, poiGrad, float/array, PoI gradient value
    """
    # NOTE assumes scikitlearn GPR currently
    # Need to convert array input "x" into dict point
    featurePoint = bayesianOptimizer.arrayToFeaturePoint(var)
    # Evaluate posterior mean and standard deviation
    mu, s = bayesianOptimizer._evaluateRegressionModel(featurePoint)
    meanGrad, stdGrad = bayesianOptimizer._model.supervisedContainer[0].evaluateGradients(var)

    # Need to retrieve current optimum point
    best = bayesianOptimizer._optPointHistory[0][-1][0]
    fopt = best[bayesianOptimizer._objectiveVar]

    # Retrieve iteration and set epsilon for threshhold
    self._transient(bayesianOptimizer._iteration[0])
    tau = fopt - self._epsilon

    # Gradient of PoI
    poiGrad = -s*np.transpose(meanGrad) - (tau-mu)*stdGrad
    poiGrad = poiGrad / (s**2)

    return poiGrad

  #####################
  # Transient Methods #
  #####################
  def exploit(self, iter):
    """
      Defines the transient method for exploit setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._epsilon = self._baseEpsilon * np.exp(-iter / self._rho)

  def explore(self, iter):
    """
      Defines the transient method for explore setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._epsilon = self._baseEpsilon * (1 - np.exp(-iter / self._rho))

  def oscillate(self, iter):
    """
      Defines the transient method for oscillate setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._epsilon = self._baseEpsilon * (np.sin((2 * np.pi * iter) / self._rho)**2)

  def decayingOscillate(self, iter):
    """
      Defines the transient method for oscillate setting
      @ In, iter, int, current iteration number
      @ Out, None
    """
    self._epsilon = self._baseEpsilon * (np.exp(-iter / (4 * self._rho))) * (np.sin((2 * np.pi * iter) / self._rho )**2)

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
    self._epsilon = self._baseEpsilon
