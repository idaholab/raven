# Copyright 2023 Battelle Energy Alliance, LLC
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
"""
# External Modules
from scipy.stats import norm
import numpy as np
# External Modules

# Internal Modules
import abc
from ...utils import utils, InputData, InputTypes
from .AcquisitionFunction import AcquisitionFunction
# Internal Modules

class ExpectedImprovement(AcquisitionFunction):
  """
    Provides class for the Expected Improvement (EI) acquisition function
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
    specs = super(ExpectedImprovement, cls).getInputSpecification()
    specs.description = r"""If this node is present within the acquisition node,
                        the expected improvement acqusition function is utilized.
                        This function is derived by applying Bayesian optimal decision making (Bellman's Principle of Optimality)
                        with a local reward utility function in conjunction with a one-step lookahead
                        horizon. The approach weighs both expected reward and likely reward with the
                        following expression (for minimization):
                        $EI(x) = (f_{opt}-\mu)\phi(\frac{f_{opt}-\mu}{s}) + s \Phi(\frac{f_{opt}-\mu}{s})$"""

    return specs

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
      @ Out, EI, float, expected improvement function value
    """
    # Need to retrieve current optimum point
    best = bayesianOptimizer._optPointHistory[0][-1][0]
    fopt = best[bayesianOptimizer._objectiveVar]

     # Need to convert array input "x" into dict point
    featurePoint = bayesianOptimizer.arrayToFeaturePoint(var)

    # Evaluate posterior mean and standard deviation
    mu, s = bayesianOptimizer._evaluateRegressionModel(featurePoint)

    # Is this evaluation vectorized?
    if vectorized:
      betaVec = np.divide(np.add(-mu, fopt),s)
      pdfVec = norm.pdf(betaVec)
      cdfVec = norm.cdf(betaVec)
      term1 = np.multiply(np.add(-mu, fopt), cdfVec)
      term2 = np.multiply(s, pdfVec)
      EI = np.add(term1, term2)
    else:
      # Breaking out components from closed-form of EI (GPR)
      # Definition of standard gaussian density function
      beta = (fopt - mu)/s
      pdf = norm.pdf(beta)
      # Standard normal cdf from scipy.stats
      cdf = norm.cdf(beta)
      # Definition of EI
      EI = ((fopt - mu) * cdf) + (s*pdf)

    return EI

  def gradient(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's gradient using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Expected Improvement gradient at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, EIGrad, float/array, EI gradient value
    """
    # TODO figure out best way to generalize kernel gradients first
    # # Evaluate gradients of posterior prediction equations
    # featurePoint = bayesianOptimizer.arrayToFeaturePoint(var)
    # dmu, dstd = bayesianOptimizer._evaluateRegressionGradient(featurePoint)
    return

  def hessian(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's hessian using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Expected Improvement hessian at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, EIHess, float/array, EI hessian value
    """
    return
