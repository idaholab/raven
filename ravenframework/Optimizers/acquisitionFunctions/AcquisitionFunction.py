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
  Class for implementing acquisition functions for Bayesian Optimization
  auth: Anthoney Griffith (@grifaa)
  date: June, 2023
"""

# External Modules
import scipy.optimize as sciopt
from smt.sampling_methods import LHS
import numpy as np
import numdifftools as nd
import copy
# External Modules

# Internal Modules
import abc
from ...utils import utils, InputData, InputTypes
# Internal Modules

class AcquisitionFunction(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    Provides Base class for acquisition functions. Holds general methods for
    optimization of the acquisition functions
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
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    specs.description = 'Base class of acquisition functions for Bayesian Optimizer.'
    specs.addSub(InputData.parameterInputFactory('optimizationMethod', contentType=InputTypes.makeEnumType("optimizationMethod", "optimizationMethodType",
                                                 ['differentialEvolution','slsqp']),
                                                 descr=r"""String to specify routine used for the optimization of the acquisition function.
                                                 Acceptable options include: ('differentialEvolution', 'slsqp'). \default{'differentialEvolution'}.
                                                 \begin{itemize}
                                                 \item Differential Evolution: A style of evolutionary algorithm, which specializes in floating point
                                                 representations of decision variables. Works similar to its parent algorithm Genetic Algorithm
                                                 \item SLSQP: A Sequential Least Squares algorithm, which uses an BFGS update and
                                                 Lawson and Hanson’s NNLS nonlinear least-squares solver.""", default='differentialEvolution'))
    specs.addSub(InputData.parameterInputFactory('seedingCount', contentType=InputTypes.IntegerType,
                                                 descr=r"""If the method is gradient based or typically handled with singular
                                                 decisions (ex. slsqp approximates a quadratic program using the gradient), this number
                                                 represents the number of trajectories for a multi-start variant (default=2N).
                                                 N is the dimension of the input space.
                                                 If the method works on populations (ex. differential evolution simulates
                                                 natural selection on a population),
                                                 the number represents the population size (default=10N)."""))
    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, vars, dict, acceptable variable names and descriptions
    """
    return {}

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self._optMethod = None                                                     # Method used to optimize acquisition function for sample selection
    self._seedingCount = 0                                                     # For multi-start gradient methods, the number of starting points and the population size for differential evolution
    self._dim = None                                                           # Dimension of the input space
    self._bounds = []                                                          # List of tuples for bounds that scipy optimizers use
    self._optValue = None                                                      # Value of the acquisition function at the recommended sample
    self._constraints = None                                                   # Scipy optimizer constraint object for applying explicit constraints
    self._polish = True                                                        # Polish setting for Differential Evolution
    self._maxiter = 100                                                        # max iter setting for Differential Evolution
    self._tol = 1e-1                                                           # tolerance setting for Differential Evolution
    self._init = 'sobol'                                                       # init setting for Differential Evolution
    self._vectorized = True                                                    # vectorized setting for Differential Evolution
    self._slsqpOptions = options = {'ftol':1e-10, 'maxiter':200, 'disp':False} # Options dict for slsqp optimizer

  def handleInput(self, specs):
    """
      Read input specs
      @ In, specs, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    settings, notFound = specs.findNodesAndExtractValues(['optimizationMethod', 'seedingCount'])
    # If no user provided setting for opt method and seeding count, use default
    if 'optimizationMethod' in notFound:
      self._optMethod = 'differentialEvolution'
    else:
      self._optMethod = settings['optimizationMethod']
    if 'seedingCount' in notFound:
      if self._optMethod == 'differentialEvolution':
        self._seedingCount = 10*self._dim
      else:
        self._seedingCount = 2*self._dim
    else:
      self._seedingCount = settings['seedingCount']

  def initialize(self):
    """
      After construction, finishes initialization of this acquisition function.
      @ In, None
      @ Out, None
    """
    # Input space is normalized, thus building the bounds is simple
    for i in range(self._dim):
      self._bounds.append((0,1))

  def conductAcquisition(self, bayesianOptimizer):
    """
      Selects new sample via optimizing the acquisition function
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, newPoint, dict, new point to sample the cost function at
    """
    # Depending on the optimization method, the cost function should be defined differently
    if self._optMethod == 'differentialEvolution':
      # NOTE -1 is to enforce maximization of the positive function
      optFunc = lambda var: -1*self.evaluate(var, bayesianOptimizer, vectorized=True)
      if self._constraints == None:
        res = sciopt.differential_evolution(optFunc, bounds=self._bounds, polish=self._polish, maxiter=self._maxiter, tol=self._tol,
                                            popsize=self._seedingCount, init=self._init, vectorized=self._vectorized, seed=bayesianOptimizer._seed)
      else:
        res = sciopt.differential_evolution(optFunc, bounds=self._bounds, polish=self._polish, maxiter=self._maxiter, tol=self._tol,
                                            popsize=self._seedingCount, init=self._init, vectorized=self._vectorized , constraints=self._constraints, seed=bayesianOptimizer._seed)
    elif self._optMethod == 'slsqp':
      optFunc = lambda var: (-1*self.evaluate(var, bayesianOptimizer),
                             -1*self.gradient(var, bayesianOptimizer))
      # Need to sample seeding points for multi-start slsqp
      limits = np.array(self._bounds)
      # NOTE one of our seeds will always come from the current recommended solution (best point)
      samplingCount = self._seedingCount - 1
      if samplingCount <= 1:
        sampler = LHS(xlimits=limits, criterion='center', random_state=bayesianOptimizer._seed)
      else:
        sampler = LHS(xlimits=limits, criterion='cm', random_state=bayesianOptimizer._seed)
      initSamples = sampler(samplingCount)
      best = bayesianOptimizer._optPointHistory[0][-1][0]
      # Need to convert 'best point' and add to init array
      tempArray = np.empty((1,self._dim))
      for index, varName in enumerate(list(bayesianOptimizer.toBeSampled)):
        tempArray[0,index] = best[varName]
      initSamples = np.concatenate((initSamples,tempArray),axis=0)

      res = None
      for x0 in initSamples:
        if self._constraints == None:
          result = sciopt.minimize(optFunc, x0, method='SLSQP', jac=True, bounds=self._bounds,
                                   options=self._slsqpOptions)
        else:
          result = sciopt.minimize(optFunc, x0, method='SLSQP', jac=True, bounds=self._bounds,
                                    options=self._slsqpOptions, constraints=self._constraints)
        if res is None:
          res = result
        elif result.fun < res.fun:
          res = result
    else:
      bayesianOptimizer.raiseAnError(RuntimeError, 'Currently only accepts differential evolution. Other methods still under construction')
    self._optValue = -1*res.fun
    newPoint = bayesianOptimizer.arrayToFeaturePoint(res.x)
    return newPoint

  ### TESTING METHOD ###
  def testGradients(self, bayesianOptimizer, nSamples):
    """
      Numerically validates gradients of acquisition functions
      @ In, bayesianOptimizer, instance of BayesianOptimizer, instance of the optimizer in its current state
      @ In, nSamples, int, number points to numerically test the analytical gradient
      @ Out, maxDiff, float, largest descrepancy found between numerical and analytical gradients
    """
    evalMethod = lambda var: self.evaluate(var, bayesianOptimizer)[0]
    funGrad = nd.Gradient(evalMethod, step=0.0001, order=4)
    diffVector = np.empty(nSamples)
    limits = np.array(self._bounds)
    sampler = LHS(xlimits=limits, criterion='cm', random_state=42)
    initSamples = sampler(nSamples)
    for i in range(nSamples):
      xI = initSamples[i,:]
      analytic = self.gradient(xI, bayesianOptimizer)[0]
      numeric = funGrad(xI)
      if not np.allclose(np.zeros(self._dim), analytic) and not np.allclose(np.zeros(self._dim), numeric):
        gradDiff = np.subtract(analytic/np.linalg.norm(analytic), numeric/np.linalg.norm(numeric))
      else:
        gradDiff = np.subtract(analytic, numeric)
      diffVector[i] = np.linalg.norm(gradDiff)
    maxDiff = np.max(diffVector)
    return maxDiff

  ######################
  # Evaluation Methods #
  ######################
  @abc.abstractmethod
  def evaluate(self, var, bayesianOptimizer, vectorized=False):
    """
      Evaluates acquisition function using the current BO instance
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Acquisition Function at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ In, vectorized, bool, whether the evaluation should be vectorized or not (useful for differential evolution)
      @ Out, acqValue, float/array, acquisition function value
    """
    pass

  @abc.abstractmethod
  def gradient(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's gradient using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Acquisition Function gradient at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, dacqValue, float/array, acquisition function gradient value
    """
    pass

  def hessian(self, var, bayesianOptimizer):
    """
      Evaluates acquisition function's hessian using the current BO instance/ROM
      Should be overwritten by specific acquisition functions
      @ In, var, np.array, input to evaluate Acquisition Function hessian at
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to model and evaluation method
      @ Out, ddacqValue, float/array, acquisition function hessian value
    """
    bayesianOptimizer.raiseAnError(NotImplementedError,'Hessian is not yet developed for this acqusition function')

  def buildConstraint(self, bayesianOptimizer):
    """
      Builds explicit constraints to enforce in acquisition
      @ In, bayesianOptimizer, instance of the BayesianOptimizer cls, provides access to constraint evaluation
      @ Out, None
    """
    # Copy to avoid overwriting
    baye = copy.copy(bayesianOptimizer)
    # Generating form that works with scipy.minimize
    self._constraints = []
    for constraint in baye._constraintFunctions:
      constraintFun = lambda var: np.array([constraint.evaluate('constrain', baye.denormalizeData(baye.arrayToFeaturePoint(var)))])
      nlc = sciopt.NonlinearConstraint(constraintFun, 0, np.inf)
      self._constraints.append(nlc)

  def needDenormalized(self):
    """
      Determines if this algorithm needs denormalized input spaces
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    return False

  def updateSolutionExport(self):
    """
      Prints information to the solution export.
      @ In, None
      @ Out, info, dict, realization of data to go in the solutionExport object
    """
    # Returning acquisition value post optimization
    info = {'acquisition':self._optValue}
    self._optValue = None # Resetting
    return info

  def _converged(self, bayesianOptimizer):
    """
      Specific Acquisition functions may want to overload this method.
      Checks for convergence on acquisition.
      @ In, bayesianOptimizer, instance of BayesianOptimizer class
      @ Out, converged, bool, has the optimizer converged on acquisition?
    """
    if self._optValue is None:
      converged = False
    elif self._optValue <= bayesianOptimizer._acquisitionConv:
      converged = True
    else:
      converged = False
    return converged

  def _recommendSolution(self, bayesianOptimizer):
    """
      Specific Acquisition functions may want to overload this method
      Uses current predictive model to recommend a solution point
      @ In, bayesianOptimizer, BayesianOptimizer object, instance of BayesianOptimizer class
      @ Out, muStar, float, recommended solution value
      @ Out, xStar, dict, point associated with muStar for the current data
      @ Out, stdStar, float, standard deviation of model prediction at xStar
    """
    # Pulling input data from BO instance
    trainingInputs = copy.copy(bayesianOptimizer._trainingInputs[0])
    for varName, array in trainingInputs.items():
      trainingInputs[varName] = np.asarray(array)
    # Evaluating the model at all training points
    modelEvaluation = bayesianOptimizer._evaluateRegressionModel(trainingInputs)
    # Evaluating constraints at all training points
    invalidIndices = []
    if self._constraints is not None:
      arrayTrainingInputs = bayesianOptimizer.featurePointToArray(trainingInputs)
      for constraint in self._constraints:
        constraintArray = constraint.fun(arrayTrainingInputs)
        invalidArray = np.less(constraintArray, np.zeros(constraintArray.shape))
        invalidWhere = np.where(invalidArray[0])
        for index in invalidWhere[0]:
          invalidIndices.append(index)
    # Pulling mean and std out of evaluation to operate on array structure
    muVec = modelEvaluation[0]
    stdVec = modelEvaluation[1]
    # Removing values at locations where constraint violation has occurred
    muVec = np.delete(muVec, invalidIndices)
    stdVec = np.delete(stdVec, invalidIndices)
    for varName in list(trainingInputs):
      trainingInputs[varName] = np.delete(trainingInputs[varName], invalidIndices)
    # Retrieving best mean value within training set locations, need index for retrieving other values
    muStar = np.min(muVec)
    minDex = np.argmin(muVec)
    stdStar = stdVec[minDex]
    # Retrieving location of recommended solution
    xStar = {}
    for varName in list(trainingInputs):
      xStar[varName] = trainingInputs[varName][minDex]
    return muStar, xStar, stdStar

  ###################
  # Utility Methods #
  ###################
  def flush(self):
    """
      Reset Acquisition Function attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    self._bounds = []
    self._optValue = None
