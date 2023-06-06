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
  Class for implementing Bayesian Optimization into the RAVEN framework
  auth: Anthoney Griffith
  date: May, 2023
"""
#External Modules------------------------------------------------------------------------------------
import copy
from collections import deque, defaultdict
import numpy as np
from smt.sampling_methods import LHS
import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib import cm
from mpl_toolkits import mplot3d
from scipy.stats import norm
import scipy.optimize as sciopt

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import InputData, InputTypes, mathUtils
from .RavenSampled import RavenSampled

#Internal Modules End--------------------------------------------------------------------------------


class BayesianOptimizer(RavenSampled):
  """
    Implements the Bayesian Optimization algorithm for cost function minimization within the RAVEN framework
  """


  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(BayesianOptimizer, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{BayesianOptimizer} optimizer is a method for black-box optimization.
                            This approach utilizes a surrogate model, in the form of a Gaussian Process Regression,
                            to find the global optima of expensive functions. Furthermore, this approach easily
                            incorporates noisy observations of the function. This approach tends to offer the
                            tradeoff of additional backend calculation (training regressions and selecting samples) in
                            favor of reducing the number of function or 'model' evaluations."""
    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, ok, list(str), list of acceptable variable names
    """
    # cannot be determined before run-time due to variables and prefixes.
    ok = super(BayesianOptimizer, cls).getSolutionExportVariableNames()
    new = {}
    # new = {'': 'the size of step taken in the normalized input space to arrive at each optimal point'}
    new['conv_{CONV}'] = 'status of each given convergence criteria'
    ok.update(new)
    return ok

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    RavenSampled.__init__(self)
    # TODO Figure out best way for tracking 'iterations', 'function evaluations', and 'steps'
    self._iteration = {}            # Tracks the optimization methods current iteration, DOES NOT INCLUDE INITIALIZATION
    self._varNameList = []          # List of string names for decision variables, allows for easier indexing down the line
    self._initialSampleSize = None  # Number of samples to build initial model with before applying acquisition (default is 5)
    self._trainingInputs = [{}]     # Dict of numpy arrays for each traj, values for inputs to actually evaluate the model and train the GPR on
    self._trainingTargets = []      # A list of function values for each trajectory from actually evaluating the model, used for training the GPR
    self._model = None              # Regression model used for Bayesian Decision making

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)
    self._model = paramInput.findFirst('Model').value

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    # FIXME currently BO assumes only one optimization 'trajectory'
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)

    meta = ['batchId']
    self.addMetaKeys(meta)
    self._initialSampleSize = len(self._initialValues)
    self.batch = self._initialSampleSize

    # Initialize model object and store within class
    for model in self.assemblerDict['Models']:
      modelName = model[2]
      if modelName == self._model:
        self._model = model[3]
    if self._model is None:
      self.raiseAnError(RuntimeError, f'No model was provided for Bayesian Optimizer. This method requires a ROM model.')
    elif self._model.subType not in ["GaussianProcessRegressor"]:
      self.raiseAnError(RuntimeError, f'Invalid model type was provided: {self._model.subType}. Bayesian Optimizer'
                                      f'currently only accepts the following: {["GaussianProcessRegressor"]}')

    # Initialize feature and target data set for conditioning regression model on
    # NOTE assuming that sampler/user provides at least one initial input
    init = self._initialValues[0]
    self._trainingTargets.append([])
    for varName, _ in init.items():
      self._trainingInputs[0][varName] = []
      if varName not in self._varNameList:
        self._varNameList.append(varName)

    # First step is to sample the model at all initial points from the sampler
    for _, point in enumerate(self._initialValues):
      self._iteration[0] = 0
      # Submitting each initial sample point to the sampler
      self._submitRun(point, 0, 0)

  def _submitRun(self, point, traj, step, moreInfo=None):
    """
      Submits a single run with associated info to the submission queue
      @ In, point, dict, point to submit
      @ In, traj, int, trajectory identifier
      @ In, step, int, iteration number identifier
      @ In, moreInfo, dict, optional, additional run-identifying information to track
      @ Out, None
    """
    info = {}
    if moreInfo is not None:
      info.update(moreInfo)
    info.update({ 'traj': traj,
                  'step': step,
                  'batchSize': self.batch})
    self.raiseADebug(f'Adding run to queue: {self.denormalizeData(point)} | {info}')
    self._submissionQueue.append((point, info))

  ###############
  # Run Methods #
  ###############
  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, xr.Datasheet or dict, realized realization (corrected for min-max)
      @ Out, None
    """
    # Checking if we have multiple samples to handle (should be initialization)
    # NOTE this will include parallel acquisition once it is implemented or at least should
    traj = info['traj']
    step = info['step'] + 1
    if not isinstance(rlz, dict):
      if step == 1:
        self.batch = 1 # FIXME when implementing parallel expected improvement, fix this
        self.raiseAMessage(f'Initialization data of dimension {self._initialSampleSize} received... '
                           f'Setting sample batch size to {self.batch}')
      else:
        self.raiseAMessage(f'Received next set of parallel samples for iteration: {self.getIteration(traj)}')
      # Add new inputs and model evaluations to the dataset
      for varName in list(self.toBeSampled):
        self._trainingInputs[traj][varName].extend(getattr(rlz, varName).values)
      self._trainingTargets[traj].extend(getattr(rlz, self._objectiveVar).values)
      self._resolveMultiSample(traj, rlz, info)
    elif isinstance(rlz, dict):
      self.raiseAMessage(f'Received next sample for iteration: {self.getIteration(traj)}')
      # Add new input and model evaluation to the dataset
      for varName in list(self.toBeSampled):
        self._trainingInputs[traj][varName].append(rlz[varName])
      self._trainingTargets[traj].append(rlz[self._objectiveVar])
      optVal = rlz[self._objectiveVar]
      self._resolveNewOptPoint(traj, rlz, optVal, info)

    # Generate posterior with training data
    self._trainRegressionModel(traj)
    # Use acquisition to select next point
    new_point = self._conductAcquisition()
    self._submitRun(new_point, traj, step)
    print(f'Model information: {self._model.supervisedContainer[0].model.kernel_}')
    self.incrementIteration(traj)

  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory identifier
      @ In, new, dict, new opt point
      @ In, old, dict, previous opt point
    """
    return

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    if old - new > 0:
      return True
    else:
      return False

  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()

  ###################
  # Utility Methods #
  ###################
  def incrementIteration(self, traj):
    """
      Increments the "generation" or "iteration" of an optimization algorithm.
      The definition of generation is algorithm-specific; this is a utility for tracking only.
      @ In, traj, int, identifier for trajectory
      @ Out, None
    """
    self._iteration[traj] += 1

  def getIteration(self, traj):
    """
      Provides the "generation" or "iteration" of an optimization algorithm.
      The definition of generation is algorithm-specific; this is a utility for tracking only.
      @ In, traj, int, identifier for trajectory
      @ Out, counter, int, iteration of the trajectory
    """
    return self._iteration[traj]

  # NOTE this method is for barebones validation, not currently intended for user access
  def plotPosteriorAndAcquisition(self):
    """
      Plots the posterior mean of the GP model conditioned on the sampled data and the acquistion function
      @ In , None
      @ Out, None
    """
    # Plot the model for fun
    xvec = np.linspace(0,1,100)
    yvec = np.linspace(0,1,100)
    X,Y = np.meshgrid(xvec,yvec)
    f_vals = np.empty((100,100))
    EI_vals = np.empty((100,100))
    for i in it.product(range(100),range(100)):
      f_vals[i[1],i[0]] = self._evaluateRegressionModel({'x':xvec[i[0]], 'y':xvec[i[1]]})[0]
      EI_vals[i[1],i[0]] = self.expectedImprovement(np.array([xvec[i[0]], yvec[i[1]]]))
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(X, Y, f_vals, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
    inputs = self.getInputs()
    ax.scatter3D(inputs[:,0], inputs[:,1], self._trainingTargets[0], color='green')
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('f(x,y)', fontsize=18)
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(X, Y, EI_vals, rstride=1, cstride=1, cmap=cm.viridis, edgecolor='none')
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('EI(x,y)', fontsize=18)
    plt.show()

  ###############################################
  # Temporary Methods for Bayesian Optimization #
  ###############################################
  def _trainRegressionModel(self, traj):
    """
      Reformats training data into form that ROM can handle
      @ In, traj, trajectory for training the model
      @ Out, None
    """
    # Build training set to feed to rom model
    # trainingSet = {}
    # for varName in list(self.toBeSampled):
    #   trainingSet[varName] = np.asarray(self._trainingInputs[traj][varName])
    # trainingSet[self._objectiveVar] = np.asarray(self._trainingTargets[traj])
    self._model.train(self._targetEvaluation)

  def _evaluateRegressionModel(self, featurePoint):
    """
      Evaluates GPR mean and standard deviation at a given input location
      @ In, featurePoint, dict, feature values to evaluate ROM
      @ Out, mu, ROM mean/prediction value at that point
      @ Out, std, ROM standard-deviation value at that point
    """
    # Evaluating the regression model
    featurePoint = self.denormalizeData(featurePoint)
    resultsDict = self._model.evaluate(featurePoint)
    if len(resultsDict) == 1:
      mu = resultsDict[self._objectiveVar][0]
      std = resultsDict[self._objectiveVar][1]
    else:
      mu = {}
      std = {}
      for objVar in list(resultsDict):
        mu[objVar] = resultsDict[objVar][0]
        std[objVar] = resultsDict[objVar][1]
    return mu, std

  def _conductAcquisition(self):
    """
      Maximizes the acquisition function to select next point to evaluate the model
      @ In, None
      @ Out, new_point, dict, new sample to evaluate
    """
    n = len(self._variableBounds)
    bounds = []
    for i in range(n):
      bounds.append((0,1))
    opt_func = lambda x: -1*self.expectedImprovement(x)
    res = sciopt.differential_evolution(opt_func, bounds, polish=True, maxiter=100, popsize=60, init='random')
    new_point = {}
    for index, varName in enumerate(self._varNameList):
      new_point[varName] = res.x[index]
    return new_point

  def _updateHyperparams(self):
    """
      Updates hyperparameters for SE using max LML
      @ In, None
      @ Out, None
    """
    # Function to optimize over
    opt_func = self.logMarginalLiklihood
    theta0 = [self._hyperParams['sigf'], self._hyperParams['l']]
    bounds = [(10e-3,10e5), (1e-3,1)]
    options = {'ftol':1e-10, 'maxiter':150, 'disp':False}
    res = sciopt.minimize(opt_func, theta0, method='SLSQP', jac=True, bounds=bounds, options=options)
    # Update the hyperparameters from selection
    self._hyperParams = {'sigf':res.x[0], 'l':res.x[1]}
    self.raiseADebug(f'Selected new hyperparameters for GPR model: {self._hyperParams}')

  # Acquisition Function
  def expectedImprovement(self, x):
    """
      Evaluates the expected improvement for the stored model
      @ In, x, np.array, input to evaluate EI at
      @ Out, EI, float, expected improvement at the given point
    """
    # Need to retrieve current optimum point
    best = self._optPointHistory[0][-1][0]
    f_opt = best[self._objectiveVar]

    # Need to convert array input "x" into dict point
    featurePoint = self.arrayToFeaturePoint(x)

    # Evaluate posterior mean and standard deviation
    mu, s = self._evaluateRegressionModel(featurePoint)

    # Breaking out components from closed-form of EI (GPR)
    # Definition of standard gaussian density function
    pdf = (1/(np.sqrt(2*np.pi))) * np.exp(-0.5*(((f_opt - mu)/s)**2))
    # Standard normal cdf from scipy.stats
    cdf = norm.cdf((f_opt - mu)/s)
    # Definition of EI
    EI = ((f_opt - mu) * cdf) + (s*pdf)

    return EI

  # Kernel
  def squaredExponential(self, x, x_prime):
    """
      Evaluates the squared exponential covariance function
      @ In, x, np.array, first input point
      @ In, x_prime, np.array, second input point
      @ Out, k, float, covariance between the two points
    """
    # The radius between the inputs are
    r = np.linalg.norm(np.subtract(x, x_prime))
    l = self._hyperParams['l']
    sigma_f = self._hyperParams['sigf']
    k = (sigma_f**2)*np.exp(-(r**2)/(2*(l**2)))
    return k

  def SEHyperparamGradient(self):
    """
      Evaluates the gradient of LML function wrt to the hyperparams (SE only)
      @ In, None
      @ Out, hyperGrad, np array, gradient of LML wrt hyperparams l and sigmaf
    """
    # Need to grab hyperparam values and input vectors
    sigma_f = self._hyperParams['sigf']
    l = self._hyperParams['l']
    inputs = self.getInputs()

    # Initializing derivatives matrices
    n = len(inputs)
    dK_dsigf = np.empty((n,n), dtype=float)
    dK_dl = np.empty((n,n), dtype=float)

    # Calculating covariance matrix for training
    for i in it.product(range(n),range(n)):
        x = inputs[i[0]]
        x_p = inputs[i[1]]
        var_norm = np.linalg.norm(np.subtract(x,x_p))**2
        dK_dsigf[i] = 2 * sigma_f * np.exp( (-1/ (2*(l**2)) ) * var_norm )
        dK_dl[i] = (sigma_f**2) * np.exp( (-1/ (2*(l**2)) ) * var_norm ) * ( ( 1/(l**3) ) * var_norm )

    # Gradient time
    K_inv = np.dot(np.linalg.inv(np.transpose(self._regressionModel['L'])), np.linalg.inv(self._regressionModel['L']))
    aa_t = np.outer(self._regressionModel['alpha'], np.transpose(self._regressionModel['alpha']))
    dsig = (1/2) * np.trace(np.dot((aa_t - K_inv), dK_dsigf))
    dl = (1/2) * np.trace(np.dot((aa_t - K_inv), dK_dl))
    hyperGrad = np.array([dsig, dl])
    return hyperGrad

  # LML selection
  def logMarginalLiklihood(self, theta):
    """
      Evaluates the LML and its gradient wrt to theta
      @ In, theta, ['sigf', 'l']
      @ Out, -LML, log-marginal likelihood
      @ Out, -hyperGrad, gradient of log-marginal likelihood wrt theta
    """
    # Update the hyperparams
    self._hyperParams['sigf'] = theta[0]
    self._hyperParams['l'] = theta[1]
    # Update GPR w/ new hyperparams
    self._trainRegressionModel()
    # Components necessary for evaluation
    y = self._trainingTargets[0]
    n = len(y)

    # Calculating second term in marginal likelihood
    term2 = 0
    for i in range(n):
        term2 += np.log(self._regressionModel['L'][i,i])
    # Similarly, using worst observation as prior mean here
    M = np.max(y)*np.ones(n)
    LML = ((-1/2) * np.dot(np.subtract(y,M), self._regressionModel['alpha'])) - (term2) - ((n/2) * np.log(2*np.pi))
    # LML = ((-1/2) * np.dot(y, self._regressionModel['alpha'])) - (term2) - ((n/2) * np.log(2*np.pi))
    hyperGrad = self.SEHyperparamGradient()
    # NOTE we want to maximize this function so we return the negative evaluation
    return -1*LML, np.multiply(-1, hyperGrad)

  # Training methods
  def buildTrainingCovariance(self):
    """
      Builds gram-matrix for conditioning the posterior
      @ In, None
      @ Out, None
    """
    # Want input data set to be easily to work with
    inputs = self.getInputs()
    n = len(inputs)
    K = np.empty((n,n), dtype=float)
    for i in it.product(range(n),range(n)):
      K[i] = self.squaredExponential(inputs[i[0]], inputs[i[1]])
    # Stabilizing cholesky decomposition
    eps = 1e-9 * self.squaredExponential(inputs[0], inputs[0])
    I = np.eye(n)
    K = np.add(K, np.multiply(eps,I))
    # Inverting the cholesky decomposition is more computationally efficient
    L = np.linalg.cholesky(K)
    self._regressionModel['L'] = L

  def calculateAlpha(self):
    """
      Computes alpha term necessary for linear combination form of GPR
      @ In, None
      @ Out, None
    """
    # This temp method is taking the prior mean to constant with value of worst observation
    y_training = np.subtract(self._trainingTargets[0], np.max(self._trainingTargets[0]))
    L = self._regressionModel['L']
    # Outer inverse term
    L1 = np.linalg.inv(np.transpose(L))
    # Inner inverse term
    L2 = np.linalg.inv(L)
    # Definition of alpha
    alpha = np.dot(L1, np.dot(L2, y_training))
    self._regressionModel['alpha'] = alpha

  # Utility
  def getInputs(self):
    """
      Converts training data attributes into something of use (np.array)
      @ In, None
      @ Out, inputs, numpy nd array,
    """
    inputs = np.empty((len(self._trainingTargets[0]), (len(self._trainingInputs[0]))))
    dummyIndex = 0
    for _, dataSet in self._trainingInputs[0].items():
      inputs[:, dummyIndex] = dataSet
      dummyIndex += 1
    return inputs

  def arrayToFeaturePoint(self, x):
    """
      Converts array input to featurePoint that model evaluation can read
      @ In, x, array input
      @ Out, featurePoint, input in dictionary form
    """
    # TODO how to properly track variable names
    dim = len(x)
    if dim != len(list(self.toBeSampled)):
      self.raiseAnError(RuntimeError, f'Dimension of input array supplied is {dim}, but the'
                                      f'dimension of the input space is {len(list(self.toBeSampled))}')

    # FIXME currently assumes indexing of array follows order of 'toBeSampled'
    index = 0
    featurePoint = {}
    for var in list(self.toBeSampled):
      featurePoint[var] = x[index]
      index += 1
    return featurePoint

  # * * * * * * * * * * * *
  # Constraint Handling
  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    return

  def _handleImplicitConstraints(self, previous):
    """
      Considers all implicit constraints
      @ In, previous, dict, NORMALIZED previous opt point
      @ Out, accept, bool, whether point was satisfied implicit constraints
    """
    normed = copy.deepcopy(previous)
    oldVal = normed[self._objectiveVar]
    normed.pop(self._objectiveVar, oldVal)
    denormed = self.denormalizeData(normed)
    denormed[self._objectiveVar] = oldVal
    accept = self._checkImpFunctionalConstraints(denormed)

    return accept

  def _checkImpFunctionalConstraints(self, previous):
    """
      Checks that provided point does not violate implicit functional constraints
      @ In, previous, dict, previous opt point (denormalized)
      @ Out, allOkay, bool, False if violations found else True
    """
    allOkay = True
    inputs = dict(previous)
    for impConstraint in self._impConstraintFunctions:
      okay = impConstraint.evaluate('implicitConstraint', inputs)
      if not okay:
        self.raiseADebug(f'Implicit constraint "{impConstraint.name}" was violated!')
        self.raiseADebug(' ... point:', previous)
      allOkay *= okay

    return bool(allOkay)

  # END constraint handling
  # * * * * * * * * * * * *

  def _resolveMultiSample(self, traj, rlz, info):
    """
      Handles case where multiple new datapoints are sampled.
      Applies resolution (acceptability, improvment, constraints, rejection) to each of the points
      @ In, traj, int, trajectory
      @ In, rlz, xr.Datasheet, realization of multiple points
      @ In, info, dict, information about the realization
      @ Out, None
    """
    # Break out info from xr rlz into the standard realization format
    singleRlz = {}
    rlzVars = list(rlz.variables)
    self.raiseADebug(f'Resolving multi-sample of size {info["batchSize"]}')
    for index in range(info['batchSize']):
      for varName in rlzVars:
        singleRlz[varName] = getattr(rlz, varName)[index].values
      optVal = singleRlz[self._objectiveVar]
      self._resolveNewOptPoint(traj, singleRlz, optVal, info)
      singleRlz = {} # FIXME is this necessary?
    self.raiseADebug(f'Multi-sample resolution completed')

  # support methods for _resolveNewOptPoint
  def _checkAcceptability(self, traj, opt, optVal, info):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
      @ Out, rejectReason, str, reject reason of opt point, or return None if accepted
    """
    if self._optPointHistory[traj]:
      old, _ = self._optPointHistory[traj][-1]
      oldVal = old[self._objectiveVar]
      # Check if new point is better
      if self._checkForImprovement(optVal, oldVal):
        acceptable = 'accepted'
        rejectReason = 'None'
      else:
        acceptable = 'rejected'
        rejectReason = 'No improvement'
    else: # no history
      old = None
      acceptable = 'first'
      rejectReason = 'None'
    return acceptable, old, rejectReason

  def _updateConvergence(self, traj, new, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, new, dict, new point
      @ In, old, dict, old point
      @ In, acceptable, str, condition of new point
      @ Out, converged, bool, True if converged on ANY criteria
    """
    self.raiseAWarning('No convergence methods aside from iteration budget has been implemented yet')
    if self.getIteration(traj) < self.limit:
      converged = False
    else:
      converged = True
    return converged

  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """
    return

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """
    return
