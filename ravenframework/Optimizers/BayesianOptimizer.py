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
  auth: Anthoney Griffith (@grifaa)
  date: May, 2023
"""
#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
from smt.sampling_methods import LHS
import scipy.optimize as sciopt
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import InputData, InputTypes, mathUtils
from .RavenSampled import RavenSampled
from .acquisitionFunctions import factory as acqFactory
#Internal Modules End--------------------------------------------------------------------------------


class BayesianOptimizer(RavenSampled):
  """
    Implements the Bayesian Optimization algorithm for cost function minimization within the RAVEN framework
  """
  convergenceOptions = {'acquisition': r"""Provides convergence criteria in terms of the value of the acquisition
                                       function at a given iteration. If the value falls below a provided threshhold,
                                       the optimizer is considered converged; however, it is recommended to pair this
                                       criteria with persistance. Default is 1e-8"""}

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

    # Acquisition function
    acqu = InputData.parameterInputFactory('Acquisition', strictMode=True,
        printPriority=106,
        descr=r"""A required node for specifying the details about the acquisition function
              used in the policy of Bayesian Optimization.""")

    # Pulling specs for each acquisition function option
    for option in acqFactory.knownTypes():
      subSpecs = acqFactory.returnClass(option).getInputSpecification()
      acqu.addSub(subSpecs)
    specs.addSub(acqu)

    # Model selection
    modelSelect = InputData.parameterInputFactory('ModelSelection', strictMode=False,
        printPriority=106,
        descr=r"""An optional node allowing the user to specify the details of model selection.
              For example, the manner in which hyperparameters are selected for the GPR model.""")
    modelSelect.addSub(InputData.parameterInputFactory('Duration', contentType=InputTypes.IntegerType,
        descr=r"""Number of iterations between each reselection of the model. Default is 1"""))
    modelSelect.addSub(InputData.parameterInputFactory('Method', contentType=InputTypes.makeEnumType("Method", "MethodType", ['External', 'Internal', 'Average']),
        descr=r"""Determines methodology for selecting the model. This methodology is applied
              after every duration length.
              \begin{itemize}
                \item External, uses whatever method the model has internal to itself
                \item Internal, selects the MAP point of the model using slsqp
                \item Average, Approximate marginalization over model space
              \end{itemize}. Default is External"""))
    specs.addSub(modelSelect)

    # Convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the desired convergence criteria for the optimization algorithm.
              Note that convergence is met when any one of the convergence criteria is met. If no convergence
              criteria are given, then the defaults are used.""")
    specs.addSub(conv)
    for name, descr in cls.convergenceOptions.items():
      conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType,descr=descr,printPriority=108  ))
    conv.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType,
        printPriority=300,
        descr=r"""provides the number of consecutive times convergence should be reached before a trajectory
              is considered fully converged. This helps in preventing early false convergence. Default is 5 (BO specific)"""))
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
    new['conv_{CONV}'] = 'status of each given convergence criteria'
    new['acquisition'] = 'value of acquisition at each iteration'
    # NOTE both are within context of normalized space
    new['radiusFromBest'] = 'radius of current point from current best point'
    new['radiusFromLast'] = 'radius of current point from previous point'
    new['solutionValue'] = 'Expected value of objective var at recommended solution point'
    # new['recommendedSolution'] = 'Location of recommended solution'
    new['solutionDeviation'] = 'Standard deviation of recommended solution'
    new['evaluationCount'] = 'Number of function evaluations up to current iteration'
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
    self._iteration = {}              # Tracks the optimization methods current iteration, DOES NOT INCLUDE INITIALIZATION
    self._initialSampleSize = None    # Number of samples to build initial model with before applying acquisition (default is 5)
    self._trainingInputs = [{}]       # Dict of numpy arrays for each traj, values for inputs to actually evaluate the model and train the GPR on
    self._trainingTargets = []        # A list of function values for each trajectory from actually evaluating the model, used for training the GPR
    self._model = None                # Regression model used for Bayesian Decision making
    self._acquFunction = None         # Acquisition function object used in optimization
    self._modelSelection = 'External' # Method for conducting model selection
    self._modelDuration = 1           # Number of iterations between model updates
    self._acquisitionConv = 1e-8      # Value for acquisition convergence criteria
    self._convergenceInfo = {}        # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = 5     # consecutive persistence required to mark convergence
    self._expectedOptVal = None       # Expected value of fopt, in other words, muopt
    self._optValSigma = None          # Standard deviations at expected solution, confidence of solution
    self._expectedSolution = None     # Decision variable values at expected solution
    self._evaluationCount = 0         # Number of function/model calls

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)
    # Model (GPR)
    self._model = paramInput.findFirst('Model').value

    # Acquisition function
    acquNode = paramInput.findFirst('Acquisition')
    if len(acquNode.subparts) != 1:
      self.raiseAnError('The <Acquisition> node requires exactly one acquisition function! Choose from: ', acqFactory.knownTypes())
    acquType = acquNode.subparts[0].getName()
    self._acquFunction = acqFactory.returnInstance(acquType)
    setattr(self._acquFunction, 'N', len(list(self.toBeSampled)))
    self._acquFunction.handleInput(acquNode.subparts[0])

    # Model Selection
    selectNode = paramInput.findFirst('ModelSelection')
    if selectNode is not None:
      self._modelDuration = selectNode.findFirst('Duration').value
      self._modelSelection = selectNode.findFirst('Method').value

    # Convergence
    convNode = paramInput.findFirst('convergence')
    if convNode is not None:
      acquConv = convNode.findFirst('acquisition')
      if acquConv is not None:
        self._acquisitionConv = acquConv.value
      persistence = convNode.findFirst('persistence')
      if persistence is not None:
        self._requiredPersistence = persistence.value

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    # FIXME currently BO assumes only one optimization 'trajectory'
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    self._convergenceInfo = {0:{'persistence':0, 'converged':False}}
    meta = ['batchId']
    self.addMetaKeys(meta)
    self._initialSampleSize = len(self._initialValues)
    self.batch = self._initialSampleSize

    # Building explicit contraints for acquisition if there are any
    if len(self._constraintFunctions) > 0:
      self._acquFunction.buildConstraint(self)

    # Initialize model object and store within class
    for model in self.assemblerDict['Models']:
      modelName = model[2]
      if modelName == self._model:
        self._model = model[3]
        break
    if self._model is None:
      self.raiseAnError(RuntimeError, 'No model was provided for Bayesian Optimizer. This method requires a ROM model.')
    elif self._model.subType not in ["GaussianProcessRegressor"]:
      self.raiseAnError(RuntimeError, f'Invalid model type was provided: {self._model.subType}. Bayesian Optimizer'
                                      f'currently only accepts the following: {["GaussianProcessRegressor"]}')
    self._setModelBounds()
    # NOTE Once again considering specifically sklearn's GPR
    optOption = self._model.supervisedContainer[0].model.get_params()['optimizer']
    # Avoiding unecessary model selection procedures that would tack on time
    if optOption is None and self._modelSelection == 'External':
      self._modelSelection = None
    else:
      self._model.supervisedContainer[0].model.set_params(optimizer=None)

    # Initialize the acquisition function
    self._acquFunction.initialize()
    # Closing extra trajectories
    for t in self._activeTraj[1:]:
      self._closeTrajectory(t, 'cancel', 'Currently BO is single trajectory', 0)

    # Initialize feature and target data set for conditioning regression model on
    # NOTE assuming that sampler/user provides at least one initial input
    # FIXME do we want to keep storage of features and targets, when targetEvaluation has this info?
    init = self._initialValues[0]
    self._trainingTargets.append([])
    for varName, _ in init.items():
      self._trainingInputs[0][varName] = []

    # First step is to sample the model at all initial points from the init sampler
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
        self.counter -= 1 #FIXME hacky way to make sure iterations are correctly counted
        self.raiseAMessage(f'Initialization data of dimension {self._initialSampleSize} received... '
                           f'Setting sample batch size to {self.batch}')
      else:
        self.raiseAMessage(f'Received next set of parallel samples for iteration: {self.getIteration(traj)}')
      # Add new inputs and model evaluations to the dataset
      for varName in list(self.toBeSampled):
        self._trainingInputs[traj][varName].extend(getattr(rlz, varName).values)
      self._trainingTargets[traj].extend(getattr(rlz, self._objectiveVar).values)
      # Generate posterior with training data
      self._generatePredictiveModel(traj)
      self._resolveMultiSample(traj, rlz, info)
    elif isinstance(rlz, dict):
      self.raiseAMessage(f'Received next sample for iteration: {self.getIteration(traj)}')
      # Add new input and model evaluation to the dataset
      for varName in list(self.toBeSampled):
        self._trainingInputs[traj][varName].append(rlz[varName])
      self._trainingTargets[traj].append(rlz[self._objectiveVar])
      # Generate posterior with training data
      self._generatePredictiveModel(traj)
      optVal = rlz[self._objectiveVar]
      self._resolveNewOptPoint(traj, rlz, optVal, info)

    # Use acquisition to select next point
    newPoint = self._acquFunction.conductAcquisition(self)
    self._submitRun(newPoint, traj, step)
    self.incrementIteration(traj)

  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self._acquFunction.flush()
    self._iteration = {}
    self._initialSampleSize = None
    self._trainingInputs = [{}]
    self._trainingTargets = []
    self._model = None
    return

  ###################
  # Utility Methods #
  ###################
  def arrayToFeaturePoint(self, x):
    """
      Converts array input to featurePoint that model evaluation can read
      @ In, x, array input
      @ Out, featurePoint, input in dictionary form
    """
    # TODO how to properly track variable names
    dim = x.shape[0]
    if dim != len(list(self.toBeSampled)):
        self.raiseAnError(RuntimeError, f'Dimension of input array supplied is {dim}, but the'
                                      f'dimension of the input space is {len(list(self.toBeSampled))}')
    featurePoint = {}
    # Receiving 2-D array of many inputs
    if len(x.shape) == 2:
      # FIXME currently assumes indexing of array follows order of 'toBeSampled'
      for index, var in enumerate(list(self.toBeSampled)):
        featurePoint[var] = x[index, :]
    # Receiving single input location
    elif len(x.shape) == 1:
      # FIXME currently assumes indexing of array follows order of 'toBeSampled'
      for index, var in enumerate(list(self.toBeSampled)):
        featurePoint[var] = x[index]
    else:
      self.raiseAnError(RuntimeError, f'Received invalid array shape in utility function: {len(x.shape)}-D')
    return featurePoint

  def featurePointToArray(self, featurePoint):
    """
      Converts featurePoint input to numpy array for easier operating
      @ In, featurePoint, dict, point in input space
      @ Out, x, np.array, array form of same input
    """
    # TODO same concerns as inverse class (see above)
    x = []
    for varName in list(featurePoint):
      x.append(featurePoint[varName])
    return np.asarray(x)

  def _addToSolutionExport(self, traj, rlz, acceptable):
    """
      Contributes additional entries to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, string, acceptability of opt point
      @ Out, toAdd, dict, additional entries
    """
    # Value of the acqusition function post-selection for this iteration
    toAdd = self._acquFunction.updateSolutionExport()
    # How close was this sample to the current best?
    pointDelta = {}
    # How close to previous sample?
    recentDelta = {}
    if self._iteration[traj] > 0:
      # If this point is the new opt, then it is zero away from itself
      if acceptable == 'accepted':
        bestDelta = 0
        for varName in list(self.toBeSampled):
          newPoint = rlz[varName]
          # -2 because newPoint is already appended to self._trainingInputs
          prevPoint = self._trainingInputs[traj][varName][-2]
          recentDelta[varName] = newPoint - prevPoint
        prevDelta = np.linalg.norm(self.featurePointToArray(recentDelta))
      else:
        for varName in list(self.toBeSampled):
          newPoint = rlz[varName]
          # -2 because newPoint is already appended to self._trainingInputs
          prevPoint = self._trainingInputs[traj][varName][-2]
          bestPoint = self._optPointHistory[traj][-1][0][varName]
          pointDelta[varName] = newPoint - bestPoint
          recentDelta[varName] = newPoint - prevPoint
        bestDelta = np.linalg.norm(self.featurePointToArray(pointDelta))
        prevDelta = np.linalg.norm(self.featurePointToArray(recentDelta))
    else:
      bestDelta = None
      prevDelta = None
    toAdd['radiusFromBest'] = bestDelta
    toAdd['radiusFromLast'] = prevDelta
    toAdd['solutionValue'] = self._expectedOptVal
    # toAdd['recommendedSolution'] = self._expectedSolution
    toAdd['solutionDeviation'] = self._optValSigma
    toAdd['evaluationCount'] = self._evaluationCount
    return toAdd

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

  #################################
  # Model Training and Evaluation #
  #################################
  def _setModelBounds(self):
    """
      Corrects hyperparameter bounds to account for normalized input space
      @ In, None
      @ Out, None
    """
    # NOTE This assumes scikitlearn GPR model
    hyperParamList = self._model.supervisedContainer[0].model.kernel.hyperparameters
    for hyperParam in hyperParamList:
      if 'length_scale' in hyperParam.name:
        hyperBound = hyperParam.name + '_bounds'
        self._model.supervisedContainer[0].model.kernel.set_params(**{hyperBound:(1e-5,1)})

  def _selectHyperparameters(self, restartCount=5):
    """
      Selects MAP model in model space, and is exclusively for sklearn GPR models
      @ In, restartCount, int, number of initial points to start optimization from
      @ Out, None
    """
    # Function to optimize, taking input as log-transformed hyperParameters
    lmlFunc = lambda logTheta: tuple([-1*x for x in self._model.supervisedContainer[0].model.log_marginal_likelihood(logTheta, eval_gradient=True, clone_kernel=False)])
    # Build bounds for optimization
    paramBounds = []
    for hyperParam in self._model.supervisedContainer[0].model.kernel.hyperparameters:
      for bound in hyperParam.bounds:
        paramBounds.append(tuple(np.log(bound)))

    # Restart locations include current parameter values
    sampler = LHS(xlimits=np.array(paramBounds), criterion='cm')
    initSamples = sampler(restartCount-1)
    initSamples = np.concatenate((initSamples, np.array([self._model.supervisedContainer[0].model.kernel.theta])))
    # Selecting MAP for each restart
    options = {'ftol':1e-10, 'maxiter':200, 'disp':False}
    res = None
    for x0 in initSamples:
      result = sciopt.minimize(lmlFunc, x0, method='SLSQP', jac=True, bounds=paramBounds, options=options)
      if res is None:
        res = result
      elif result.fun < res.fun:
        res = result
    newTheta = res.x
    newKernel = self._model.supervisedContainer[0].model.kernel_.clone_with_theta(newTheta)
    self._model.supervisedContainer[0].model.kernel = newKernel
    self._trainRegressionModel(0)

  def _trainRegressionModel(self, traj):
    """
      Reformats training data into form that ROM can handle
      @ In, traj, trajectory for training the model
      @ Out, None
    """
    # Build training set to feed to rom model
    trainingSet = {}
    for varName in list(self.toBeSampled):
      trainingSet[varName] = np.asarray(self._trainingInputs[traj][varName])
    trainingSet[self._objectiveVar] = np.asarray(self._trainingTargets[traj])
    self._model.train(trainingSet)
    # NOTE It would be preferrable to use targetEvaluation;
    # however, there does not appear a built in normalization method and as
    # consequence, it is preferrable to use the in class attributes to train the ROM
    # on our normalized space.
    # self._model.train(self._targetEvaluation)

  def _generatePredictiveModel(self, traj):
    """
      Applies model selection if necessary, otherwise just fits training data to predictive model
      @ In, traj, trajectory
      @ Out, None
    """
    # Generate posterior with training data
    if self._iteration[traj] % self._modelDuration == 0:
      if self._modelSelection == 'External':
        self._model.supervisedContainer[0].model.set_params(optimizer='fmin_l_bfgs_b')
        self._trainRegressionModel(traj)
        self._model.supervisedContainer[0].model.set_params(optimizer=None)
      elif self._modelSelection == 'Internal':
        self._trainRegressionModel(traj)
        restartCount = self._model.supervisedContainer[0].model.get_params()['n_restarts_optimizer']
        self._selectHyperparameters(restartCount=restartCount)
      elif self._modelSelection == 'Average':
        self.raiseAnError(RuntimeError, 'Model averaging is not yet available')
      else:
        self._trainRegressionModel(traj)
    else:
      self._trainRegressionModel(traj)

  def _evaluateRegressionModel(self, featurePoint):
    """
      Evaluates GPR mean and standard deviation at a given input location
      @ In, featurePoint, dict, feature values to evaluate ROM
      @ Out, mu, ROM mean/prediction value at that point
      @ Out, std, ROM standard-deviation value at that point
    """
    # Evaluating the regression model
    # featurePoint = self.denormalizeData(featurePoint) # NOTE this is because model is trained on unormalized 'targetEvaluation' dataobject
    resultsDict = self._model.evaluate(featurePoint)
    # NOTE only allowing single targets, needs to be fixed when multi-objective optimization is added
    mu = resultsDict[self._objectiveVar]
    std = resultsDict[self._objectiveVar+'_std']
    return mu, std

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

  def _resolveNewOptPoint(self, traj, rlz, optVal, info):
    """
      Consider and store a new optimal point
      @ In, traj, int, trajectory for this new point
      @ In, info, dict, identifying information about the realization
      @ In, rlz, xr.DataSet, batched realizations
      @ In, optVal, list of floats, values of objective variable
    """
    # Recommending solutions is based on the acqusition function's utility, typically local reward
    muStar, xStar, stdStar = self._acquFunction._recommendSolution(self)
    self._expectedOptVal = muStar
    self._optValSigma = stdStar
    self._expectedSolution = xStar
    self._evaluationCount += 1
    # RavenSampled._resolveNewOptPoint(self, traj, rlz, optVal, info)
    # FIXME, is the preferred method of
    self.raiseADebug('*' * 80)
    self.raiseADebug(f'Trajectory {traj} iteration {info["step"]} resolving new opt point ...')
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    # FIXME check implicit constraints? Function call, - Jia
    acceptable, old, rejectReason = self._checkAcceptability(traj, rlz, optVal, info)
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    # BO should consider convergence on every iteration and by extension Persistence
    self._updatePersistence(traj, converged, optVal)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.
    if self._writeSteps == 'every':
      self._updateSolutionExport(traj, rlz, acceptable, rejectReason)
    # Solution should reflect our expectation on the true latent function value,
    # This is equivalent to the observation when no corrupting noise is modeled/included
    currentPoint = {}
    for decisionVarName in list(self.toBeSampled):
      currentPoint[decisionVarName] = rlz[decisionVarName]
    rlz[self._objectiveVar] = self._evaluateRegressionModel(currentPoint)[0][0]
    self.raiseADebug('*' * 80)
    if acceptable in ['accepted', 'first']:
      # record history
      self._optPointHistory[traj].append((rlz, info))
      self._rerunsSinceAccept[traj] = 0
      # nothing else to do but wait for the grad points to be collected
    elif acceptable == 'rejected' and rejectReason == 'Not Recommended':
      # If the last recommended solution point is the same, update the expected function value
      if all(old[var] == xStar[var] for var in list(self.toBeSampled)):
        newEstimate = copy.copy(old)
        newEstimate[self._objectiveVar] = muStar
        self._optPointHistory[traj].append((newEstimate, info))
      else:
        newRealization = copy.copy(old)
        for var in list(self.toBeSampled):
          newRealization[var] = xStar[var]
        newRealization[self._objectiveVar] = muStar
    else:
      self.raiseAnError(f'Unrecognized acceptability: "{acceptable}"')

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
      # Need to check explicit constraints
      for constraint in self._constraintFunctions:
        constrained = constraint.evaluate('constrain', self.denormalizeData(opt))
        mostViolated = np.min(constrained)
        if mostViolated < 0:
          acceptable = 'rejected'
          rejectReason = 'Constraint Violation'
      # Is our new point the best point for the data available?
      if all(opt[var] == self._expectedSolution[var] for var in list(self.toBeSampled)):
        acceptable = 'accepted'
        rejectReason = None
      else:
        acceptable = 'rejected'
        rejectReason = 'Not Recommended'
    else: # no history
      old = None
      acceptable = 'first'
      rejectReason = 'None'
    return acceptable, old, rejectReason

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, None
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    return

  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory identifier
      @ In, new, dict, new opt point
      @ In, old, dict, previous opt point
      @ Out, converged, bool, has this traj converged or not?
    """
    converged = self._acquFunction._converged(self)
    self._convergenceInfo['converged'] = converged
    return converged

  def _updateConvergence(self, traj, new, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, new, dict, new point
      @ In, old, dict, old point
      @ In, acceptable, str, condition of new point
      @ Out, converged, bool, True if converged on ANY criteria
    """
    # No point in checking convergence if no feasible point has been found
    if len(self._optPointHistory[0]) == 0:
      converged = False
    elif self.getIteration(traj) < self.limit:
      converged = self.checkConvergence(traj, new, old)
      # # Should update persistence even if not accepted/improved point for BO
      # if acceptable not in ['accepted']:
      #   self._updatePersistence(traj, converged, self._optPointHistory[0][-1][0][self._objectiveVar])
    else:
      converged = True
    self._convergenceInfo['converged'] = converged
    return converged

  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """
    # update persistence
    if converged:
      self._convergenceInfo[traj]['persistence'] += 1
      self.raiseADebug(f'Trajectory {traj} has converged successfully {self._convergenceInfo[traj]["persistence"]} / {self._requiredPersistence} time(s)!')
      if self._convergenceInfo[traj]['persistence'] >= self._requiredPersistence:
        self._closeTrajectory(traj, 'converge', 'converged', optVal)
    else:
      self._convergenceInfo[traj]['persistence'] = 0
      self.raiseADebug(f'Resetting convergence for trajectory {traj}.')

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """
    return
