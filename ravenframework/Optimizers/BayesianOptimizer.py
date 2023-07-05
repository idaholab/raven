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
    dim = len(featurePoint)
    x = np.empty(dim)
    for index, varName in enumerate(list(featurePoint)):
      x[index] = featurePoint[varName]
    return x

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

  ###############################################
  # Model Training and Evaluation #
  ###############################################
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
    print(res)
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
    if self._iteration[traj] // self._modelDuration == 0:
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
