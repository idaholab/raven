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
  The Optimizer is a specialization of adaptive sampling.
  This base class defines the principle methods required for optimizers and provides some general utilities.

  Reworked 2020-01
  @author: talbpaul
"""
# for future compatibility with Python 3------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
# End compatibility block for Python 3--------------------------------------------------------------

# External Modules----------------------------------------------------------------------------------
import copy
import abc
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..utils import randomUtils, InputData, InputTypes
from ..Samplers import AdaptiveSampler, Sampler
# Internal Modules End------------------------------------------------------------------------------

class Optimizer(AdaptiveSampler):
  """
    The Optimizer is a specialization of adaptive sampling.
    This base class defines the principle methods required for optimizers and provides some general utilities.
    This base class is responsible for:
     - Implementing Sampler API
     - Handling stochastic resampling
     - Establishing "trajectory" counter
     - Handling Constant, Function variables
     - Specifying objective variable
     - Assembling constraints
     - API for adding, removing trajectories
     - Prefix handling for trajectory, denoising
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def userManualDescription(cls):
    """
      Provides a user manual description for this actor. Should only be needed for base classes.
      @ In, None
      @ Out, descr, string, description
    """
    descr = r"""
    \section{Optimizers} \label{sec:Optimizers}
    The optimizer is another important entity in the RAVEN framework. It performs the driving of a
    specific ``goal function'' or ``objective function'' over the model for value optimization. The
    Optimizer can be used almost anywhere a Sampler can be used, and is only distinguished from other
    AdaptiveSampler strategies for clarity.
    """

    return descr

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(Optimizer, cls).getInputSpecification()
    specs.description = 'Optimizers'

    # objective variable
    specs.addSub(InputData.parameterInputFactory('objective', contentType=InputTypes.StringType, strictMode=True,
        printPriority=90, # more important than <variable>
        descr=r"""Name of the response variable (or ``objective function'') that should be optimized
        (minimized or maximized)."""))

    # modify Sampler variable nodes
    variable = specs.getSub('variable')
    variable.addSub(InputData.parameterInputFactory('initial', contentType=InputTypes.FloatListType,
        descr=r"""indicates the initial values where independent trajectories for this optimization
              effort should begin. The number of entries should be the same for all variables, unless
              a variable is initialized with a sampler (see \xmlNode{samplerInit} below). Note these
              entries are ordered; that is, if the optimization variables are $x$ and $y$, and the initial
              values for $x$ are \xmlString{1, 2, 3, 4} and initial values for $y$ are \xmlString{5, 6, 7, 8},
              then there will be four starting trajectories beginning at the locations (1, 5), (2, 6),
              (3, 7), and (4, 8)."""))

    # initialization
    # TODO similar to MonteCarlo and other samplers, maybe overlap?
    init = InputData.parameterInputFactory('samplerInit', strictMode=True,
        printPriority=105, # more important than <variable>
        descr=r"""collection of nodes that describe the initialization of the optimization algorithm.""")
    seed = InputData.parameterInputFactory('initialSeed', contentType=InputTypes.IntegerType,
        descr=r"""seed for random number generation. Note that by default RAVEN uses an internal seed,
              so this seed must be changed to observe changed behavior. \default{RAVEN-determined}""")
    minMaxEnum = InputTypes.makeEnumType('MinMax', 'MinMaxType', ['min', 'max'])
    minMax = InputData.parameterInputFactory('type', contentType=minMaxEnum,
        descr=r"""the type of optimization to perform. \xmlString{min} will search for the lowest
              \xmlNode{objective} value, while \xmlString{max} will search for the highest value.""")
    init.addSub(seed)
    init.addSub(minMax)
    specs.addSub(init)

    ConstraintInput = InputData.parameterInputFactory('Constraint', contentType=InputTypes.StringType, strictMode=True,
        printPriority=150,
        descr=r"""name of \xmlNode{Function} which contains explicit constraints for the sampling of
              the input space of the Model. From a practical point of view, this XML node must contain
              the name of a function defined in the \xmlNode{Functions} block (see Section~\ref{sec:functions}).
              This external function must contain a method called ``constrain'', which returns True for
              inputs satisfying the explicit constraints and False otherwise. \nb Currently this accepts any number of constraints from the user.""")
    ConstraintInput.addParam("class", InputTypes.StringType, True,
        descr=r"""RAVEN class for this source. Options include \xmlString{Functions}. """)
    ConstraintInput.addParam("type", InputTypes.StringType, True,
        descr=r"""RAVEN type for this source. Options include \xmlNode{External}.""")


    ImplicitConstraintInput =  InputData.parameterInputFactory('ImplicitConstraint', contentType=InputTypes.StringType, strictMode=True,
        printPriority=150,
        descr=r"""name of \xmlNode{Function} which contains implicit constraints of the Model. From a practical
              point of view, this XML node must contain the name of a function defined in the \xmlNode{Functions}
              block (see Section~\ref{sec:functions}). This external function must contain a method called
              ``implicitConstrain'', which returns True for outputs satisfying the implicit constraints and False otherwise.""")
    ImplicitConstraintInput.addParam("class", InputTypes.StringType, True,
        descr=r"""RAVEN class for this source. Options include \xmlString{Functions}. """)
    ImplicitConstraintInput.addParam("type", InputTypes.StringType, True,
        descr=r"""RAVEN type for this source. Options include \xmlNode{External}.""")

    specs.addSub(ConstraintInput)
    specs.addSub(ImplicitConstraintInput)


    # assembled objects
    specs.addSub(InputData.assemblyInputFactory('Sampler', contentType=InputTypes.StringType, strictMode=True,
        printPriority=175,
        descr=r"""name of a Sampler that can be used to initialize the starting points for the trajectories
              of some of the variables. From a practical point of view, this XML node must contain the
              name of a Sampler defined in the \xmlNode{Samplers} block (see Section~\ref{subsec:onceThroughSamplers}).
              The Sampler will be used to initialize the trajectories' initial points for some or all
              of the variables. For example, if the Sampler selected samples only 2 of the 5 optimization
              variables, the \xmlNode{initial} XML node is required only for the remaining 3 variables."""))

    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    AdaptiveSampler.__init__(self)
    # Instance Variable Initialization
    # public
    # _protected
    self._seed = None                   # random seed to apply
    self._minMax = 'min'                # maximization or minimization?
    self._activeTraj = []               # tracks live trajectories
    self._cancelledTraj = {}            # tracks cancelled trajectories, and reasons
    self._convergedTraj = {}            # tracks converged trajectories, and values obtained
    self._numRepeatSamples = 1          # number of times to repeat sampling (e.g. denoising)
    self._objectiveVar = None           # objective variable for optimization
    self._initialValuesFromInput = None # initial variable values from inputs, list of dicts (used to reset optimizer when re-running workflow)
    self._initialValues = None          # initial variable values (trajectory starting locations), list of dicts
    self._variableBounds = None         # dictionary of upper/lower bounds for each variable (may be inf?)
    self._trajCounter = 0               # tracks numbers to assign to trajectories
    self._initSampler = None            # sampler to use for picking initial seeds
    self._constraintFunctions = []      # list of constraint functions
    self._impConstraintFunctions = []   # list of implicit constraint functions
    self._requireSolnExport = True      # optimizers only produce result in solution export
    self.optAssemblerList = ['DataObjects', 'Distributions', 'Functions', 'Files'] # List of assembler entities required to initialize an optmizer
    # __private
    # additional methods
    self.addAssemblerObject('Constraint', InputData.Quantity.zero_to_infinity)      # Explicit (input-based) constraints
    self.addAssemblerObject('ImplicitConstraint', InputData.Quantity.zero_to_infinity)      # Implicit constraints
    self.addAssemblerObject('Sampler', InputData.Quantity.zero_to_one)          # This Sampler can be used to initialize the optimization initial points (e.g. partially replace the <initial> blocks for some variables)


    # register adaptive sample identification criteria
    self.registerIdentifier('traj') # the trajectory of interest

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the optimizers that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = AdaptiveSampler._localWhatDoINeed(self)
    needDict['Functions'] = [(None, 'all')]

    return needDict

  def _localGenerateAssembler(self, initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      Overloads the base Sampler class since optimizer has different requirements
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    AdaptiveSampler._localGenerateAssembler(self, initDict)
    # functions and distributions already collected
    self.assemblerDict['DataObjects'] = []
    self.assemblerDict['Distributions'] = []
    self.assemblerDict['Functions'] = []
    self.assemblerDict['Files'] = []
    for mainClass in self.optAssemblerList:
      for funct in initDict[mainClass]:
        self.assemblerDict[mainClass].append([mainClass,
                                              initDict[mainClass][funct].type,
                                              funct,
                                              initDict[mainClass][funct]])

  def localInputAndChecks(self, xmlNode, paramInput):
    """
      Method that serves as a pass-through for input reading.
      comes from inheriting from Sampler and _readMoreXML chain.
      @ In, xmlNode, xml.etree.ElementTree.Element, xml element node (don't use!)
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    # this is just a passthrough until sampler gets reworked or renamed
    self.handleInput(paramInput)

  def _localWhatDoINeed(self):
    """
      Identifies needed distributions and functions.
      Overloads Sampler base implementation because of unique needs.
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {}
    for elem in self.optAssemblerList:
      needDict[elem] = [(None,'all')]

    return needDict

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    # the reading of variables (dist or func) and constants already happened in _readMoreXMLbase in Sampler
    # objective var
    self._objectiveVar = paramInput.findFirst('objective').value

    # sampler init
    # self.readSamplerInit() can't be used because it requires the xml node
    init = paramInput.findFirst('samplerInit')
    if init is not None:
      # initialSeed
      seed = init.findFirst('initialSeed')
      if seed is not None:
        self._seed = seed.value
      # minmax
      minMax = init.findFirst('type')
      if minMax is not None:
        self._minMax = minMax.value

    # variables additional reading
    for varNode in paramInput.findAll('variable'):
      if varNode.findFirst('function') is not None:
        continue # handled by Sampler base class, so skip it
      var = varNode.parameterValues['name']
      initsNode = varNode.findFirst('initial')
      # note: initial values might also come later from samplers!
      if initsNode:
        inits = initsNode.value
        # initialize list of dictionaries if needed
        if not self._initialValuesFromInput:
          self._initialValuesFromInput = [{} for _ in inits]
        # store initial values
        for i, init in enumerate(inits):
          self._initialValuesFromInput[i][var] = init
    self._initialValues = copy.copy(self._initialValuesFromInput)

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    if solutionExport is None:
      self.raiseAnError(IOError, 'Optimizers require a SolutionExport DataObject. Please add a <SolutionExport> node in the Step!')

    # functional constraints
    for entry in self.assemblerDict.get('Constraint', []):
      self._constraintFunctions.append(entry[3])

    for entry in self.assemblerDict.get('ImplicitConstraint', []):
      self._impConstraintFunctions.append(entry[3])
    AdaptiveSampler.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    # sampler
    self._initializeInitSampler(externalSeeding)
    # seed
    if self._seed is not None:
      randomUtils.randomSeed(self._seed)
    # variable bounds
    self._variableBounds = {}
    for var in self.toBeSampled:
      dist = self.distDict[var]
      lower = dist.lowerBound if dist.lowerBound is not None else -np.inf
      upper = dist.upperBound if dist.upperBound is not None else np.inf
      self._variableBounds[var] = [lower, upper]
      self.raiseADebug(f'Set bounds for opt var "{var}" to {self._variableBounds[var]}')
    # trajectory initialization
    for i, init in enumerate(self._initialValues):
      self._initialValues[i] = self.normalizeData(init)
      self.initializeTrajectory()

  ###############
  # Run Methods #
  ###############
  def amIreadyToProvideAnInput(self):
    """
      This is a method that should be called from any user of the optimizer before requiring the generation of a new input.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of model evaluation, convergence criteria met, etc.
      @ In, None
      @ Out, ready, bool, indicating the readiness of the optimizer to generate a new input.
    """
    # if any trajectories are still active, we're ready to provide an input
    ready = AdaptiveSampler.amIreadyToProvideAnInput(self)
    if not self._activeTraj:
      self.raiseADebug(' ... No active optimization trajectories.')
      ready = False
    return ready

  ###################
  # Utility Methods #
  ###################
  @abc.abstractmethod
  def checkConvergence(self, traj, new, old):
    """
      Method to check whether the convergence criteria has been met.
      @ In, traj, int, trajectory identifier
      @ In, new, dict, new opt point
      @ In, old, dict, previous opt point
    """

  @abc.abstractmethod
  def _updateSolutionExport(self, traj, rlz, acceptable, rejectReason):
    """
      Stores information to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ In, rejectReason, str, reject reason of opt point, or return None if accepted
      @ Out, None
    """

  def _addTrackingInfo(self, info, **kwargs):
    """
      Creates realization identifiers to identify particular realizations as they return from the JobHandler.
      Expandable by inheritors.
      @ In, info, dict, dictionary of potentially-existing added identifiers
      @ In, kwargs, dict, dictionary of keyword arguments
      @ Out, None (but "info" gets modified)
    """
    # TODO shouldn't this require the realization and information to do right?
    info['traj'] = kwargs['traj']

  def _collectOptValue(self, rlz):
    """
      collects the objective variable from a realization and adjusts the sign for min/max
      @ In, rlz, dict, realization particularly including objective variable
      @ Out, optVal, float, sign-adjust objective value
    """
    optVal = (-1 if self._minMax == 'max' else 1) * rlz[self._objectiveVar]

    return optVal

  def _collectOptPoint(self, rlz):
    """
      collects the point (dict) from a realization
      @ In, rlz, dict, realization particularly including objective variable
      @ Out, point, dict, point used in this realization
    """
    point = dict((var, float(rlz[var])) for var in self.toBeSampled)

    return point

  def _initializeInitSampler(self, externalSeeding):
    """
      Initializes samplers to be used for seeding trajectories.
      @ In, externalSeeding, int, unused
      @ Out, None
    """
    if not self.assemblerDict.get('Sampler', False):
      return
    sampler = self.assemblerDict['Sampler'][0][3]
    if not isinstance(sampler, Sampler):
      self.raiseAnError(IOError, 'Initialization samplers must be a Forward sampling type, such as MonteCarlo or Grid!')
    self._initSampler = sampler
    # initialize sampler
    samplerInit = {}
    for entity in self.optAssemblerList:
      samplerInit[entity] = dict((entry[2], entry[3]) for entry in self.assemblerDict.get(entity, []))
    self._initSampler._localGenerateAssembler(samplerInit)
    # assure sampler provides useful info
    for sampled in self._initSampler.toBeSampled:
      # all sampled variables should be used in the optimizer TODO is this really required? Or should this be a warning?
      if sampled not in self.toBeSampled:
        self.raiseAnError(IOError, f'Variable "{sampled}" initialized by Sampler "{self._initSampler.name}" is not an optimization variable for "{self.name}"!')
    self._initSampler.initialize(externalSeeding)
    # initialize points
    numTraj = len(self._initialValues) if self._initialValues else None
    # if there are already-initialized variables (i.e. not sampled, but given), then check num samples
    if numTraj:
      if numTraj != self._initSampler.limit:
        self.raiseAnError(IOError, f'{numTraj} initial points have been given, but Initialization Sampler "{self._initSampler.name}" provides {self._initSampler.limit} samples!')
    else:
      numTraj = self._initSampler.limit
      self._initialValues = [{} for _ in range(numTraj)]
    for n, _ in enumerate(self._initialValues):
      # prep the sampler, in case it needs it #TODO can we get rid of this for forward sampler?
      self._initSampler.amIreadyToProvideAnInput()
      # get the sample
      self._initSampler.generateInput(None, None)
      rlz = self._initSampler.inputInfo['SampledVars']
      # NOTE by looping over self.toBeSampled, we could potentially not error out when extra vars are sampled
      for var in self.toBeSampled:
        if var in rlz:
          self._initialValues[n][var] = rlz[var] # TODO float or np.1darray?

  def initializeTrajectory(self, traj=None):
    """
      Sets up a new trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, trajectory number
    """
    if traj is None:
      traj = self._trajCounter
      self._trajCounter += 1
    if traj not in self._activeTraj:
      self._activeTraj.append(traj)

    return traj

  def _closeTrajectory(self, traj, action, reason, value):
    """
      Removes a trajectory from active space.
      @ In, traj, int, trajectory identifier
      @ In, action, str, method in which to close ('converge' or 'cancel')
      @ In, reason, str, reason for closure
      @ In, value, float, opt value obtained
      @ Out, None
    """
    if traj in self._activeTraj:
      self._activeTraj.remove(traj)
    info = {'reason': reason, 'value': value}
    assert action in ['converge', 'cancel']
    if action == 'converge':
      self._convergedTraj[traj] = info
    else: # action == 'cancel'
      self._cancelledTraj[traj] = info

  def normalizeData(self, denormed):
    """
      Method to normalize the data
      @ In, denormed, dict, dictionary containing the value of decision variables to be normalized, in form of {varName: varValue}
      @ Out, normalized, dict, dictionary containing the value of normalized decision variables, in form of {varName: varValue}
    """
    # some algorithms should not be normalizing and denormalizing!
    # in that case, we allow this method to turn off normalization
    if self.needDenormalized():
      return denormed
    normalized = copy.deepcopy(denormed)
    for var in self.toBeSampled:
      val = denormed[var]
      lower, upper = self._variableBounds[var]
      normalized[var] = (val - lower) / (upper - lower)

    return normalized

  def denormalizeData(self, normalized):
    """
      Method to normalize the data
      @ In, normalized, dict, dictionary containing the value of decision variables to be denormalized, in form of {varName: varValue}
      @ Out, denormed, dict, dictionary containing the value of denormalized decision variables, in form of {varName: varValue}
    """
    # some algorithms should not be normalizing and denormalizing!
    # in that case, we allow this method to turn off normalization
    if self.needDenormalized():
      return normalized
    denormed = copy.deepcopy(normalized)
    for var in self.toBeSampled:
      val = normalized[var]
      lower, upper = self._variableBounds[var]
      denormed[var] = val * (upper - lower) + lower

    return denormed

  def needDenormalized(self):
    """
      Determines if the currently used algorithms should be normalizing the input space or not
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    # overload as needed in inheritors
    return False

  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    # Sampler also has a flush method
    super().flush()
    self.metadataKeys = set()
    self.assemblerDict = {}
    self._cancelledTraj = {}
    self._convergedTraj = {}
    self._initialValues = copy.copy(self._initialValuesFromInput)
    self._variableBounds = None
    self._trajCounter = 0
    self._constraintFunctions = []
    self._impConstraintFunctions = []
    self._initSampler = None
