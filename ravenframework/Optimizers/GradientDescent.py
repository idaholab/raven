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
  Base class for Sampled Optimizers using gradient descent optimization methods.

  Created 2020-01
  @author: talbpaul
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
from collections import deque, defaultdict
import numpy as np

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import InputData, InputTypes, mathUtils
from .RavenSampled import RavenSampled

from .gradients import factory as gradFactory
from .stepManipulators import factory as stepFactory
from .acceptanceConditions import factory as acceptFactory

from .stepManipulators import NoConstraintResolutionFound, NoMoreStepsNeeded

#Internal Modules End--------------------------------------------------------------------------------

class GradientDescent(RavenSampled):
  """
    Base class for Sampled Optimizers using gradient descent optimization methods.
    Handles the following:
     - Implements API for step size handling
       - Initialization, iteration, on constrain violation, etc
     - Implements API for gradient handling
       - Algorithm for estimating local/global gradient
       - Perturbation distance
       - Perturbation direction
     - Implements method(s) for stepping around constraints
     - Implements history tracking
       - evaluations, gradients, step sizes
     - Implements trajectory handling
       - Initial points (user or sampled)
       - Trajectory removal criteria (using Optimizer API)
     - Implement iterative step limit checking
     - Implement relative/absolute convergence
       - converge on gradient magnitude, change in evaluation, min step size
     - Implement summary of step iteration to SolutionExport
  """
  # convergence option names and their user manual descriptions
  convergenceOptions = {'gradient': r"""provides the desired value for the local estimated of the gradient
                                    for convergence. \default{1e-6, if no criteria specified}""",
                        # TODO change in input space?
                        'objective': r"""provides the maximum relative change in the objective function for convergence.""",
                        'stepSize': r"""provides the maximum size in relative step size for convergence.""",
                       }

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
    specs = super(GradientDescent, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{GradientDescent} optimizer represents an a la carte option
                            for performing gradient-based optimization with a variety of gradient
                            estimation techniques, stepping strategies, and acceptance criteria. \hspace{12pt}
                            Gradient descent optimization generally behaves as a ball rolling down a hill;
                            the algorithm estimates the local gradient at a point, and attempts to move
                            ``downhill'' in the opposite direction of the gradient (if minimizing; the
                            opposite if maximizing). Once the lowest point along the iterative gradient search
                            is discovered, the algorithm is considered converged. \hspace{12pt}
                            Note that gradient descent algorithms are particularly prone to being trapped
                            in local minima; for this reason, depending on the model, multiple trajectories
                            may be needed to obtain the global solution.
                            """
    # gradient estimation options
    grad = InputData.parameterInputFactory('gradient', strictMode=True,
        printPriority=106,
        descr=r"""a required node containing the information about which gradient approximation algorithm to
              use, and its settings if applicable. Exactly one of the gradient approximation algorithms
              below may be selected for this Optimizer.""")
    specs.addSub(grad)
    ## get specs for each gradient subclass, and add them to this class's options
    for option in gradFactory.knownTypes():
      subSpecs = gradFactory.returnClass(option).getInputSpecification()
      grad.addSub(subSpecs)

    # step sizing options
    step = InputData.parameterInputFactory('stepSize', strictMode=True,
        printPriority=107,
        descr=r"""a required node containing the information about which iterative stepping algorithm to
              use, and its settings if applicable. Exactly one of the stepping algorithms
              below may be selected for this Optimizer.""")
    specs.addSub(step)
    # common options to all stepManipulator descenders
    # TODO
    # get specs for each stepManipulator subclass, and add them to this class's options
    for option in stepFactory.knownTypes():
      subSpecs = stepFactory.returnClass(option).getInputSpecification()
      step.addSub(subSpecs)

    # acceptance conditions
    accept = InputData.parameterInputFactory('acceptance', strictMode=True,
        printPriority=108,
        descr=r"""a required node containing the information about the acceptability criterion for iterative
              optimization steps, i.e. when a potential new optimal point should be rejected and when
              it can be accepted. Exactly one of the acceptance criteria
              below may be selected for this Optimizer.""")
    specs.addSub(accept)
    # common options to all acceptanceCondition descenders
    # get specs for each acceptanceCondition subclass, and add them to this class's options
    for option in acceptFactory.knownTypes():
      subSpecs = acceptFactory.returnClass(option).getInputSpecification()
      accept.addSub(subSpecs)

    # convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True,
        printPriority=109,
        descr=r"""a node containing the desired convergence criteria for the optimization algorithm.
              Note that convergence is met when any one of the convergence criteria is met. If no convergence
              criteria are given, then nominal convergence on gradient value is used.""")
    specs.addSub(conv)

    conv.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType,
        printPriority=300,
        descr=r"""provides the number of consecutive times convergence should be reached before a trajectory
              is considered fully converged. This helps in preventing early false convergence."""))
    conv.addSub(InputData.parameterInputFactory('constraintExplorationLimit', contentType=InputTypes.IntegerType,
        printPriority=9999,
        descr=r"""provides the number of consecutive times a functional constraint boundary can be explored
              for an acceptable sampling point before aborting search. Only apples if using a
              \xmlNode{Constraint}. \default{500}"""))

    for name, descr in cls.convergenceOptions.items():
      conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType, descr=descr))
    terminate = InputData.parameterInputFactory('terminateFollowers', contentType=InputTypes.BoolType,
        descr=r"""indicates whether a trajectory should be terminated when it begins following the path
              of another trajectory.""")
    terminate.addParam('proximity', param_type=InputTypes.FloatType, required=False,
        descr=r"""provides the normalized distance at which a trajectory's head should be proximal to
              another trajectory's path before terminating the following trajectory.""")
    conv.addSub(terminate)
    # NOTE to add new convergence options, add them to convergenceOptions above, not here!

    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, ok, dict, {varName: description} for valid solution export variable names
    """
    # cannot be determined before run-time due to variables and prefixes.
    ok = super(GradientDescent, cls).getSolutionExportVariableNames()
    new = {'stepSize': 'the size of step taken in the normalized input space to arrive at each optimal point'}
    new['conv_{CONV}'] = 'status of each given convergence criteria'
    # TODO need to include StepManipulators and GradientApproximators solution export entries as well!
    # -> but really should only include active ones, not all of them. This seems like it should work
    #    when the InputData can scan forward to determine which entities are actually used.
    for grad in gradFactory.knownTypes():
      new.update(gradFactory.returnClass(grad).getSolutionExportVariableNames())
    for step in stepFactory.knownTypes():
      new.update(stepFactory.returnClass(step).getSolutionExportVariableNames())
    ok.update(new)

    return ok

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    RavenSampled.__init__(self)
    # Instance Variable Initialization
    # public
    self.type = 'GradientDescent Optimizer'
    # _protected
    self._gradientInstance = None  # instance of GradientApproximator
    self._stepInstance = None      # instance of StepManipulator
    self._acceptInstance = None    # instance of AcceptanceCondition
    # history trackers, by traj, are deques (-1 is most recent)
    self._gradHistory = {}         # gradients
    self._stepHistory = {}         # {'magnitude': size, 'versor': direction, 'info': dict} for step
    self._acceptHistory = {}       # acceptability
    self._stepRecommendations = {} # by traj, if a 'cut' or 'grow' is recommended else None
    self._acceptRerun = {}         # by traj, if True then override accept for point rerun
    self._convergenceCriteria = defaultdict(mathUtils.giveZero) # names and values for convergence checks
    self._convergenceInfo = {}       # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = None # consecutive persistence required to mark convergence
    self._terminateFollowers = True  # whether trajectories sharing a point should cause termination
    self._followerProximity = 1e-2   # distance at which annihilation can start occurring, in ?normalized? space
    self._trajectoryFollowers = defaultdict(list) # map of trajectories to the trajectories following them
    self._functionalConstraintExplorationLimit = 500 # number of input-space explorations allowable for functional constraints
    # __private
    # additional methods
    # register adaptive sample identification criteria
    self.registerIdentifier('purpose') # whether an opt, or which grad point

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)

    # grad strategy
    gradParentNode = paramInput.findFirst('gradient')
    if len(gradParentNode.subparts) != 1:
      self.raiseAnError('The <gradient> node requires exactly one gradient strategy! Choose from: ', gradFactory.knownTypes())
    gradNode = next(iter(gradParentNode.subparts))
    gradType = gradNode.getName()
    self._gradientInstance = gradFactory.returnInstance(gradType)
    self._gradientInstance.handleInput(gradNode)

    # stepping strategy
    stepNode = paramInput.findFirst('stepSize')
    if len(stepNode.subparts) != 1:
      self.raiseAnError('The <stepNode> node requires exactly one stepping strategy! Choose from: ', stepFactory.knownTypes())
    stepNode = next(iter(stepNode.subparts))
    stepType = stepNode.getName()
    self._stepInstance = stepFactory.returnInstance(stepType)
    self._stepInstance.handleInput(stepNode)

    # acceptance strategy
    acceptNode = paramInput.findFirst('acceptance')
    if acceptNode:
      if len(acceptNode.subparts) != 1:
        self.raiseAnError('The <acceptance> node requires exactly one acceptance strategy! Choose from: ', acceptFactory.knownTypes())
      acceptNode = next(iter(acceptNode.subparts))
      acceptType = acceptNode.getName()
      self._acceptInstance = acceptFactory.returnInstance(acceptType)
      self._acceptInstance.handleInput(acceptNode)
    else:
      # default to strict mode acceptance
      acceptNode = acceptFactory.returnInstance('Strict')

    # convergence options
    convNode = paramInput.findFirst('convergence')
    if convNode is not None:
      for sub in convNode.subparts:
        if sub.getName() == 'persistence':
          self._requiredPersistence = sub.value
        elif sub.getName() == 'terminateFollowers':
          self._terminateFollowers = sub.value
          self._followerProximity = sub.parameterValues.get('proximity', 1e-2)
        elif sub.getName() == 'constraintExplorationLimit':
          self._functionalConstraintExplorationLimit = sub.value
        else:
          self._convergenceCriteria[sub.name] = sub.value
    if not self._convergenceCriteria:
      self.raiseAWarning('No convergence criteria given; using defaults.')
      self._convergenceCriteria['gradient'] = 1e-6
    # same point is ALWAYS a criterion
    self._convergenceCriteria['samePoint'] = 1e-16 # TODO user option?
    # set persistence to 1 if not set
    if self._requiredPersistence is None:
      self.raiseADebug('No persistence given; setting to 1.')
      self._requiredPersistence = 1

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    self._gradientInstance.initialize(self.toBeSampled)
    self._stepInstance.initialize(self.toBeSampled, persistence=self._requiredPersistence)
    self._acceptInstance.initialize()
    # if single trajectory, turn off follower termination
    if len(self._initialValues) < 2:
      self.raiseADebug('Setting terminateFollowers to False since only 1 trajectory exists.')
      self._terminateFollowers = False
    # queue up the first run for each trajectory
    initialStepSize = self._stepInstance.initialStepSize(len(self.toBeSampled)) # TODO user scaling option
    for traj, init in enumerate(self._initialValues):
      self._stepHistory[traj].append({'magnitude': initialStepSize, 'versor': None, 'info': None})
      self._submitOptAndGrads(init, traj, 0, initialStepSize)


  ###############
  # Run Methods #
  ###############
  def checkConvergence(self, traj, new, old):
    """
      Checks the active convergence criteria.
      @ In, traj, int, trajectory identifier
      @ In, new, dict, new opt point
      @ In, old, dict, previous opt point
      @ Out, checkConvergence[0], bool, convergence state
      @ Out, convs, dict, state of convergence criterions
    """
    convs = {}
    for conv in self._convergenceCriteria:
      # special treatment for same point check
      if conv == 'samePoint':
        convs[conv] = self._checkConvSamePoint(new, old)
        continue
      # fix capitalization for RAVEN standards
      fName = conv[:1].upper() + conv[1:]
      # get function from lookup
      f = getattr(self, f'_checkConv{fName}')
      # check convergence function
      okay = f(traj)
      # store and update
      convs[conv] = okay

    return any(convs.values()), convs

  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ Out, None
    """
    traj = info['traj']
    optVal = rlz[self._objectiveVar]
    info['optVal'] = optVal
    purpose = info['purpose']
    if purpose.startswith('opt'):
      self._resolveNewOptPoint(traj, rlz, optVal, info)
    elif purpose.startswith('grad'):
      self._resolveNewGradPoint(traj, rlz, optVal, info)
    if self._checkStepReady(traj):
      # get new gradient
      self.raiseADebug('Opt point accepted and gradient points collected, searching new opt point ...')
      opt, _ = self._stepTracker[traj]['opt']
      grads, gradInfos = zip(*self._stepTracker[traj]['grads'])
      gradMag, gradVersor, _ = self._gradientInstance.evaluate(opt,
                                                               grads,
                                                               gradInfos,
                                                               self._objectiveVar)
      self.raiseADebug(' ... gradient calculated ...')
      self._gradHistory[traj].append((gradMag, gradVersor))
      # get new step information
      try:
        newOpt, stepSize, stepInfo = self._stepInstance.step(opt,
                                                  objVar=self._objectiveVar,
                                                  optHist=self._optPointHistory[traj],
                                                  gradientHist=self._gradHistory[traj],
                                                  prevStepSize=self._stepHistory[traj],
                                                  recommend=self._stepRecommendations[traj]
                                                  )
      except NoMoreStepsNeeded:
        # the stepInstance has decided it's done
        self.raiseAMessage(f'Step Manipulator "{self._stepInstance.type}" has declared no more steps needed!')
        self._closeTrajectory(traj, 'converge', 'converged', optVal)
        return

      self.raiseADebug(' ... found new proposed opt point ...')
      # check new opt point against constraints
      try:
        suggested, _ = self._handleExplicitConstraints(newOpt, opt, 'opt')
      except NoConstraintResolutionFound:
        # we've tried everything, but we just can't hack it
        self.raiseAMessage(f'Optimizer "{self.name}" trajectory {traj} was unable to continue due to functional or boundary constraints.')
        self._closeTrajectory(traj, 'converge', 'no constraint resolution', opt[self._objectiveVar])
        return

      # update values if modified by constraint handling
      deltas = dict((var, suggested[var] - opt[var]) for var in self.toBeSampled)
      actualStepSize, stepVersor, _ = mathUtils.calculateMagnitudeAndVersor(np.array(list(deltas.values())))

      # erase recommendations on step size, since we took the recommendation by now
      self._stepRecommendations[traj] = None
      # update the step history with the full suggested step size, NOT the actually-used step size
      # that came as a result of the boundary modifications
      self._stepHistory[traj].append({'magnitude': stepSize, 'versor': stepVersor, 'info': stepInfo})
      # start new step
      self._initializeStep(traj)
      self.raiseADebug(f'Taking step {self.getIteration(traj)} for traj {traj} ...')
      self.raiseADebug(f' ... gradient magn: {gradMag:1.2e} direction: {gradVersor}')
      self.raiseADebug(f' ... normalized desired step size: {stepSize}')
      self.raiseADebug(f' ... normalized actual  step size: {actualStepSize}')
      # TODO denorm calcs could potentially be expensive, maybe not worth running
      # initial tests show it's not a big deal for small systems
      # @alfoa: we might "get the verbosity" here and "compute the denorm just in
      #         case the verbosity is == debug...
      self.raiseADebug(f' ... current opt point: {self.denormalizeData(opt)}')
      self.raiseADebug(f' ... new optimum candidate: {self.denormalizeData(suggested)}')
      # initialize step
      self._submitOptAndGrads(suggested, traj, self.getIteration(traj), self._stepHistory[traj][-1]['magnitude'])
    # otherwise, continue submitting and collecting

  ###################
  # Utility Methods #
  ###################
  def _checkStepReady(self, traj):
    """
      Checks if enough information has been collected to proceed with optimization
      @ In, traj, int, identifier for trajectory of interest
      @ Out, _checkStepReady, bool, True if all required data has been collected
    """
    # ready to move step forward if (a) we have the opt point, and (b) we have the grad points
    tracker = self._stepTracker[traj]
    # check (a) we have the opt point
    if tracker['opt'] is None:
      return False
    # check (b) we have the grad points
    if len(tracker['grads']) < self._gradientInstance.numGradPoints():
      return False
    # if all checks passed, we're ready

    return True

  def _initializeStep(self, traj):
    """
      Initializes a new step in the optimization process.
      @ In, traj, int, the trajectory of interest
      @ Out, None
    """
    RavenSampled._initializeStep(self, traj)
    self.incrementIteration(traj)
    # tracker 'opt' set up in Sampled
    self._stepTracker[traj]['grads'] = []

  def initializeTrajectory(self, traj=None):
    """
      Handles the generation of a trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, new trajectory number
    """
    traj = RavenSampled.initializeTrajectory(self)
    self._gradHistory[traj] = deque(maxlen=self._maxHistLen)
    self._stepHistory[traj] = deque(maxlen=self._maxHistLen)
    self._acceptHistory[traj] = deque(maxlen=self._maxHistLen)
    self._stepRecommendations[traj] = None
    self._acceptRerun[traj] = False
    self._convergenceInfo[traj] = {'persistence': 0}
    for criteria in self._convergenceCriteria:
      self._convergenceInfo[traj][criteria] = False
    return traj

  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self._gradientInstance.flush()
    self._stepInstance.flush()
    self._gradHistory = {}
    self._stepHistory = {}
    self._acceptHistory = {}
    self._stepRecommendations = {}
    self._acceptRerun = {}
    self._trajectoryFollowers = defaultdict(list)

  def _resolveNewGradPoint(self, traj, rlz, optVal, info):
    """
      Consider and store a new gradient evaluation point
      @ In, traj, int, trajectory for this new point
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ In, optVal, float, value of objective variable (corrected for min/max)
    """
    self._stepTracker[traj]['grads'].append((rlz, info))

  # * * * * * * * * * * * * * * * *
  # Resolving potential opt points
  def _applyFunctionalConstraints(self, suggested, previous):
    """
      @ In, suggested, dict, NORMALIZED suggested point
      @ In, previous, dict, NORMALIZED previous point
      @ Out, suggested, dict, fixed up normalized point
      @ Out, modded, bool, True if point was modified within this method
    """
    # assume no modifications until proved otherwise
    modded = False
    # are we violating functional constraints?
    passFuncs = self._checkFunctionalConstraints(self.denormalizeData(suggested))
    # while in violation of constraints ...
    info = {'minStepSize': self._convergenceCriteria.get('stepSize', 1e-10)} # TODO why 1e-10?
    tries = self._functionalConstraintExplorationLimit
    while not passFuncs:
      modded = True
      #  try to find new acceptable point
      denormed = self.denormalizeData(suggested)
      # DEBUGG the following lines will add constraint search attempts to the solution export.
      # rlz = {'trajID': 0,
      #        'x': denormed['x'],
      #        'y': denormed['y'],
      #        'ans': 1 - tries / 100,
      #        'stepSize': 9999,
      #        'iteration': 9999,
      #        'accepted': 'search',
      #        'conv_gradient': 0,
      #       }
      # rlz = dict((key, np.atleast_1d(val)) for key, val in rlz.items())
      # self._solutionExport.addRealization(rlz)
      # END DEBUGG
      suggested, modStepSize, info = self._stepInstance.fixConstraintViolations(suggested, previous, info)
      denormed = self.denormalizeData(suggested)
      self.raiseADebug(f' ... suggested norm step {modStepSize:1.2e}, new opt {denormed}')
      passFuncs = self._checkFunctionalConstraints(denormed) and self._checkBoundaryConstraints(denormed)
      tries -= 1
      if tries == 0:
        self.raiseAnError(NotImplementedError, 'No acceptable point findable! Now what?')

    return suggested, modded

  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  # * * * * * * * * * * * * * * * *
  # Queuing Runs
  def _submitOptAndGrads(self, opt, traj, step, stepSize):
    """
      Submits a set of opt + grad points to the submission queue
      @ In, opt, dict, suggested opt point to evaluate
      @ In, traj, int, trajectory identifier
      @ In, step, int, iteration number identifier
      @ In, stepSize, float, nominal step size to use
      @ Out, None
    """
    # OPT POINT
    # submit opt point
    self.raiseADebug('* Submitting new opt and grad points *')
    self._submitRun(opt, traj, step, 'opt')
    # GRAD POINTS
    # collect grad points
    # HACK FIXME TODO adding constraints too
    # boundary is distribution dict of bounds
    # constraint functions is list of functions to call "evaluate" on
    # inputs is dictionary of other stuff that constraints might need to be evaluated
    # TODO we really should pass the checkFunctionalConstraints and check/applyBoundaryConstraints instead!!
    constraints = {'boundary': self.distDict,
                   'functional': self._constraintFunctions,
                   'inputs': copy.deepcopy(self.constants),
                   'normalize': self.normalizeData,
                   'denormalize': self.denormalizeData}
    gradPoints, gradInfos = self._gradientInstance.chooseEvaluationPoints(opt, stepSize, constraints=constraints)
    for i, grad in enumerate(gradPoints):
      self._submitRun(grad, traj, step, f'grad_{i}', moreInfo=gradInfos[i])

  def _submitRun(self, point, traj, step, purpose, moreInfo=None):
    """
      Submits a single run with associated info to the submission queue
      @ In, point, dict, point to submit
      @ In, traj, int, trajectory identifier
      @ In, step, int, iteration number identifier
      @ In, purpose, str, purpose of run (usually "opt" or "grad" or similar)
      @ In, moreInfo, dict, optional, additional run-identifying information to track
      @ Out, None
    """
    info = {}
    if moreInfo is not None:
      info.update(moreInfo)
    info.update({'traj': traj,
                 'step': step,
                 'purpose': purpose,
                })
    # NOTE: explicit constraints have been checked before this!
    self.raiseADebug(f'Adding run to queue: {self.denormalizeData(point)} | {info}')
    self._submissionQueue.append((point, info))
  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  # * * * * * * * * * * * * * * * *
  # Resolving potential opt points
  def _checkAcceptability(self, traj, opt, optVal, info):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ In, info, dict, identifying information about the opt point
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
      @ Out, rejectReason, str, reject reason of opt point, or return None if accepted
    """

    # Check acceptability
    if self._optPointHistory[traj]:
      old, _ = self._optPointHistory[traj][-1]
      oldVal = old[self._objectiveVar]
      # check if following another trajectory
      if self._terminateFollowers:
        following = self._stepInstance.trajIsFollowing(traj, self.denormalizeData(opt), info,
                                                       self._solutionExport,
                                                       self._trajectoryFollowers.get(traj, None),
                                                       self._followerProximity)
        if following is not None:
          self.raiseADebug(f'Cancelling Trajectory {traj} because it is following Trajectory {following}')
          self._trajectoryFollowers[following].append(traj) # "traj" is killed by "following"
          self._closeTrajectory(traj, 'cancel', f'following {following}', optVal)
          return 'accepted', old, 'None'

      self.raiseADebug(f' ... change: {optVal-oldVal:1.3e} new: {optVal:1.6e} old: {oldVal:1.6e}')
      rejectReason = 'None'
      # some stepManipulators may need to override the acceptance criteria, e.g. conjugate gradient
      if self._stepInstance.needsAccessToAcceptance:
        acceptable = self._stepInstance.modifyAcceptance(old, oldVal, opt, optVal)
      # if this is an opt point rerun, accept it without checking.
      elif self._acceptRerun[traj]:
        acceptable = 'rerun'
        self._acceptRerun[traj] = False
        self._stepRecommendations[traj] = 'shrink' # FIXME how much do we really want this?
      # check if same point
      elif all(opt[var] == old[var] for var in self.toBeSampled):
        # this is the classic "same point" trap; we accept the same point, and check convergence later
        acceptable = 'accepted'
      else:
        if self._impConstraintFunctions:
          accept = self._handleImplicitConstraints(opt)
          if accept:
            acceptable, rejectReason = self._checkForImprovement(optVal, oldVal)
          else:
            acceptable = 'rejected'
            rejectReason = 'implicitConstraintsViolation'
        else:
          acceptable, rejectReason = self._checkForImprovement(optVal, oldVal)
    else: # no history
      # if first sample, simply assume it's better!
      rejectReason = 'None'
      if self._impConstraintFunctions:
        accept = self._handleImplicitConstraints(opt)
        if not accept:
          self.raiseAWarning('First point violate Implicit constraint, please change another point to start!')
          rejectReason = 'implicitConstraintsViolation'
      acceptable = 'first'
      old = None
    self._acceptHistory[traj].append(acceptable)
    self.raiseADebug(f' ... {acceptable}!')

    return acceptable, old, rejectReason

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
      @ Out, rejectReason, str, reject reason of opt point, or return None if accepted
    """
    # TODO could this be a base RavenSampled class?
    improved = self._acceptInstance.checkImprovement(new, old)
    if improved:
      return 'accepted', 'None'
    else:
      return 'rejected', 'noImprovement'

  def _updateConvergence(self, traj, new, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, new, dict, new point
      @ In, old, dict, old point
      @ In, acceptable, str, condition of new point
      @ Out, converged, bool, True if converged on ANY criteria
    """
    # NOTE we have multiple "if acceptable" trees here, as we need to update soln export regardless
    if acceptable == 'accepted':
      self.raiseADebug(f'Convergence Check for Trajectory {traj}:')
      # check convergence
      converged, convDict = self.checkConvergence(traj, new, old)
    else:
      converged = False
      # since not accepted, none of the convergence criteria are acceptable
      # HOWEVER, since not accepted, do NOT reset persistence!
      convDict = dict((var, False) for var in self._convergenceInfo[traj] if var not in ['persistence'])
    self._convergenceInfo[traj].update(convDict)

    return converged

  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, int, identifier
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

  def _addToSolutionExport(self, traj, rlz, acceptable):
    """
      Contributes additional entries to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ Out, toAdd, dict, additional entries
    """
    toAdd = {'stepSize': self._stepHistory[traj][-1]['magnitude']}
    for key, val in self._convergenceInfo[traj].items():
      toAdd[f'conv_{key}'] = val
    # collect any additions from gradient and stepper
    # gradient
    grads, gradInfos = zip(*self._stepTracker[traj]['grads']) if len(self._stepTracker[traj]['grads']) else [], []
    fromGrad = self._gradientInstance.updateSolutionExport(grads, gradInfos)
    toAdd.update(fromGrad)
    # stepper
    fromStep = self._stepInstance.updateSolutionExport(self._stepHistory[traj])
    toAdd.update(fromStep)

    return toAdd

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
      @ Out, none
    """
    # cancel grad runs
    self._cancelAssociatedJobs(info['traj'], step=info['step'])
    # what do do if a point is rejected?
    # for now, rerun the opt point and gradients, AND cut step
    # TODO user option to EITHER rerun opt point OR cut step!
    # initialize a new step
    self._initializeStep(traj)
    # track that the next recommended step size for this traj should be "cut"
    self._stepRecommendations[traj] = 'shrink'
    # get new grads around new point
    self._submitOptAndGrads(old, traj, self.getIteration(traj), self._stepHistory[traj][-1]['magnitude'])
    self._acceptRerun[traj] = True
  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  # * * * * * * * * * * * * * * * *
  # Convergence Checks
  convFormat = RavenSampled.convFormat

  # NOTE checkConvSamePoint has a different call than the others
  # should this become an informational dict that can be passed to any of them?
  def _checkConvSamePoint(self, new, old):
    """
      Checks for a repeated same point
      @ In, new, dict, new opt point
      @ In, old, dict, old opt point
      @ Out, converged, bool, convergence state
    """
    # TODO diff within tolerance? Exactly equivalent seems good for now
    same = list(new[var] == old[var] for var in self.toBeSampled)
    converged = all(same)
    self.raiseADebug(self.convFormat.format(name='same point',
                                            conv=str(converged),
                                            got=sum(same),
                                            req=len(same)))

    return converged

  def _checkConvGradient(self, traj):
    """
      Checks the gradient magnitude for convergence
      @ In, traj, int, trajectory identifier
      @ Out, converged, bool, convergence state
    """
    gradMag, _ = self._gradHistory[traj][-1]
    gradMag = self.denormalizeGradient(gradMag)
    converged = gradMag < self._convergenceCriteria['gradient']
    self.raiseADebug(self.convFormat.format(name='gradient',
                                            conv=str(converged),
                                            got=gradMag,
                                            req=self._convergenceCriteria['gradient']))

    return converged

  def _checkConvStepSize(self, traj):
    """
      Checks the step size for convergence
      @ In, traj, int, trajectory identifier
      @ Out, converged, bool, convergence state
    """
    stepSize = self._stepHistory[traj][-1]['magnitude']
    converged = stepSize < self._convergenceCriteria['stepSize']
    self.raiseADebug(self.convFormat.format(name='stepSize',
                                            conv=str(converged),
                                            got=stepSize,
                                            req=self._convergenceCriteria['stepSize']))

    return converged

  def _checkConvObjective(self, traj):
    """
      Checks the change in objective for convergence
      @ In, traj, int, trajectory identifier
      @ Out, converged, bool, convergence state
    """
    if len(self._optPointHistory[traj]) < 2:
      return False
    o1, _ = self._optPointHistory[traj][-1]
    o2, _ = self._optPointHistory[traj][-2]
    delta = mathUtils.relativeDiff(o2[self._objectiveVar], o1[self._objectiveVar])
    converged = abs(delta) < self._convergenceCriteria['objective']
    self.raiseADebug(self.convFormat.format(name='objective',
                                            conv=str(converged),
                                            got=delta,
                                            req=self._convergenceCriteria['objective']))

    return converged
  # END convergence Checks
  # * * * * * * * * * * * * * * * *

  def needDenormalized(self):
    """
      Determines if the currently used algorithms should be normalizing the input space or not
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    return self._stepInstance.needDenormalized() or self._gradientInstance.needDenormalized()

  def denormalizeGradient(self, gradMag):
    """
      Denormalizes the gradient to correspond to the original space.
      @ In, gradMag, float, normalized space gradient magnitude
      @ Out, denormed, float, original (denormalized) gradient magnitude
    """
    # if no normalization is occuring, then just return as is
    if self.needDenormalized():
      return gradMag
    # scale by the product of the dimensions
    scale = 1
    for var in self.toBeSampled:
      lower, upper = self._variableBounds[var]
      scale *= upper - lower

    return gradMag / scale

  def _formatSolutionExportVariableNames(self, acceptable):
    """
      Does magic formatting for variables, based on this class's needs.
      Extend in inheritors as needed.
      @ In, acceptable, set, set of acceptable entries for solution export for this entity
      @ Out, new, set, modified set of acceptable variables with all formatting complete
    """
    # remaking the list is easier than using the existing one
    acceptable = RavenSampled._formatSolutionExportVariableNames(self, acceptable)
    new = []
    while acceptable:
      template = acceptable.pop()
      if '{CONV}' in template:
        new.extend([template.format(CONV=conv) for conv in self._convergenceCriteria])
      else:
        new.append(template)

    return set(new)
