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
from enum import Enum
from collections import deque, defaultdict
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import InputData, InputTypes, mathUtils
from .Sampled import Sampled
from .gradients import knownTypes as gradKnownTypes
from .gradients import returnInstance as gradReturnInstance
from .gradients import returnClass as gradReturnClass
from .stepManipulators import knownTypes as stepKnownTypes
from .stepManipulators import returnInstance as stepReturnInstance
from .stepManipulators import returnClass as stepReturnClass
from .acceptanceConditions import knownTypes as acceptKnownTypes
from .acceptanceConditions import returnInstance as acceptReturnInstance
from .acceptanceConditions import returnClass as acceptReturnClass
#Internal Modules End--------------------------------------------------------------------------------

class Process(Enum):
  """
    Enum for processes the GradientDescent is active in
  """
  INITIALIZING = 1      # starting trajectory
  SUBMITTING = 2      # submitting new opt/grad points
  COLLECTING_OPT = 3  # collecting evaluations of opt point
  COLLECTING_GRAD = 4 # collecting evaluations of grad point(s)
  INACTIVE = 9      # no longer active

class Motive(Enum):
  """
    Enum for the motivations for the GradientDescent's current process
  """
  CONVERGED = 0
  STARTING = 1
  SEEKING_OPT = 2
  ACCEPTED_OPT = 3
  REJECTED_OPT = 4
  REDUNDANT = 9

# utility function for defaultdict
def giveZero():
  """
    Utility function for defaultdict to 0
    @ In, None
    @ Out, giveZero, int, zero
  """
  return 0

class GradientDescent(Sampled):
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
  convergenceOptions = ['gradient',    # gradient magnitude
                        # TODO change in input space?
                        'objective',   # relative change in objective value
                        'stepSize'  # normalized step size
                       ]

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
    # gradient estimation options
    grad = InputData.parameterInputFactory('gradient', strictMode=True)
    specs.addSub(grad)
    ## common options to all gradient descenders
    # TODO grad.addSub(InputData.parameterInputFactory('proximity',
    # contentType=InputTypes.FloatType))
    ## get specs for each gradient subclass, and add them to this class's options
    for option in gradKnownTypes():
      subSpecs = gradReturnClass(option, cls).getInputSpecification()
      grad.addSub(subSpecs)

    # step sizing options
    step = InputData.parameterInputFactory('stepSize', strictMode=True)
    specs.addSub(step)
    ## common options to all stepManipulator descenders
    ## TODO
    ## get specs for each stepManipulator subclass, and add them to this class's options
    for option in stepKnownTypes():
      subSpecs = stepReturnClass(option, cls).getInputSpecification()
      step.addSub(subSpecs)

    # acceptance conditions
    accept = InputData.parameterInputFactory('acceptance', strictMode=True)
    specs.addSub(accept)
    ## common options to all acceptanceCondition descenders
    ## TODO
    ## get specs for each acceptanceCondition subclass, and add them to this class's options
    for option in acceptKnownTypes():
      subSpecs = acceptReturnClass(option, cls).getInputSpecification()
      accept.addSub(subSpecs)

    # convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True)
    specs.addSub(conv)
    for name in cls.convergenceOptions:
      conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType))
    conv.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType))
    # NOTE to add new convergence options, add them to convergenceOptions above, not here!

    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Sampled.__init__(self)
    ## Instance Variable Initialization
    # public
    self.type = 'GradientDescent Optimizer'
    # _protected
    self._gradientInstance = None  # instance of GradientApproximater
    self._stepInstance = None      # instance of StepManipulator
    self._acceotInstance = None    # instance of AcceptanceCondition
    self._gradProximity = 0.01     # TODO user input, the proximity for gradient evaluations
    # history trackers, by traj, are deques (-1 is most recent)
    self._gradHistory = {}         # gradients
    self._stepHistory = {}         # step sizes
    self._acceptHistory = {}       # acceptability
    self._stepRecommendations = {} # by traj, if a 'cut' or 'grow' is recommended else None
    self._acceptRerun = {}         # by traj, if True then override accept for point rerun
    self._convergenceCriteria = defaultdict(giveZero) # names and values for convergence checks
    self._convergenceInfo = {}     # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = 0  # consecutive persistence required to mark convergence
    # __private
    # additional methods
    ## register adaptive sample identification criteria
    self.registerIdentifier('purpose') # whether an opt, or which grad point

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    Sampled.handleInput(self, paramInput)

    # grad strategy
    gradParentNode = paramInput.findFirst('gradient')
    if len(gradParentNode.subparts) != 1:
      self.raiseAnError('The <gradient> node requires exactly one gradient strategy! Choose from: ', gradKnownTypes())
    gradNode = next(iter(gradParentNode.subparts))
    gradType = gradNode.getName()
    self._gradientInstance = gradReturnInstance(gradType, self)
    self._gradientInstance.handleInput(gradNode)

    # stepping strategy
    stepNode = paramInput.findFirst('stepSize')
    if len(stepNode.subparts) != 1:
      self.raiseAnError('The <stepNode> node requires exactly one stepping strategy! Choose from: ', stepKnownTypes())
    stepNode = next(iter(stepNode.subparts))
    stepType = stepNode.getName()
    self._stepInstance = stepReturnInstance(stepType, self)
    self._stepInstance.handleInput(stepNode)

    # acceptance strategy # FIXME this might be useful to more than just gradient descent! Maybe
    # FIXME continued ... move to "sampled"?
    acceptNode = paramInput.findFirst('acceptance')
    if acceptNode:
      if len(acceptNode.subparts) != 1:
        self.raiseAnError('The <acceptance> node requires exactly one acceptance strategy! Choose from: ', acceptKnownTypes())
      acceptNode = next(iter(acceptNode.subparts))
      acceptType = acceptNode.getName()
      self._acceptInstance = acceptReturnInstance(acceptType, self)
      self._acceptInstance.handleInput(acceptNode)
    else:
      # default to strict mode acceptance
      acceptNode = acceptReturnInstance('Strict', self)

    # convergence options
    convNode = paramInput.findFirst('convergence')
    if convNode is not None:
      for sub in convNode.subparts:
        if sub.getName() == 'persistence':
          self._requiredPersistence = sub.value
        else:
          self._convergenceCriteria[sub.name] = sub.value
    if not self._convergenceCriteria:
      self.raiseAWarning('No convergence criteria given; using defaults.')
      self._convergenceCriteria['gradient'] = 1e-6

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    Sampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    self._gradientInstance.initialize(self.toBeSampled, self._gradProximity)
    self._stepInstance.initialize(self.toBeSampled)
    self._acceptInstance.initialize()
    # queue up the first run for each trajectory
    initialStepSize = self._stepInstance.initialStepSize(len(self.toBeSampled)) # TODO user scaling option
    for traj, init in enumerate(self._initialValues):
      self._stepHistory[traj].append(initialStepSize)
      self._submitOptAndGrads(init, traj, 0, initialStepSize)

  ###############
  # Run Methods #
  ###############
  def checkConvergence(self, traj):
    """
      Checks the active convergence criteria.
      @ In, traj, int, trajectory identifier
      @ Out, converged, bool, convergence state
      @ Out, convs, dict, state of convergence criterions
    """
    convs = {}
    for conv in self._convergenceCriteria:
      # fix capitalization for RAVEN standards
      fName = conv[:1].upper() + conv[1:]
      # get function from lookup
      f = getattr(self, '_checkConv{}'.format(fName))
      # check convergence function
      okay = f(traj)
      # store and update
      convs[conv] = okay
    return any(convs.values()), convs

  def _useRealization(self, info, rlz, optVal):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ In, optVal, float, value of objective variable (corrected for min/max)
      @ Out, None
    """
    traj = info['traj']
    info['optVal'] = optVal
    purpose = info['purpose']
    # FIXME we assume all the denoising has already happened by now.
    if purpose.startswith('opt'):
      self._resolveNewOptPoint(traj, rlz, optVal, info)
    elif purpose.startswith('grad'):
      self._resolveNewGradPoint(traj, rlz, optVal, info)
    if self._checkStepReady(traj):
      # get new gradient
      opt, _ = self._stepTracker[traj]['opt']
      grads, gradInfos = zip(*self._stepTracker[traj]['grads'])
      gradMag, gradVersor, _ = self._gradientInstance.evaluate(opt,
                                                                      grads, gradInfos,
                                                                      self._objectiveVar)
      self._gradHistory[traj].append((gradMag, gradVersor))
      # get new step information
      newOpt, stepSize = self._stepInstance.step(opt, gradientHist=self._gradHistory[traj],
                                                 prevStepSize=self._stepHistory[traj],
                                                 recommend=self._stepRecommendations[traj])
      # clear recommendations on step size, since we took the recommendation
      self._stepRecommendations[traj] = None
      self._stepHistory[traj].append(stepSize)
      # start new step
      self._initializeStep(traj)
      self.raiseADebug('Taking step {} for traj {} ...'.format(self._stepCounter[traj], traj))
      self.raiseADebug(' ... gradient magn: {:1.2e} direction: {}'.format(gradMag, gradVersor))
      self.raiseADebug(' ... normalized step size: {}'.format(stepSize))
      # TODO denorm calcs could potentially be expensive, maybe not worth running
      ## initial tests show it's not a big deal for small systems
      self.raiseADebug(' ... current opt point:', self.denormalizeData(opt))
      self.raiseADebug(' ... new optimum candidate:', self.denormalizeData(newOpt))
      # initialize step
      self._submitOptAndGrads(newOpt, traj, self._stepCounter[traj], stepSize)
    # otherwise, continue submitting and collecting

  ###################
  # Utility Methods #
  ###################
  def _checkStepReady(self, traj):
    """
      Checks if enough information has been collected to proceed with optimization
      @ In, traj, int, identifier for trajectory of interest
      @ Out, ready, bool, True if all required data has been collected
    """
    # need to make sure opt point, grad points are all present
    tracker = self._stepTracker[traj]
    if tracker['opt'] is None:
      return False
    if len(tracker['grads']) < self._gradientInstance.numGradPoints():
      return False
    return True

  def _initializeStep(self, traj):
    """
      Initializes a new step in the optimization process.
      @ In, traj, int, the trajectory of interest
      @ Out, None
    """
    Sampled._initializeStep(self, traj)
    # tracker 'opt' set up in Sampled
    self._stepTracker[traj]['grads'] = []

  def initializeTrajectory(self, traj=None):
    """
      Handles the generation of a trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, new trajectory number
    """
    traj = Sampled.initializeTrajectory(self)
    self._gradHistory[traj] = deque(maxlen=self._maxHistLen)
    self._stepHistory[traj] = deque(maxlen=self._maxHistLen)
    self._acceptHistory[traj] = deque(maxlen=self._maxHistLen)
    self._stepRecommendations[traj] = None
    self._acceptRerun[traj] = False
    self._convergenceInfo[traj] = {'persistence': 0}
    for criteria in self._convergenceCriteria:
      self._convergenceInfo[traj][criteria] = False
    return traj

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
    # submit opt point
    self.raiseADebug('* Submitting new opt and grad points *')
    self._submitRun(opt, traj, step, 'opt')
    # collect grad points
    gradPoints, gradInfos = self._gradientInstance.chooseEvaluationPoints(opt, stepSize)
    for i, grad in enumerate(gradPoints):
      self._submitRun(grad, traj, step, 'grad_{}'.format(i), moreInfo=gradInfos[i])

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
    self.raiseADebug('Adding run to queue: {} | {}'.format(point, info))
    #for key, inf in info.items():
    #  self.raiseADebug(' ... {}: {}'.format(key, inf))
    #self.raiseADebug(' ... {}: {}'.format('point', point))
    self._submissionQueue.append((point, info))
  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  # * * * * * * * * * * * * * * * *
  # Resolving potential opt points
  def _checkAcceptability(self, traj, optVal):
    """ TODO """
    # Check acceptability
    # NOTE: if self._optPointHistory[traj]: -> faster to use "try" for all but the first time
    try:
      old, _ = self._optPointHistory[traj][-1]
      oldVal = self._collectOptValue(old)
      self.raiseADebug(' ... change: {d: 1.3e} new: {n: 1.6e} old: {o: 1.6e}'
                      .format(d=optVal-oldVal, o=oldVal, n=optVal))
      # if this is an opt point rerun, accept it without checking.
      if self._acceptRerun[traj]:
        acceptable = 'rerun'
        self._acceptRerun[traj] = False
        self._stepRecommendations[traj] = 'shrink' # FIXME how much do we really want this?
      else:
        acceptable = self._checkForImprovement(optVal, oldVal)
    except IndexError:
      # if first sample, simply assume it's better!
      acceptable = 'first'
      old = None
    self._acceptHistory[traj].append(acceptable)
    self.raiseADebug(' ... {a}!'.format(a=acceptable))
    return acceptable, old

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    improved = self._acceptInstance.checkImprovement(new, old)
    return 'accepted' if improved else 'rejected'

  def _updateConvergence(self, traj, acceptable):
    """ TODO """
    ## NOTE we have multiple "if acceptable" trees here, as we need to update soln export regardless
    if acceptable == 'accepted':
      self.raiseADebug('Convergence Check for Trajectory {}:'.format(traj))
      # check convergence
      converged, convDict = self.checkConvergence(traj)
    else:
      converged = False
      convDict = dict((var, False) for var in self._convergenceInfo[traj])
    self._convergenceInfo[traj].update(convDict)
    return converged, convDict

  def _updatePersistence(self, traj, converged, optVal):
    """ TODO """
    # update persistence
    if converged:
      self._convergenceInfo[traj]['persistence'] += 1
      self.raiseADebug('Trajectory {} has converged successfully {} time(s)!'.format(traj, self._convergenceInfo[traj]['persistence']))
      if self._convergenceInfo[traj]['persistence'] >= self._requiredPersistence:
        self._closeTrajectory(traj, 'converge', 'converged', optVal)
    else:
      self._convergenceInfo[traj]['persistence'] = 0
      self.raiseADebug('Resetting convergence for trajectory {}.'.format(traj))

  def _updateSolutionExport(self, traj, rlz, acceptable):
    """
      Prints information to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ Out, None
    """
    # FIXME abstract this for Sampled base class!!
    denormed = self.denormalizeData(rlz)
    # meta variables
    solution = {'iteration': self._stepCounter[traj],
                'trajID': traj,
                'stepSize': self._stepHistory[traj][-1],
                'accepted': acceptable,
               }
    for key, val in self._convergenceInfo[traj].items():
      solution['conv_{}'.format(key)] = val
    # variables, objective function, constants, etc
    solution[self._objectiveVar] = rlz[self._objectiveVar]
    for var in self.toBeSampled:
      # TODO dimensionality?
      solution[var] = denormed[var]
    for var, val in self.constants.items():
      solution[var] = val
    for var in self.dependentSample:
      solution[var] = rlz[var]
    # format rlz for dataobject
    solution = dict((var, np.atleast_1d(val)) for var, val in solution.items())
    self._solutionExport.addRealization(solution)

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In,
    """
    # cancel grad runs
    self._cancelAssociatedJobs(info['traj'], step=info['step'])
    ## what do do if a point is rejected?
    # for now, rerun the opt point and gradients, AND cut step
    # TODO user option to EITHER rerun opt point OR cut step!
    # initialize a new step
    self._initializeStep(traj)
    # track that the next recommended step size for this traj should be "cut"
    self._stepRecommendations[traj] = 'shrink'
    # get new grads around new point
    self._stepCounter[traj] += 1
    self._submitOptAndGrads(old, traj, self._stepCounter[traj], self._stepHistory[traj][-1])
    self._acceptRerun[traj] = True
  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  # * * * * * * * * * * * * * * * *
  # Convergence Checks
  # Note these names need to be formatted according to checkConvergence check!
  convFormat = ' ... {name:^12s}: {conv:5s}, {got:1.2e} / {req:1.2e}'

  def _checkConvGradient(self, traj):
    """
      Checks the gradient magnitude for convergence
      @ In, traj, int, trajectory identifier
      @ Out, converged, bool, convergence state
    """
    gradMag, _ = self._gradHistory[traj][-1]
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
    stepSize = self._stepHistory[traj][-1]
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
    delta = mathUtils.relativeDiff(self._collectOptValue(o2), self._collectOptValue(o1))
    converged = abs(delta) < self._convergenceCriteria['objective']
    self.raiseADebug(self.convFormat.format(name='objective',
                                            conv=str(converged),
                                            got=delta,
                                            req=self._convergenceCriteria['objective']))
    return converged
  # END convergence Checks
  # * * * * * * * * * * * * * * * *