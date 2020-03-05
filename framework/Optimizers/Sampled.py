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
  Base class for Optimizers using RAVEN's internal sampling mechanics.

  Created 2020-01
  @author: talbpaul
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import copy
import abc
from collections import deque
from functools import reduce
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils, randomUtils, InputData, InputTypes
from BaseClasses import BaseType
from Assembler import Assembler
from .Optimizer import Optimizer
from .acceptanceConditions import returnInstance as acceptReturnInstance
from .acceptanceConditions import returnClass as acceptReturnClass
#Internal Modules End--------------------------------------------------------------------------------

class Sampled(Optimizer):
  """
    Base class for Optimizers using RAVEN's internal sampling mechanics.
    Handles the following:
      - Maintain queue for required realizations
      - Label and retrieve realizations given labels
      - Establish API for convergence checking
      - Establish API to extend labels for particular implementations
      - Implements constraint checking
      - Implements model evaluation limitations
      - Implements rejection strategy (?)
      - Implements convergence persistence
      - Establish API for iterative sample output to solution export
      - Implements specific sampling methods from Sampler (when not present in Optimizer)
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
    specs = super(Sampled, cls).getInputSpecification()
    # initialization: add sampling-based options
    whenSolnExpEnum = InputTypes.makeEnumType('whenWriteEnum', 'whenWriteType', ['final', 'every'])
    init = specs.getSub('samplerInit')
    #specs.addSub(init)
    limit = InputData.parameterInputFactory('limit', contentType=InputTypes.IntegerType)
    write = InputData.parameterInputFactory('writeSteps', contentType=whenSolnExpEnum)
    init.addSub(limit)
    init.addSub(write)
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    Optimizer.__init__(self)
    ## Instance Variable Initialization
    # public
    self.limit = None
    self.type = 'Sampled Optimizer'
    # _protected
    self._writeSteps = 'final'
    self._submissionQueue = deque() # TODO change to Queue.Queue if multithreading samples
    self._stepCounter = {}
    self._stepTracker = {}          # action tracking: what is collected, what needs collecting?
    self._optPointHistory = {}      # by traj, is a deque (-1 is most recent)
    self._maxHistLen = 2            # FIXME who should set this?
    # self._nextTrajToConsider = 0  # which is the next trajectory to check up on?
    # __private
    # additional methods
    ## register adaptive sample identification criteria
    self.registerIdentifier('step') # the step within the action

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    Optimizer.handleInput(self, paramInput)
    # samplerInit
    init = paramInput.findFirst('samplerInit')
    if init is not None:
      # limit
      limit = init.findFirst('limit')
      if limit is not None:
        self.limit = limit.value
      # writeSteps
      writeSteps = init.findFirst('writeSteps')
      if writeSteps is not None:
        self._writeSteps = writeSteps.value

    # Acceptance
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

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    Optimizer.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)

  ###############
  # Run Methods #
  ###############
  @abc.abstractmethod
  def _useRealization(self, info, rlz, optVal):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ In, optVal, float, value of objective variable (corrected for min/max)
      @ Out, None
    """

  @abc.abstractmethod
  def checkConvergence(self, traj):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory to consider
      @ Out, None? FIXME
    """

  @abc.abstractmethod
  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """

  def _initializeStep(self, traj):
    """
      Initializes a new step in the optimization process.
      @ In, traj, int, the trajectory of interest
      @ Out, None
    """
    self._stepCounter[traj] += 1
    self._stepTracker[traj] = {'opt': None} # add entries in inheritors as needed

  def amIreadyToProvideAnInput(self):
    """
      This is a method that should be called from any user of the optimizer before requiring the generation of a new input.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of model evaluation, convergence criteria met, etc.
      @ In, None
      @ Out, ready, bool, indicating the readiness of the optimizer to generate a new input.
    """
    # if any trajectories are still active, we're ready to provide an input
    ready = Optimizer.amIreadyToProvideAnInput(self)
    # we're not ready yet if we don't have anything in queue
    ready = ready and len(self._submissionQueue)
    return ready

  def localGenerateInput(self, model, inp):
    """
      TODO
    """
    # get point from stack
    point, info = self._submissionQueue.popleft()
    point = self.denormalizeData(point)
    # assign a tracking prefix
    prefix = self.inputInfo['prefix']
    # register the point tracking information
    self._registerSample(prefix, info)
    # build the point in the way the Sampler expects
    for var in self.toBeSampled: #, val in point.items():
      val = point[var]
      self.values[var] = val # TODO should be np.atleast_1d?
      ptProb = self.distDict[var].pdf(val)
      # sampler-required meta information # TODO should we not require this?
      self.inputInfo['ProbabilityWeight-{}'.format(var)] = ptProb
      self.inputInfo['SampledVarsPb'][var] = ptProb
    self.inputInfo['ProbabilityWeight'] = 1 # TODO assume all weight 1? Not well-distributed samples
    self.inputInfo['PointProbability'] = np.prod([x for x in self.inputInfo['SampledVarsPb'].values()])
    self.inputInfo['SamplerType'] = self.type

  def localFinalizeActualSampling(self, job, model, inp):
    """
      Runs after each sample is collected from the JobHandler.
      @ In, job, Runner instance, job runner entity
      @ In, model, Model instance, RAVEN model that was run
      @ In, inp, list, generated inputs for run
      @ Out, None
    """
    Optimizer.localFinalizeActualSampling(self, job, model, inp)
    # TODO should this be an Optimizer class action instead of Sampled?
    # collect finished job
    prefix = job.getMetadata()['prefix']
    # If we're not looking for the prefix, don't bother with using it
    ## this usually happens if we've cancelled the run but it's already done
    if not self.stillLookingForPrefix(prefix):
      return
    if job.getReturnCode() != 0: # TODO shouldn't this be "if job failed"?
      raise NotImplementedError # FIXME   handle failed runs
    # FIXME implicit constraints probable should be handled here too
    # get information and realization, and update trajectories
    info = self.getIdentifierFromPrefix(prefix, pop=True)
    _, full = self._targetEvaluation.realization(matchDict={'prefix': prefix})
    # trim down opt point to the useful parts
    # TODO making a new dict might be costly, maybe worth just passing whole point?
    ## testing suggests no big deal on smaller problem
    rlz = dict((var, full[var]) for var in (list(self.toBeSampled.keys()) + [self._objectiveVar] + list(self.dependentSample.keys())))
    optVal = self._collectOptValue(rlz)
    rlz = self.normalizeData(rlz)
    self._useRealization(info, rlz, optVal)

  ###################
  # Utility Methods #
  ###################
  # * * * * * * * * * * * *
  # Constraint Handling
  def _handleExplicitConstraints(self, proposed, previous, pointType):
    """
      Considers all explicit (i.e. input-based) constraints
      @ In, traj, int, identifier
      @ In, proposed, dict, NORMALIZED sample opt point
      @ In, proposed, dict, NORMALIZED previous opt point
      @ In, pointType, string, type of point to handle constraints for
      @ Out, normed, dict, suggested NORMALIZED contraint-handled point
      @ Out, modded, bool, whether point was modified or not
    """
    denormed = self.denormalizeData(proposed)
    # check and fix boundaries
    denormed, boundaryModded = self._applyBoundaryConstraints(denormed)
    normed = self.normalizeData(denormed)
    # fix functionals
    normed, funcModded = self._applyFunctionalConstraints(normed, previous)
    modded = boundaryModded or funcModded
    # passFuncs = self._checkFunctionalConstraints(denormed)
    # # fix functionals
    # while not passFuncs:
    #   modded = True
    #   # try to find acceptable point
    #   new = self._applyFunctionalConstraints(denormed, )
    #   # check functionals
    #   passFuncs = self._checkFunctionalConstraints(denormed)
    #   if passFuncs:
    #     # check boundaries
    #     denormed, _ = self._applyBoundaryConstraints(denormed)
    #   TODO
    if funcModded:
      print('DEBUGG modified by functional constraint!')
    elif boundaryModded:
      print('DEBUGG modified by boundary constraint!')
    else:
      print('DEBUGG unmodified by constraints!')
    # return
    if modded:
      #normed = self.normalizeData(denormed)
      print('DEBUGG had some modding! New:', normed)
    #else:
    #  normed = proposed
    return normed, modded

  def _checkFunctionalConstraints(self, point):
    """
      Checks that provided point does not violate functional constraints
      @ In, traj, int, identifier
      @ In, point, dict, suggested point to submit (denormalized)
      @ In, info, dict, data about suggested point
      @ Out, allOkay, bool, False if violations found else True
    """
    allOkay = True
    inputs = dict(point)
    inputs.update(self.constants)
    # TODO add functional evaluations?
    for constraint in self._constraintFunctions:
      okay = constraint.evaluate('constrain', inputs)
      if not okay:
        self.raiseADebug('Functional constraint "{n}" was violated!'.format(n=constraint.name))
        self.raiseADebug(' ... point:', point)
      allOkay &= okay
    # FIXME TODO check functions
    return allOkay

  def _applyBoundaryConstraints(self, point):
    """
      Checks and fixes boundary constraints of variables in "point" -> DENORMED point expected!
      @ In, point, dict, potential point against which to check
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    # TODO should some of this go into the parent Optimizer class, such as the boundary acquiring?
    modded = False
    for var in self.toBeSampled:
      dist = self.distDict[var]
      val = point[var]
      lower = dist.lowerBound
      upper = dist.upperBound
      if lower is None:
        lower = -np.inf
      if upper is None:
        upper = np.inf
      if val < lower:
        self.raiseADebug(' BOUNDARY VIOLATION "{}" suggested value: {:1.3e} lower bound: {:1.3e} under by {:1.3e}'
                         .format(var, val, lower, lower - val))
        self.raiseADebug(' ... -> for point {}'.format(point))
        point[var] = lower
        modded = True
      elif val > upper:
        self.raiseADebug(' BOUNDARY VIOLATION "{}" suggested value: {:1.3e} upper bound: {:1.3e} over by {:1.3e}'
                         .format(var, val, upper, val - upper))
        self.raiseADebug(' ... -> for point {}'.format(point))
        point[var] = upper
        modded = True
    return point, modded

  @abc.abstractmethod
  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
  # END constraint handling
  # * * * * * * * * * * * *

  # * * * * * * * * * * * * * * * *
  # Resolving potential opt points
  def _resolveNewOptPoint(self, traj, rlz, optVal, info):
    """
      Consider and store a new optimal point
      @ In, traj, int, trajectory for this new point
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ In, optVal, float, value of objective variable (corrected for min/max)
    """
    self.raiseADebug('*'*80)
    self.raiseADebug('Trajectory {} iteration {} resolving new opt point ...'.format(traj, info['step']))
    # note the collection of the opt point
    self._stepTracker[traj]['opt'] = (rlz, info)
    # FIXME check implicit constraints? Function call, - Jia
    acceptable, old = self._checkAcceptability(traj, rlz, optVal)
    converged = self._updateConvergence(traj, rlz, old, acceptable)
    self._updatePersistence(traj, converged, optVal)
    # NOTE: the solution export needs to be updated BEFORE we run rejectOptPoint or extend the opt
    #       point history.
    self._updateSolutionExport(traj, rlz, acceptable) # NOTE: only on opt point!
    self.raiseADebug('*'*80)
    # decide what to do next
    if acceptable in ['accepted', 'first']:
      # record history
      self._optPointHistory[traj].append((rlz, info))
      self._stepCounter[traj] += 1
      # nothing else to do but wait for the grad points to be collected
    elif acceptable == 'rejected':
      self._rejectOptPoint(traj, info, old)
    else: # e.g. rerun
      pass # nothing to do, just keep moving

  # support methods for _resolveNewOptPoint
  @abc.abstractmethod
  def _checkAcceptability(self, traj, opt, optVal):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
    """

  @abc.abstractmethod
  def _updateConvergence(self, traj, rlz, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, acceptable, str, condition of point
      @ Out, converged, bool, True if converged on ANY criteria
    """

  @abc.abstractmethod
  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """

  @abc.abstractmethod
  def _updateSolutionExport(self, traj, rlz, acceptable):
    """
      Prints information to the solution export.
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ Out, None
    """

  @abc.abstractmethod
  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """
  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  def _cancelAssociatedJobs(self, traj, step=None):
    """
      Queues jobs to be cancelled based on opt run
      @ In, traj, int, trajectory identifier
      @ In, step, int, optional, iteration identifier
      @ Out, None
    """
    # generic tracking info: we want this trajectory, this step, all purposes
    ginfo = {'traj': traj}
    if step is not None:
      ginfo['step'] = step
    # remove them from the submission queue
    toRemove = []
    # NOTE use a queue lock here if taking samples in multithreading (not currently true)
    for point, info in self._submissionQueue:
      if all(item in info.items() for item in ginfo.items()):
        toRemove.append((point, info))
    for x in toRemove:
      try:
        self._submissionQueue.remove(x)
      except ValueError:
        pass # it must have been submitted since we flagged it for removal
    # get prefixes of already-submitted jobs; get all matches, and pop them so we don't track them anymore
    prefixes = self.getPrefixFromIdentifier(ginfo, getAll=True, pop=True)
    self.raiseADebug('Canceling grad jobs for traj "{}" iteration "{}":'.format(traj, 'all' if step is None else step), prefixes)
    self._jobsToEnd.extend(prefixes)

  def initializeTrajectory(self, traj=None):
    """
      Sets up a new trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, trajectory number
    """
    traj = Optimizer.initializeTrajectory(self, traj=traj)
    self._optPointHistory[traj] = deque(maxlen=self._maxHistLen)
    self._stepCounter[traj] = -1
    self._initializeStep(traj)
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
    Optimizer._closeTrajectory(self, traj, action, reason, value)
    # kill jobs associated with trajectory
    self._cancelAssociatedJobs(traj)
