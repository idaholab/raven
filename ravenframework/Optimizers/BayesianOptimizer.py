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

#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ..utils import InputData, InputTypes, mathUtils
from .RavenSampled import RavenSampled

#Internal Modules End--------------------------------------------------------------------------------


class BayesianOptimizer(RavenSampled):
  """
   
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
    # TODO need to include StepManipulators and GradientApproximators solution export entries as well!
    # -> but really should only include active ones, not all of them. This seems like it should work
    #    when the InputData can scan forward to determine which entities are actually used.
    new['amp_{VAR}'] = 'amplitude associated to each variable used to compute step size based on cooling method and the corresponding next neighbor'
    new ['delta_{VAR}'] = 'step size associated to each variable'
    new['Temp'] = 'temperature at current state'
    new['fraction'] = 'current fraction of the max iteration limit'
    ok.update(new)
    return ok
  
  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    RavenSampled.__init__(self)
    self._iteration = {0:0}

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)

    meta = ['batchId']
    self.addMetaKeys(meta)
    # queue up the first run for each trajectory
    for traj, init in enumerate(self._initialValues):
      self._submitRun(init,traj,1)

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
    info.update({'traj': traj,
                  'step': step
                })
    # NOTE: explicit constraints have been checked before this!
    self.raiseADebug(f'Adding run to queue: {self.denormalizeData(point)} | {info}')
    self._submissionQueue.append((point, info))

  ###############
  # Run Methods #
  ###############
  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization (corrected for min-max)
      @ Out, None
    """
    traj = info['traj']
    step = info['step'] + 1
    optVal = rlz[self._objectiveVar]
    self.raiseADebug(f'Input: ({rlz["x"]},{rlz["y"]})')
    self.raiseADebug(f'Function Value: ({optVal})')
    info['optVal'] = optVal
    self._optPointHistory[traj].append((rlz,info))

    # Generate random input to the sampler
    x_new = np.random.rand()
    y_new = np.random.rand()
    point = {'x':x_new, 'y':y_new}
    self._iteration[traj] += 1
    self._submitRun(point, traj, step)

  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory identifier
      @ In, new, dict, new opt point
      @ In, old, dict, previous opt point
    """
    RavenSampled.checkConvergence(traj, new, old)

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    RavenSampled._checkForImprovement(new, old)

  def _initializeStep(self, traj):
    """
      Initializes a new step in the optimization process.
      @ In, traj, int, the trajectory of interest
      @ Out, None
    """
    self._stepTracker[traj] = {'opt': None}  # add entries in inheritors as needed

  def amIreadyToProvideAnInput(self):
    """
      This is a method that should be called from any user of the optimizer before requiring the generation of a new input.
      This method act as a "traffic light" for generating a new input.
      Reason for not being ready could be for example: exceeding number of model evaluation, convergence criteria met, etc.
      @ In, None
      @ Out, ready, bool, indicating the readiness of the optimizer to generate a new input.
    """
    # if any trajectories are still active, we're ready to provide an input
    ready = RavenSampled.amIreadyToProvideAnInput(self)
    # we're not ready yet if we don't have anything in queue
    ready = ready and len(self._submissionQueue) != 0

    return ready
    
  def localFinalizeActualSampling(self, jobObject, model, myInput):
    """
      Runs after each sample is collected from the JobHandler.
      @ In, jobObject, Runner instance, job runner entity
      @ In, model, Model instance, RAVEN model that was run
      @ In, myInput, list, generated inputs for run
      @ Out, None
    """
    RavenSampled.localFinalizeActualSampling(self, jobObject, model, myInput)

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
    self.__stepCounter[traj] += 1

  def getIteration(self, traj):
    """
      Provides the "generation" or "iteration" of an optimization algorithm.
      The definition of generation is algorithm-specific; this is a utility for tracking only.
      @ In, traj, int, identifier for trajectory
      @ Out, counter, int, iteration of the trajectory
    """
    return self._iteration[traj]

  # * * * * * * * * * * * *
  # Constraint Handling
  def _handleExplicitConstraints(self, proposed, previous, pointType):
    """
      Considers all explicit (i.e. input-based) constraints
      @ In, proposed, dict, NORMALIZED sample opt point
      @ In, previous, dict, NORMALIZED previous opt point
      @ In, pointType, string, type of point to handle constraints for
      @ Out, normed, dict, suggested NORMALIZED constraint-handled point
      @ Out, modded, bool, whether point was modified or not
    """
    denormed = self.denormalizeData(proposed)
    # check and fix boundaries
    denormed, boundaryModded = self._applyBoundaryConstraints(denormed)
    normed = self.normalizeData(denormed)
    # fix functionals
    normed, funcModded = self._applyFunctionalConstraints(normed, previous)
    modded = boundaryModded or funcModded

    return normed, modded

  def _checkFunctionalConstraints(self, point):
    """
      Checks that provided point does not violate functional constraints
      @ In, point, dict, suggested point to submit (denormalized)
      @ Out, allOkay, bool, False if violations found else True
    """
    allOkay = True
    inputs = dict(point)
    inputs.update(self.constants)
    for constraint in self._constraintFunctions:
      okay = constraint.evaluate('constrain', inputs)
      if not okay:
        self.raiseADebug(f'Functional constraint "{constraint.name}" was violated!')
        self.raiseADebug(' ... point:', point)
      allOkay *= okay

    return bool(allOkay)

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
      if val < lower:
        self.raiseADebug(f' BOUNDARY VIOLATION "{var}" suggested value: {val:1.3e} lower bound: {lower:1.3e} under by {lower - val:1.3e}')
        self.raiseADebug(f' ... -> for point {point}')
        point[var] = lower
        modded = True
      elif val > upper:
        self.raiseADebug(f' BOUNDARY VIOLATION "{var}" suggested value: {val:1.3e} upper bound: {upper:1.3e} over by {val - upper:1.3e}')
        self.raiseADebug(f' ... -> for point {point}')
        point[var] = upper
        modded = True

    return point, modded

  def _checkBoundaryConstraints(self, point):
    """
      Checks (NOT fixes) boundary constraints of variables in "point" -> DENORMED point expected!
      @ In, point, dict, potential point against which to check
      @ Out, okay, bool, True if no constraints violated
    """
    okay = True
    for var in self.toBeSampled:
      dist = self.distDict[var]
      val = point[var]
      lower = dist.lowerBound
      upper = dist.upperBound
      if val < lower or val > upper:
        okay = False
        break
    return okay

  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    RavenSampled._applyFunctionalConstraints(suggested, previous)

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

  # * * * * * * * * * * * * * * * *
  # Resolving potential opt points
  def _resolveNewOptPoint(self, traj, rlz, optVal, info):
    """
      Consider and store a new optimal point
      @ In, traj, int, trajectory for this new point
      @ In, info, dict, identifying information about the realization
      @ In, rlz, xr.DataSet, batched realizations
      @ In, optVal, list of floats, values of objective variable
    """
    RavenSampled._resolveNewOptPoint(traj, rlz, optVal, info)

  # support methods for _resolveNewOptPoint
  def _checkAcceptability(self, traj, opt, optVal):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
      @ Out, rejectReason, str, reject reason of opt point, or return None if accepted
    """
    RavenSampled._checkAcceptability(traj, opt, optVal)

  def _updateConvergence(self, traj, new, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, new, dict, new point
      @ In, old, dict, old point
      @ In, acceptable, str, condition of new point
    """
    RavenSampled._updateConvergence(traj, new, old, acceptable)

  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """
    RavenSampled._updatePersistence(traj, converged, optVal)

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
    """
    RavenSampled._rejectOptPoint(traj, info, old)

  def _addToSolutionExport(self, traj, rlz, acceptable):
    """
      Contributes additional entries to the solution export.
      Should be used by inheritors instead of overloading updateSolutionExport
      @ In, traj, int, trajectory which should be written
      @ In, rlz, dict, collected point
      @ In, acceptable, bool, acceptability of opt point
      @ Out, toAdd, dict, additional entries
    """
    return {}

  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  def _cancelAssociatedJobs(self, traj, step=None):
    """
      Queues jobs to be cancelled based on opt run
      @ In, traj, int, trajectory identifier
      @ In, step, int, optional, iteration identifier (unused if not provided)
      @ Out, None
    """
    RavenSampled._cancelAssociatedJobs(traj, step=step)

  def initializeTrajectory(self, traj=None):
    """
      Sets up a new trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, trajectory number
    """
    traj = RavenSampled.initializeTrajectory(self, traj=traj)

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
    RavenSampled._closeTrajectory(self, traj, action, reason, value)