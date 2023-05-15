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
  Simulated Annealing class for global optimization.

  Created Feb,20,2020
  @author: Mohammad Abdo

  References
    ----------
    .. [1] Kirkpatrick, S.; Gelatt Jr, C. D.; Vecchi, M. P. (1983).
        ``Optimization by Simulated Annealing". Science. 220 (4598): 671–680.
    .. [2] P. J. M. van Laarhoven and E. H. L. Aarts, ``Simulated Annealing: Theory
        and Applications", Kluwer Academic Publishers, 1987.
    .. [3] W.H. Press et al., ``Numerical Recipies: The Art of Scientific Computing",
        Cambridge U. Press, 1987.
    .. [4] Tsallis C. ``Possible generalization of Boltzmann-Gibbs
        statistics". Journal of Statistical Physics, 52, 479-487 (1998).
    .. [5] Tsallis C, Stariolo DA. ``Generalized Simulated Annealing."
        Physica A, 233, 395-406 (1996).
    .. [6] Xiang Y, Sun DY, Fan W, Gong XG. ``Generalized Simulated
        Annealing Algorithm and Its Application to the Thomson Model."
        Physics Letters A, 233, 216-220 (1997).
    .. [7] Xiang Y, Gong XG. ``Efficiency of Generalized Simulated
        Annealing". Physical Review E, 62, 4473 (2000).
    .. [8] Xiang Y, Gubian S, Suomela B, Hoeng J. ``Generalized
        Simulated Annealing for Efficient Global Optimization: the GenSA
        Package for R". The R Journal, Volume 5/1 (2013).
    .. [9] Mullen, K. ``Continuous Global Optimization in R". Journal of
        Statistical Software, 60(6), 1 - 45, (2014). DOI:10.18637/jss.v060.i06
    .. [10] V. Granville and M. Krivanek and J–P. Rasson,
        "Simulated annealing: A proof of convergence",
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 1994.
    .. [11] L. Ingber, "Simulated Annealing: Practice versus theory",
        Math. Comput. Modelling, 1993.
    .. [12] S. Kirkpatrick, "Optimization by simulated annealing: Quantitative studies",
        Journal of Statistical Physics, 1983.
    .. [13] M. P. Vecchi and S. Kirkpatrick, "Global wiring by simulated annealing",
        IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 1983.
"""
# External Modules----------------------------------------------------------------------------------
from collections import deque, defaultdict
import numpy as np
# External Modules End------------------------------------------------------------------------------

# Internal Modules----------------------------------------------------------------------------------
from ..utils import mathUtils, randomUtils, InputData, InputTypes
from .RavenSampled import RavenSampled
from .stepManipulators import NoConstraintResolutionFound
# Internal Modules End------------------------------------------------------------------------------

class SimulatedAnnealing(RavenSampled):
  """
    This class performs simulated annealing optimization utilizing several cooling scheduling methods.
    Cooling Schedule includes Boltzmann, Exponential, Cauchy, and VeryFast cooling.
    The Simulated Annealing optimizer is a metaheuristic approach to perform a global
    search in large design spaces. The methodology rose from statistical physics
    and was inspired by metallurgy where it was found that fast cooling might lead
    to smaller and defected crystals, and that reheating and slowly controling cooling
    will lead to better states. This allows climbing to avoid being stuck in local minima
    and hence facilitates finding the global minima for non-convex probloems.
  """
  convergenceOptions = {'objective': r""" provides the desired value for the convergence criterion of the objective function
                        ($\epsilon^{obj}$), i.e., convergence is reached when: $$ |newObjevtive - oldObjective| \le \epsilon^{obj}$$.
                        \default{1e-6}, if no criteria specified""",
                        'temperature': r""" provides the desired value for the convergence creiteron of the system temperature,
                        ($\epsilon^{temp}$), i.e., convergence is reached when: $$T \le \epsilon^{temp}$$.
                        \default{1e-10}, if no criteria specified"""}
  coolingOptions = {#'linear': {'beta':r"""slope"""},
                    'exponential':{'alpha':r"""slowing down constant, should be between 0,1 and preferable very close to 1. \default{0.94}"""},
                    #'fast':{'c':r"""decay constant, \default{1.0}"""},
                    'veryfast':{'c':r"""decay constant, \default{1.0}"""},
                    'cauchy':{'d':r"""bias, \default{1.0}"""},
                    'boltzmann':{'d':r"""bias, \default{1.0}"""}}
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = super(SimulatedAnnealing, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{SimulatedAnnealing} optimizer is a metaheuristic approach
                            to perform a global search in large design spaces. The methodology rose
                            from statistical physics and was inspired by metallurgy where
                            it was found that fast cooling might lead to smaller and defected crystals,
                            and that reheating and slowly controlling cooling will lead to better states.
                            This allows climbing to avoid being stuck in local minima and hence facilitates
                            finding the global minima for non-convex problems.
                            More information can be found in: Kirkpatrick, S.; Gelatt Jr, C. D.; Vecchi, M. P. (1983).
                            ``Optimization by Simulated Annealing". Science. 220 (4598): 671–680."""

    # convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True,
        printPriority=108,
        descr=r"""a node containing the desired convergence criteria for the optimization algorithm.
              Note that convergence is met when any one of the convergence criteria is met. If no convergence
              criteria are given, then the defaults are used.""")
    specs.addSub(conv)
    for name, descr in cls.convergenceOptions.items():
      conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType,descr=descr,printPriority=108  ))

    # Persistance
    conv.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType,
        printPriority = 109,
        descr=r"""provides the number of consecutive times convergence should be reached before a trajectory
              is considered fully converged. This helps in preventing early false convergence."""))

    # Cooling Schedule
    coolingSchedule = InputData.parameterInputFactory('coolingSchedule',
        printPriority=109,
        descr=r""" The function governing the cooling process. Currently, user can select between,"""
                  # \xmlString{linear},
                  +r"""\xmlString{exponential},
                  \xmlString{cauchy},
                  \xmlString{boltzmann},"""
                  # \xmlString{fast},
                  +r"""or \xmlString{veryfast}.\\ \\"""
                  #In case of \xmlString{linear} is provided, The cooling process will be governed by: $$ T^{k} = T^0 - 0.1 * k$$
                  +r"""In case of \xmlString{exponential} is provided, The cooling process will be governed by: $$ T^{k} = T^0 * \alpha^k$$
                  In case of \xmlString{boltzmann} is provided, The cooling process will be governed by: $$ T^{k} = \frac{T^0}{log(k + d)}$$
                  In case of \xmlString{cauchy} is provided, The cooling process will be governed by: $$ T^{k} = \frac{T^0}{k + d}$$"""
                  #In case of \xmlString{fast} is provided, The cooling process will be governed by: $$ T^{k} = T^0 * \exp(-ck)$$
                  +r"""In case of \xmlString{veryfast} is provided, The cooling process will be governed by: $$ T^{k} =  T^0 * \exp(-ck^{1/D}),$$
                  where $D$ is the dimentionality of the problem (i.e., number of optimized variables), $k$ is the number of the current iteration
                  $T^{0} = \max{(0.01,1-\frac{k}{\xmlNode{limit}})}$ is the initial temperature, and $T^{k}$ is the current temperature
                  according to the specified cooling schedule.
                  \default{exponential}.""")
    specs.addSub(coolingSchedule)

    for schedule, param in cls.coolingOptions.items(): # FIXME: right now this allows multiple cooling schedule, which should be fixed as soon as
                                                              # InputData can allow having list of subnodes
      sch = InputData.parameterInputFactory(schedule, contentType=InputTypes.StringType, descr=schedule+' cooling schedule')
      for par, descr in param.items():
        sch.addSub(InputData.parameterInputFactory(par, contentType=InputTypes.FloatType,descr=descr))
      coolingSchedule.addSub(sch)

    return specs

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, cls, the class for which we are retrieving the solution export
      @ Out, ok, dict, {varName: description} for valid solution export variable names
    """
    # cannot be determined before run-time due to variables and prefixes.
    ok = super(SimulatedAnnealing, cls).getSolutionExportVariableNames()
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
    self._convergenceCriteria = defaultdict(mathUtils.giveZero) # names and values for convergence checks
    self._acceptHistory = {}                                    # acceptability
    self._acceptRerun = {}                                      # by traj, if True then override accept for point rerun
    self._convergenceInfo = {}                                  # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = 0                               # consecutive persistence required to mark convergence
    self.T = None                                               # current temperature
    self._coolingMethod = None                                  # initializing cooling method
    self._coolingParameters = {}                                # initializing the cooling schedule parameters
    self.info = {}

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    RavenSampled.handleInput(self, paramInput)
    # Convergence Criterion
    convNode = paramInput.findFirst('convergence')
    if convNode is not None:
      for sub in convNode.subparts:
        if sub.getName() == 'persistence':
          self._requiredPersistence = sub.value
        else:
          self._convergenceCriteria[sub.name] = sub.value
    if not self._convergenceCriteria:
      self.raiseAWarning('No convergence criteria given; using defaults.')
      self._convergenceCriteria['objective'] = 1e-6
      self._convergenceCriteria['temperature'] = 1e-10
    # same point is ALWAYS a criterion
    self._convergenceCriteria['samePoint'] = -1 # For simulated Annealing samePoint convergence
                                                # should not be one of the stopping criteria
    # set persistence to 1 if not set
    if self._requiredPersistence is None:
      self.raiseADebug('No persistence given; setting to 1.')
      self._requiredPersistence = 1
    # Cooling Schedule
    coolingNode = paramInput.findFirst('coolingSchedule')
    if coolingNode is None:
      self._coolingMethod = 'exponential'
    else:
      for sub in coolingNode.subparts:
        self._coolingMethod = sub.name
        for subSub in sub.subparts:
          self._coolingParameters = {subSub.name: subSub.value}

    #defaults
    if not self._coolingMethod:
      self._coolingMethod = 'exponential'

    if not self._coolingParameters:
      self._coolingParameters['alpha'] = 0.94
      self._coolingParameters['beta'] = 0.1
      self._coolingParameters['c'] = 1.0
      self._coolingParameters['d'] = 1.0

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    for var in self.toBeSampled:
      self.info['amp_'+var] = None
      self.info['delta_'+var] = None
    # queue up the first run for each trajectory
    for traj, init in enumerate(self._initialValues):
      self._submitRun(init,traj,self.getIteration(traj))

  def initializeTrajectory(self, traj=None):
    """
      Handles the generation of a trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, new trajectory number
    """
    traj = RavenSampled.initializeTrajectory(self)
    self._acceptHistory[traj] = deque(maxlen=self._maxHistLen)
    self._acceptRerun[traj] = False
    self._convergenceInfo[traj] = {'persistence': 0}
    for criteria in self._convergenceCriteria:
      self._convergenceInfo[traj][criteria] = False

    return traj

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
  # END queuing Runs
  # * * * * * * * * * * * * * * * *

  ###############
  # Run Methods #
  ###############
  def _useRealization(self, info, rlz):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ In, optVal, float, value of objective variable (corrected for min/max)
      @ Out, None
    """
    traj = info['traj']
    info['optVal'] = rlz[self._objectiveVar]
    self.incrementIteration(traj)
    self._resolveNewOptPoint(traj, rlz, rlz[self._objectiveVar], info)
    if self._stepTracker[traj]['opt'] is None:
      # revert to the last accepted point
      rlz = self._optPointHistory[traj][-1][0]
      info = self._optPointHistory[traj][-1][1]
      info['step'] = self.getIteration(traj)
    iteration = int(self.getIteration(traj) + 1) # Is that ok or should we always keep the traj in case I have multiple trajectories in parallel?
    fraction = iteration/self.limit
    currentPoint = self._collectOptPoint(rlz)
    T0 = self._temperature(fraction)
    self.T = self._coolingSchedule(iteration, T0)
    if traj in self._activeTraj:
      newPoint = self._nextNeighbour(rlz, fraction)
      # check new opt point against constraints
      try:
        suggested, _ = self._handleExplicitConstraints(newPoint, currentPoint, 'opt')
      except NoConstraintResolutionFound:
        # we've tried everything, but we just can't hack it
        self.raiseAMessage(f'Optimizer "{self.name}" trajectory {traj} was unable to continue due to functional or boundary constraints.')
        self._closeTrajectory(traj, 'converge', 'no constraint resolution', newPoint[self._objectiveVar])
        return
      self._submitRun(suggested, traj, self.getIteration(traj))

  def flush(self):
    """
      Reset Optimizer attributes to allow rerunning a workflow
      @ In, None
      @ Out, None
    """
    super().flush()
    self.T = None
    self.info = {}

  # * * * * * * * * * * * * * * * *
  # Convergence Checks
  convFormat = RavenSampled.convFormat

  # NOTE checkConvSamePoint has a different call than the others
  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory to consider
      @ In, new, dict, new point
      @ In, old, dict, old point
      @ Out, any(convs.values()), bool, True of any of the convergence criteria was reached
      @ Out, convs, dict, on the form convs[conv] = bool, where conv is in self._convergenceCriteria
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

  def _checkConvSamePoint(self, new, old):
    """
      Checks for a repeated same point
      @ In, new, dict, new opt point
      @ In, old, dict, old opt point
      @ Out, converged, bool, convergence state
    """
    # TODO diff within tolerance? Exactly equivalent seems good for now
    same = list(abs(new[var] - old[var])==self._convergenceCriteria['samePoint'] for var in self.toBeSampled)
    converged = all(same)
    self.raiseADebug(self.convFormat.format(name='same point',
                                            conv=str(converged),
                                            got=sum(same),
                                            req=len(same)))

    return converged

  def _checkConvObjective(self, traj):
    """
      Checks the change in objective for convergence
      @ In, traj, int, trajectory identifier
      @ Out, converged, bool, convergence state
    """
    if len(self._optPointHistory[traj]) < 2 or (self._convergenceCriteria['objective'] < 0):
      return False
    o1, _ = self._optPointHistory[traj][-1]
    o2, _ = self._optPointHistory[traj][-2]
    delta = o2[self._objectiveVar]-o1[self._objectiveVar]
    converged = abs(delta) < self._convergenceCriteria['objective']
    self.raiseADebug(self.convFormat.format(name='objective',
                                            conv=str(converged),
                                            got=delta,
                                            req=self._convergenceCriteria['objective']))

    return converged

  def _checkConvTemperature(self, traj):
    """
      Checks temperature for the current state for convergence
      @ In, traj, int, trajectory identifier
      @ Out, converged, bool, convergence state
    """
    converged = abs(self.T) <= self._convergenceCriteria['temperature']
    self.raiseADebug(self.convFormat.format(name='temperature',
                                            conv=str(converged),
                                            got=self.T,
                                            req=self._convergenceCriteria['temperature']))

    return converged

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    # This is not required for simulated annealing as it's handled in the probabilistic acceptance criteria
    # But since it is an abstract method it has to exist
    return True

  def _checkAcceptability(self, traj, opt, optVal, info):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ In, info, dict, meta information about the opt point
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
      @ Out, rejectReason, str, reject reason of opt point, or return None if accepted
    """
    # Check acceptability
    # NOTE: if self._optPointHistory[traj]: -> faster to use "try" for all but the first time
    try:
      old, _ = self._optPointHistory[traj][-1]
      oldVal = old[self._objectiveVar]
      # check if same point
      self.raiseADebug(f' ... change: {opt[self._objectiveVar]-oldVal:1.3e} new objective: {opt[self._objectiveVar]:1.6e} old objective: {oldVal:1.6e}')
      # if this is an opt point rerun, accept it without checking.
      if self._acceptRerun[traj]:
        acceptable = 'rerun'
        self._acceptRerun[traj] = False
      elif all(opt[var] == old[var] for var in self.toBeSampled):
        # this is the classic "same point" trap; we accept the same point, and check convergence later
        acceptable = 'accepted'
      else:
        if self._acceptabilityCriterion(oldVal,opt[self._objectiveVar])>randomUtils.random(dim=1, samples=1): # TODO replace it back
          acceptable = 'accepted'
        else:
          acceptable = 'rejected'
    except IndexError:
      # if first sample, simply assume it's better!
      acceptable = 'first'
      old = None
    self._acceptHistory[traj].append(acceptable)
    self.raiseADebug(f' ... {acceptable}!')

    return acceptable, old, 'None'

  def _acceptabilityCriterion(self,currentObjective,newObjective):
    """
      Check if new opt point is acceptably better than the old one
      @ In, currentObjective, float, the current value of the objective function (i.e., current energy)
      @ In, newObjective, float, the value of the objective function at the new candidate
      @ Out, prob, float, the acceptance probability
    """
    kB = 1

    if newObjective <= currentObjective:
      prob = 1
    else:
      deltaE = newObjective - currentObjective
      prob = min(1,np.exp(-deltaE/(kB * self.T)))

    return prob

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
      convDict = dict((var, False) for var in self._convergenceInfo[traj])
    self._convergenceInfo[traj].update(convDict)

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
      self.raiseADebug(f'Trajectory {traj} has converged successfully {self._convergenceInfo[traj]["persistence"]} time(s)!')
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
    # meta variables
    toAdd = {'Temp': self.T,
             'fraction': self.getIteration(traj)/self.limit
            }

    for var in self.toBeSampled:
      toAdd[f'amp_{var}'] = self.info[f'amp_{var}']
      toAdd[f'delta_{var}'] = self.info[f'delta_{var}']

    for var, val in self.constants.items():
      toAdd[var] = val

    toAdd = dict((key, np.atleast_1d(val)) for key, val in toAdd.items())
    for key, val in self._convergenceInfo[traj].items():
      toAdd[f'conv_{key}'] = bool(val)

    return toAdd

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
      elif '{VAR}' in template:
        new.extend([template.format(VAR=var) for var in self.toBeSampled])
      else:
        new.append(template)

    return set(new)

  def _rejectOptPoint(self, traj, info, old):
    """
      Having rejected the suggested opt point, take actions so we can move forward
      @ In, traj, int, identifier
      @ In, info, dict, meta information about the opt point
      @ In, old, dict, previous optimal point (to resubmit)
      @ Out, none
    """
    self._cancelAssociatedJobs(info['traj'], step=info['step'])
    # initialize a new step
    self._initializeStep(traj)
  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  def _applyFunctionalConstraints(self, suggested, previous):
    """
      applies functional constraints of variables in "suggested" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    # assume no modifications until proved otherwise
    modded = False
    # are we violating functional constraints?
    passFuncs = self._checkFunctionalConstraints(self.denormalizeData(suggested))
    # while in violation of constraints ...
    tries = 500
    while not passFuncs:
      modded = True
      #  try to find new acceptable point
      denormed = self.denormalizeData(suggested)
      suggested, _= self._fixFuncConstraintViolations(suggested)
      denormed = self.denormalizeData(suggested)
      self.raiseADebug(f' ... suggested new opt {denormed}')
      passFuncs = self._checkFunctionalConstraints(denormed)
      tries -= 1
      if tries == 0:
        self.raiseAnError(NotImplementedError, 'No acceptable point findable! Now what?')

    return suggested, modded

  ###########
  # Utility Methods #
  ###########
  def _temperature(self, fraction):
    """
      A utility function to compute the initial temperature
      currently it is just a function of how far in the process are we
      @ In, fraction, float, the current iteration divided by the iteration limit i.e., $\frac{iteration}{Limit}$
      @ Out, _temperature, float, initial temperature, i.e., $T0 = max(0.01,1-fraction) $
    """
    return max(0.01, 1 - fraction)

  def _coolingSchedule(self, iteration, T0):
    """
      A utility function to compute the current cooled state temperature
      based on the user-selected cooling schedule methodology
      @ In, iteration, int, the iteration number
      @ In, T0, float, The previous temperature before cooling
      @ Out, _coolingSchedule, float, the cooled state temperature i.e., $T^{k} = f(T^0, coolingSchedule);$ where k is the iteration number
    """
    coolType = self._coolingMethod
    if coolType in ['exponential','geometric']:
      alpha = self._coolingParameters['alpha']
      return alpha ** iteration * T0
    elif coolType == 'boltzmann':
      d = self._coolingParameters['d']
      return T0/(np.log10(iteration + d))
    elif coolType == 'veryfast':
      c = self._coolingParameters['c']
      return np.exp(-c*iteration**(1/len(self.toBeSampled.keys()))) * T0
    elif coolType == 'cauchy':
      d = self._coolingParameters['d']
      return T0/(iteration + d)
    else:
      self.raiseAnError(NotImplementedError, 'cooling schedule type not implemented.')

  def _nextNeighbour(self, rlz, fraction=1):
    r"""
      Perturbs the state to find the next random neighbor based on the cooling schedule
      @ In, rlz, dict, current realization
      @ In, fraction, float, optional, the current iteration divided by the iteration limit i.e., $\frac{iteration}{Limit}$
      @ Out, nextNeighbour, dict, the next random state

      for exponential cooling:
      .. math::

          fraction = \\frac{iteration}{Limit}

          amp = 1-fraction

          delta = \\frac{-amp}{2} + amp * r

      where :math: `r \sim \mathcal{U}(0,1)`

      for boltzmann cooling:
      .. math::

          amp = min(\\sqrt(T), \\frac{1}{3*alpha}

          delta = r * alpha * amp

      where :math: `r \\sim \\mathcal{N}(0,1)`

      for cauchy cooling:
      .. math::

          amp = r

          delta = alpha * T * tan(amp)

      where :math: `r \\sim \\mathcal{U}(-\\pi,\\pi)`

      for veryfast cooling:
      .. math::

          amp = r

          delta = \\sign(amp-0.5)*T*((1.0+\\frac{1.0}{T})^{\\abs{2*amp-1}-1.0}

      where :math: `r \\sim \\mathcal{U}(0,1)`
    """
    nextNeighbour = {}
    D = len(self.toBeSampled.keys())
    alpha = 0.94
    if self._coolingMethod in ['exponential', 'geometric']:
      amp = ((fraction)**-1) / 20
      r = randomUtils.random(dim=D, samples=1)
      delta = (-amp/2.)+ amp * r
    elif self._coolingMethod == 'boltzmann':
      amp = min(np.sqrt(self.T), 1/3.0/alpha)
      delta =  randomUtils.randomNormal(size=D)*alpha*amp
    elif self._coolingMethod == 'veryfast':
      amp = randomUtils.random(dim=D, samples=1)
      delta = np.sign(amp-0.5)*self.T*((1+1.0/self.T)**abs(2*amp-1)-1.0)
    elif self._coolingMethod == 'cauchy':
      amp = (np.pi - (-np.pi))*randomUtils.random(dim=D, samples=1)-np.pi
      delta = alpha*self.T*np.tan(amp)
    for i,var in enumerate(self.toBeSampled.keys()):
      nextNeighbour[var] = rlz[var] + delta[i]
      self.info['amp_'+var] = amp
      self.info['delta_'+var] = delta[i]
    self.info['fraction'] = fraction

    return nextNeighbour

  def _fixFuncConstraintViolations(self,suggested):
    """
      fixes functional constraints of variables in "suggested"
      and finds the new point that does not violate the constraints
      @ In, suggested, dict, potential point to apply constraints to
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    fraction = self.info['fraction']
    new = self._nextNeighbour(suggested,fraction)
    point, modded = self._handleExplicitConstraints(new, suggested, 'opt')

    return point, modded

  ##############
  # Destructor #
  ##############
  def __del__(self):
    """
      Destructor.
      @ In, None
      @ Out, None
    """
    return
