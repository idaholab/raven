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
    .. [2] Tsallis C. ``Possible generalization of Boltzmann-Gibbs
        statistics". Journal of Statistical Physics, 52, 479-487 (1998).
    .. [3] Tsallis C, Stariolo DA. ``Generalized Simulated Annealing."
        Physica A, 233, 395-406 (1996).
    .. [4] Xiang Y, Sun DY, Fan W, Gong XG. ``Generalized Simulated
        Annealing Algorithm and Its Application to the Thomson Model."
        Physics Letters A, 233, 216-220 (1997).
    .. [5] Xiang Y, Gong XG. ``Efficiency of Generalized Simulated
        Annealing". Physical Review E, 62, 4473 (2000).
    .. [6] Xiang Y, Gubian S, Suomela B, Hoeng J. ``Generalized
        Simulated Annealing for Efficient Global Optimization: the GenSA
        Package for R". The R Journal, Volume 5/1 (2013).
    .. [7] Mullen, K. ``Continuous Global Optimization in R". Journal of
        Statistical Software, 60(6), 1 - 45, (2014). DOI:10.18637/jss.v060.i06
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import abc
import math
import matplotlib.pyplot as plt
from collections import deque, defaultdict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils, randomUtils, InputData, InputTypes, mathUtils
from .RavenSampled import RavenSampled
from .stepManipulators import knownTypes as stepKnownTypes
from .stepManipulators import returnInstance as stepReturnInstance
from .stepManipulators import returnClass as stepReturnClass
from .stepManipulators import NoConstraintResolutionFound
from .acceptanceConditions import knownTypes as acceptKnownTypes
from .acceptanceConditions import returnInstance as acceptReturnInstance
from .acceptanceConditions import returnClass as acceptReturnClass
#Internal Modules End--------------------------------------------------------------------------------
# utility function for defaultdict
def giveZero():
  """
    Utility function for defaultdict to 0
    @ In, None
    @ Out, giveZero, int, zero
  """
  return 0

class SimulatedAnnealing(RavenSampled):
  """
  This class performs simulated annealing optimization
  """
  convergenceOptions = {'objective': r""" provides the desired value for the convergence criterion of the objective function
                        ($\epsilon^{obj}$), i.e., convergence is reached when: $$ |newObjevtive - oldObjective| \le \epsilon^{obj}$$.
                        \default{1e-6}, if no criteria specified""",
                        'temperature': r""" provides the desired value for the convergence creiteron of the system temperature,
                        ($\epsilon^{temp}$), i.e., convergence is reached when: $$T \le \epsilon^{temp}$$.
                        \default{1e-10}, if no criteria specified"""}
  coolingOptions = ['linear',
                    'exponential',
                    'fast',
                    'veryfast',
                    'cauchy',
                    'boltzmann']
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
    specs = super(SimulatedAnnealing, cls).getInputSpecification()
    specs.description = r"""The \xmlNode{SimulatedAnnealing} optimizer is a metaheuristic approach
                            to perform a global search in large design spaces. The methodology rose
                            from statistical physics and was inspitred by metallurgy where
                            it was found that fast cooling might lead to smaller and defected crystals,
                            and that reheating and slowly controling cooling will lead to better states.
                            This allows climbing to avoid being stuck in local minima and hence facilitates
                            finding the global minima for non-convex probloems.
                            More information can be found in: Kirkpatrick, S.; Gelatt Jr, C. D.; Vecchi, M. P. (1983).
                            ``Optimization by Simulated Annealing". Science. 220 (4598): 671–680."""
    # initialization: add sampling-based options
    whenSolnExpEnum = InputTypes.makeEnumType('whenWriteEnum', 'whenWriteType', ['final', 'every'])
    init = specs.getSub('samplerInit')
    specs.addSub(init)
    limit = InputData.parameterInputFactory('limit', contentType=InputTypes.IntegerType)
    write = InputData.parameterInputFactory('writeSteps', contentType=whenSolnExpEnum)
    init.addSub(limit)
    init.addSub(write)

    # acceptance conditions
    accept = InputData.parameterInputFactory('acceptance', strictMode=True,
                        descr=r"""a required node containing the information about the acceptability creterion
                        for iterative optimization steps, i.e. when a potential new optimal point should be
                        rejected and when it can be accepted. Exactly one of the acceptance criteria below
                        may be selected for this optimizer.""")
    specs.addSub(accept)
    ## common options to all acceptanceCondition descenders
    ## TODO
    ## get specs for each acceptanceCondition subclass, and add them to this class's options
    for option in acceptKnownTypes():
      subSpecs = acceptReturnClass(option, cls).getInputSpecification()
      accept.addSub(subSpecs)

    # convergence
    conv = InputData.parameterInputFactory('convergence', strictMode=True,
        printPriority=109,
        descr=r"""a node containing the desired convergence criteria for the optimization algorithm.
              Note that convergence is met when any one of the convergence criteria is met. If no convergence
              criteria are given, then the defaults are used.""")
    specs.addSub(conv)
    for name,descr in cls.convergenceOptions.items():
      conv.addSub(InputData.parameterInputFactory(name, contentType=InputTypes.FloatType,descr=descr ))

    # Presistance
    conv.addSub(InputData.parameterInputFactory('persistence', contentType=InputTypes.IntegerType,
        printPriority = 108,
        descr=r"""provides the number of consecutive times convergence should be reached before a trajectory
              is considered fully converged. This helps in preventing early false convergence."""))

    # Cooling Schedule
    coolingSchedule = InputData.parameterInputFactory('coolingSchedule',contentType=InputTypes.makeEnumType('coolingSchedule','',['linear','exponential','boltzmann','cauchy','fast','veryfast']),
        printPriority=109,
        descr=r""" The function governing the cooling process. Currently, user can select between, \xmlString{linear}, \xmlString{exponential}, \xmlString{cauchy}, \xmlString{boltzmann}, \xmlString{fast}, or \xmlString{veryfats}.
                  In case of \xmlString{linear} is provided, The cooling process will be governed by: $$ T^{k} = T^0 - k * \beta$$.
                  In case of \xmlString{exponential} is provided, The cooling process will be governed by: $$ T^{k} = T^0 * \alpha^k $$.
                  In case of \xmlString{boltzmann} is provided, The cooling process will be governed by: $$ T^{k} = \frac{T^0}{log10(k + 1.0)} $$.
                  In case of \xmlString{cauchy} is provided, The cooling process will be governed by: $$ T^{k} = \frac{T^0}{k + 1.0} $$.
                  In case of \xmlString{fast} is provided, The cooling process will be governed by: $$ T^{k} = T^{k} = T^0 * \exp(-k) $$.
                  In case of \xmlString{veryfast} is provided, The cooling process will be governed by: $$ T^{k} =  T^0 * \exp(-ck^{1/D})$$.
                  For more information about the selection of learning rates and other parameters, consult the Theory Manual""") # TODO: Update the theory manual for simulated annealing
    specs.addSub(coolingSchedule)
    return specs

  def __init__(self):
    RavenSampled.__init__(self)
    self._convergenceCriteria = defaultdict(giveZero) # names and values for convergence checks
    self._stepHistory = {}         # {'magnitude': size, 'versor': direction} for step
    self._acceptHistory = {}       # acceptability
    self._stepRecommendations = {} # by traj, if a 'cut' or 'grow' is recommended else None
    self._acceptRerun = {}         # by traj, if True then override accept for point rerun
    self._convergenceInfo = {}     # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = 0  # consecutive persistence required to mark convergence
    self._stepInstance = None      # instance of StepManipulator
    self._acceptInstance = None    # instance of AcceptanceCondition
    self.T = None                  # current temperature
    # np.random.seed(42) # TODO remove this

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
    self._convergenceCriteria['samePoint'] = 1e-16 #
    # Cooling Schedule
    coolingNode = paramInput.findFirst('coolingSchedule')
    if coolingNode is None:
      self._coolingMethod = 'exponential'
    else:
      self._coolingMethod = coolingNode.value

  def initialize(self, externalSeeding=None, solutionExport=None):
    """
      This function should be called every time a clean optimizer is needed. Called before takeAstep in <Step>
      @ In, externalSeeding, int, optional, external seed
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    RavenSampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    self.info = {}
    for var in self.toBeSampled:
      self.info['amp_'+var] = None
      self.info['delta_'+var] = None
    self._acceptInstance.initialize()
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
    self._stepHistory[traj] = deque(maxlen=self._maxHistLen)
    self._acceptHistory[traj] = deque(maxlen=self._maxHistLen)
    self._stepRecommendations[traj] = None
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
    self.raiseADebug('Adding run to queue: {} | {}'.format(point, info))
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
    self._resolveNewOptPoint(traj, rlz, rlz[self._objectiveVar], info)
    if self._stepTracker[traj]['opt'] == None:
      # revert to the last accepted point
      rlz = self._optPointHistory[traj][-1][0]
      info = self._optPointHistory[traj][-1][1]
      self.incrementIteration(traj)
      info['step'] = self.getIteration(traj)
      optVal = rlz[self._objectiveVar]
    iter = self.getIteration(traj) +1
    fraction = iter/self.limit
    currentPoint = self._collectOptPoint(rlz)
    self.T0 = self._temperature(fraction)
    self.T = self._coolingSchedule(iter,self.T0, self._coolingMethod, alpha = 0.94, beta = 0.1,d=1.0)
    if traj in self._activeTraj:
      newPoint = self._nextNeighbour(rlz,fraction)
      # check new opt point against constraints
      try:
        suggested, modded = self._handleExplicitConstraints(newPoint, currentPoint, 'opt')
      except NoConstraintResolutionFound:
        # we've tried everything, but we just can't hack it
        self.raiseAMessage('Optimizer "{}" trajectory {} was unable to continue due to functional or boundary constraints.'
                          .format(self.name, traj))
        self._closeTrajectory(traj, 'converge', 'no constraint resolution', newPoint[self._objectiveVar])
        return
      self._submitRun(suggested, traj, self.getIteration(traj))

  # * * * * * * * * * * * * * * * *
  # Convergence Checks
  # Note these names need to be formatted according to checkConvergence check!
  convFormat = ' ... {name:^12s}: {conv:5s}, {got:1.2e} / {req:1.2e}'

  # NOTE checkConvSamePoint has a different call than the others
  # should this become an informational dict that can be passed to any of them?

  def checkConvergence(self, traj, new, old):
    """
      Check for trajectory convergence
      @ In, traj, int, trajectory to consider
      @ Out, None? FIXME
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
      f = getattr(self, '_checkConv{}'.format(fName))
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
    same = list(new[var] == old[var] for var in self.toBeSampled)
    converged = all(same)
    self.raiseADebug(self.convFormat.format(name='same point',
                                            conv=str(converged),
                                            got=sum(same),
                                            req=len(same)))
    return converged

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    pass

  def _checkAcceptability(self, traj, opt):
    """
      Check if new opt point is acceptably better than the old one
      @ In, traj, int, identifier
      @ In, opt, dict, new opt point
      @ In, optVal, float, new optimization value
      @ Out, acceptable, str, acceptability condition for point
      @ Out, old, dict, old opt point
    """
    # Check acceptability
    # NOTE: if self._optPointHistory[traj]: -> faster to use "try" for all but the first time
    try:
      old, _ = self._optPointHistory[traj][-1]
      oldVal = old[self._objectiveVar]
      # check if same point
      self.raiseADebug(' ... change: {d: 1.3e} new objective: {n: 1.6e} old objective: {o: 1.6e}'
                      .format(d=opt[self._objectiveVar]-oldVal, o=oldVal, n=opt[self._objectiveVar]))
      # if this is an opt point rerun, accept it without checking.
      if self._acceptRerun[traj]:
        acceptable = 'rerun'
        self._acceptRerun[traj] = False
        #self._stepRecommendations[traj] = 'shrink' # FIXME how much do we really want this?
      elif all(opt[var] == old[var] for var in self.toBeSampled):
        # this is the classic "same point" trap; we accept the same point, and check convergence later
        acceptable = 'accepted'
      else:
        if self._acceptabilityCriterion(oldVal,opt[self._objectiveVar])>randomUtils.random(dim=1, samples=1): # TODO replace it back
          acceptable = 'accepted'
          # self._stepCounter[traj] +=1
        else:
          #acceptable = self._checkForImprovement(opt[self._objectiveVar], oldVal) DO I NEED THIS HERE?!
          acceptable = 'rejected'
    except IndexError:
      # if first sample, simply assume it's better!
      acceptable = 'first'
      old = None
    self._acceptHistory[traj].append(acceptable)
    self.raiseADebug(' ... {a}!'.format(a=acceptable))
    return acceptable, old

  def _acceptabilityCriterion(self,currentObjective,newObjective):
    """
      Check if new opt point is acceptably better than the old one
      @ In, currentObjective, float, the current value of the objective function (i.e., current energy)
      @ In, newObjective, float, the value of the objective function at the new candidate
      @ Out, Prob, float, the acceptance probability
    """
    # Boltzman Constant
    kB = 1 #1.380657799e-23 # or 1
    if self.T0 == None:
      self.T0 = 1e4
    if self.T == None:
      self.T = self.T0

    if newObjective <= currentObjective:
      prob = 1
    else:
      deltaE = newObjective - currentObjective
      # prob = 1/(1+np.exp(deltaE/(kB * self.T)))
      prob = np.exp(-deltaE/(kB * self.T))
    return prob

  def _updateConvergence(self, traj, new, old, acceptable):
    """
      Updates convergence information for trajectory
      @ In, traj, int, identifier
      @ In, acceptable, str, condition of point
      @ Out, converged, bool, True if converged on ANY criteria
    """
    ## NOTE we have multiple "if acceptable" trees here, as we need to update soln export regardless
    if acceptable == 'accepted':
      self.raiseADebug('Convergence Check for Trajectory {}:'.format(traj))
      # check convergence
      converged, convDict = self.checkConvergence(traj, new, old)
    else:
      converged = False
      convDict = dict((var, False) for var in self._convergenceInfo[traj])
    self._convergenceInfo[traj].update(convDict)
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
    converged = self.T <= self._convergenceCriteria['temperature']
    self.raiseADebug(self.convFormat.format(name='temperature',
                                            conv=str(converged),
                                            got=self.T,
                                            req=self._convergenceCriteria['temperature']))
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
    solution = {'iteration': self.getIteration(traj),
                'trajID': traj,
                'Temp': self.T,
                'accepted': acceptable,
                'fraction': self.getIteration(traj)/self.limit
                }
    for key, val in self._convergenceInfo[traj].items():
      solution['conv_{}'.format(key)] = val
    # variables, objective function, constants, etc
    solution[self._objectiveVar] = rlz[self._objectiveVar]
    for var in self.toBeSampled:
      solution[var] = denormed[var]
      solution['amp_'+var] = self.info['amp_'+var]
      solution['delta_'+var] = self.info['delta_'+var]
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
      @ In, old, dict, previous optimal point (to resubmit)
    """
    self._cancelAssociatedJobs(info['traj'], step=info['step'])
    ## what do do if a point is rejected?
    # TODO user option to EITHER rerun opt point OR cut step!
    # initialize a new step
    self._initializeStep(traj)
    #self._acceptRerun[traj] = True #TODO: What if the model is inherintly stochastic ?!
  # END resolving potential opt points
  # * * * * * * * * * * * * * * * *

  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    # ## TODO: right now I do not handle functional Constraints
    modded = False
    return suggested, modded

  ###########
  # Utility Methods #
  ###########
  def _temperature(self, fraction):
    return max(0.01,min(1,1-fraction))

  def _coolingSchedule(self, iter, T0, type='exponential', alpha = 0.94, beta = 0.1,d=1.0):
    """TODO write the math here"""
    if T0 == None:
      T0 = 1e4
    if type == 'linear':
      return T0 - iter * beta
    elif type == 'exponential':
      return alpha ** iter * T0
    elif type == 'boltzmann':
      return T0/(np.log10(iter + d))
    elif type == 'fast':
      return np.exp(-iter) * T0
    elif type == 'cauchy':
      return T0/(iter + d)
    else:
      raise NotImplementedError('cooling schedule type not implemented.')

  def _nextNeighbour(self, rlz,fraction=1,alpha = 0.94):
    """ Perturb x to find the next random neighbour
        for linear  and exponential cooling:
        .. math::

            fraction = \frac{iter}{Limit}

            amp = 1-fraction

            delta = \frac{-amp}{2} + amp * r
        where :math: `r \sim \mathcal{U}(0,1)`

        for boltzmann cooling:
        .. math::

            amp = min(np.sqrt(T), \frac{1}{3*alpha}

            delta = r * alpha * amp

        where :math: `r \sim \mathcal{N}(0,1)`

        for fast cooling:
        .. math::

            amp = r

            delta = sign(amp-0.5)*T*((1+\frac{1.0}{T})^{\abs{2*amp-1}-1.0)

        where :math: `r \sim \mathcal{U}(0,1)`

        for cauchy cooling:
        .. math::

            amp = r

            delta = alpha * T * tan(amp)

        where :math: `r \sim \mathcal{U}(-\pi,\pi)`
    """

    nextNeighbour = {}
    if self._coolingMethod in ['linear' , 'exponential']:
      amp = ((fraction)**-1) / 20
      r = randomUtils.random(dim=len(self.toBeSampled.keys()), samples=1)
      delta = (-amp/2.)+ amp * r
    elif self._coolingMethod == 'boltzmann':
      amp = min(np.sqrt(self.T), 1/3.0/alpha)
      delta =  randomUtils.randomNormal(dim=len(self.toBeSampled.keys()), samples=1)*alpha*amp
    elif self._coolingMethod == 'fast':
      amp = randomUtils.random(dim=1, samples=1)
      T = self.T
      delta = np.sign(amp-0.5)*T*((1+1.0/T)**abs(2*amp-1)-1.0)
    elif self._coolingMethod == 'cauchy':
      amp = (np.pi - (-np.pi))*randomUtils.random(dim=len(self.toBeSampled.keys()), samples=1)-np.pi #(dim=1, samples=1)
      delta = alpha*self.T*np.tan(amp)
    for i,var in enumerate(self.toBeSampled.keys()):
      nextNeighbour[var] = rlz[var] + delta[i]
      self.info['amp_'+var] = amp
      self.info['delta_'+var] = delta[i]
    self.info['fraction'] = fraction
    return nextNeighbour

  ##############
  # Destructor #
  ##############
  def __del__(self):
    print('simulatedAnnealing() has been destroyed')
