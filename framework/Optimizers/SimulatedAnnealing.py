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

  Created 2020-02
  @author: Mohammad Abdo
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
#from BaseClasses import BaseType
#from Assembler import Assembler
from .Sampled import Sampled
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

class SimulatedAnnealing(Sampled):
  """
  This class performs simulated annealing optimization
  """
  convergenceOptions = ['objective']   # relative change in objective value
  coolingOptions = ['Linear','Exponential','Logarthmic']
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
    # initialization: add sampling-based options
    whenSolnExpEnum = InputTypes.makeEnumType('whenWriteEnum', 'whenWriteType', ['final', 'every'])
    init = specs.getSub('samplerInit')
    #specs.addSub(init)
    limit = InputData.parameterInputFactory('limit', contentType=InputTypes.IntegerType)
    write = InputData.parameterInputFactory('writeSteps', contentType=whenSolnExpEnum)
    init.addSub(limit)
    init.addSub(write)
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
  
    # Cooling Schedule
    coolingSchedule = InputData.parameterInputFactory('coolingSchedule',contentType=InputTypes.StringType)
    specs.addSub(coolingSchedule)
    return specs
  
  def __init__(self):
    Sampled.__init__(self)
    # self._currentPoint = None #currentPoint
    # self._objectiveFunction = None #objectiveFunction
    # self._acceptanceCriterion = None #acceptanceCriterion
    # self.limit = None #maxIter
    # self._lb = None #lb
    # self._ub = None #ub
    
    self._convergenceCriteria = defaultdict(giveZero) # names and values for convergence checks
    self._stepHistory = {}         # {'magnitude': size, 'versor': direction} for step
    self._acceptHistory = {}       # acceptability
    self._stepRecommendations = {} # by traj, if a 'cut' or 'grow' is recommended else None
    self._acceptRerun = {}         # by traj, if True then override accept for point rerun
  
    self._convergenceInfo = {}     # by traj, the persistence and convergence information for most recent opt
    self._requiredPersistence = 0  # consecutive persistence required to mark convergence
    self._stepInstance = None      # instance of StepManipulator
    self._acceptInstance = None    # instance of AcceptanceCondition
    self._stepCounter = {}         #
    self._prevPoint = {}
    self._states = {}
    self.costs = [] 
    self.T0 = None
    self.T = None

  def __str__(self):
    return  "Simulated Annealing CLass:\nInitial Guess: " + self._currentPoint + "\nObjective Function: " + self._objectiveFunction + "\nAcceptence Criterion: " + self._acceptanceCriterion + "\nTemperature: " + self._temperature + "\nCooling Schedule: " + self._collingSchedule + "\n Max. Number of iterations: " + self.limit    

  def __repr__(self):
    # This is for debugging puposes for developers not users
    # if o is the class instance then:
    # o == eval(__repr__(o)) should be true
    return "simulatedAnnealing('" + str(self._initialGuess) + "', " +  self._objectiveFunction + "', " + self._acceptanceCriterion + "', " + self._temperatue + "', " + self._coolingSchedule + "', " + self.limit +")"
  
  @property
  def currentPoint(self):
    return self._currentPoint

  @currentPoint.setter
  def currentPoint(self, x0):
    self._currentPoint = x0
    
  @property
  def lb(self):
    return self._lb

  @lb.setter
  def lb(self, lb):
    self._lb = lb
    
  @property
  def ub(self):
    return self._ub

  @ub.setter
  def ub(self, ub):
    self._ub = ub      

  def handleInput(self, paramInput):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    Sampled.handleInput(self, paramInput)
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
      self._convergenceCriteria['gradient'] = 1e-6
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
    Sampled.initialize(self, externalSeeding=externalSeeding, solutionExport=solutionExport)
    #self._stepInstance.initialize(self.toBeSampled)
    self._acceptInstance.initialize()
    self._lb,self._ub =[],[]
    for var in self.toBeSampled:
      if var in self._variableBounds.keys():
        self._lb.append(self._variableBounds[var][0])
        self._ub.append(self._variableBounds[var][1])
    self._lb = np.array(self._lb)
    self._ub = np.array(self._ub)    
    # queue up the first run for each trajectory
    for traj, init in enumerate(self._initialValues):
      #self._stepHistory[traj].append({'magnitude': initialStepSize, 'versor': None}) # None is the direction, we don't know it yet
      self._submitRun(init,traj,self._stepCounter[traj])    

  def initializeTrajectory(self, traj=None):
    """
      Handles the generation of a trajectory.
      @ In, traj, int, optional, label to use
      @ Out, traj, int, new trajectory number
    """
    traj = Sampled.initializeTrajectory(self)
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
    #for key, inf in info.items():
    #  self.raiseADebug(' ... {}: {}'.format(key, inf))
    #self.raiseADebug(' ... {}: {}'.format('point', point))
    self._submissionQueue.append((point, info))
  # END queuing Runs
  # * * * * * * * * * * * * * * * *
  
  ###############
  # Run Methods #
  ###############
  def _useRealization(self, info, rlz, optVal):
    """
      Used to feedback the collected runs into actionable items within the sampler.
      @ In, info, dict, identifying information about the realization
      @ In, rlz, dict, realized realization
      @ In, optVal, float, value of objective variable (corrected for min/max)
      @ Out, None
    """
    currentPoint = {}
    optPoint = {}
    traj = info['traj']
    self._currentInfo = {}
    self._stepCounter[traj] += 1
    info['optVal'] = optVal
    # p1 = 0.7
    # # Probability of accepting worse solution at the end
    # pn = 0.001
    # # Initial temperature
    # t1 = -1.0/math.log(p1)
    # # Final temperature
    # tn = -1.0/math.log(pn)
    # # Fractional reduction every cycle
    # fraction = (tn/t1)**(1.0/(self.limit-1.0))
    fraction = self._stepCounter[traj]/self.limit
    T = self._coolingSchedule(self._stepCounter[traj], self.T0, type='exponential', alpha = 0.94, beta = 0.1,d=10)
    self.T0 = T
    self._currentInfo['T'] = T
    self._currentInfo['fraction'] = fraction
    self.currentPoint = self._collectOptPoint(rlz)  
    self.currentObjective = self._collectOptValue(rlz)
    self._resolveNewOptPoint(traj, rlz, self.currentObjective, info)
    # check new opt point against constraints
    try:
      suggested, modded = self._handleExplicitConstraints(newOpt, opt, 'opt')
    except NoConstraintResolutionFound:
      # we've tried everything, but we just can't hack it
      self.raiseAMessage('Optimizer "{}" trajectory {} was unable to continue due to functional or boundary constraints.'
                         .format(self.name, traj))
      self._closeTrajectory(traj, 'converge', 'no constraint resolution', opt[self._objectiveVar])
      return
    # update values if modified by constraint handling
    deltas = dict((var, suggested[var] - opt[var]) for var in self.toBeSampled)
    actualStepSize, stepVersor, _ = mathUtils.calculateMagnitudeAndVersor(np.array(list(deltas.values())))    
    
    if self._stepCounter[traj] == 1:
      self._prevRlz = rlz
      for var in rlz.keys():
        if var != self._objectiveVar:
          currentPoint[var] = rlz[var]  
      #self._states.update(currentPoint)
      self.currentObjective = self._collectOptValue(rlz)
      self.costs.append(self.currentObjective)
      newPoint = self._nextNeighbour(rlz,fraction)
      self._resolveNewOptPoint(traj, rlz, optVal, info)
      
      # check new opt point against constraints
      try:
        suggested, modded = self._handleExplicitConstraints(newPoint, newPoint, 'opt')
      except NoConstraintResolutionFound:
        # we've tried everything, but we just can't hack it
        self.raiseAMessage('Optimizer "{}" trajectory {} was unable to continue due to functional or boundary constraints.'
                           .format(self.name, traj))
        self._closeTrajectory(traj, 'converge', 'no constraint resolution', opt[self._objectiveVar])
        return
      self._submitRun(newPoint, traj, self._stepCounter[traj])
      return
    # Check Acceptance:
    #T0 = self._temperature(fraction)
    #T0 = 1e4
    oldObjective = self._collectOptValue(self._prevRlz)
    newObjective = self._collectOptValue(rlz)
    if self.Acceptability(traj,oldObjective, newObjective, T, 1)> randomUtils.random(dim=1, samples=1):
      # Accepted
      acceptable = 'accepted'
      self.raiseADebug('Accepting step {}, temp {}'.format(self._stepCounter[traj],T)) 
      for var in rlz.keys():
        if var != self._objectiveVar:
          optPoint[var] = rlz[var]
      self._prevRlz = rlz
      if T <= 1e-12:
      #if abs(oldObjective - newObjective)/newObjective < 1e-6:
      #if abs(oldObjective - newObjective) < 1e-6:
        opt = optPoint
        #print('Optimal Point is ',opt)
        converged = True
        return
      #Told = T
      #self._states.update(newPoint)
      self.costs.append(newObjective)
    else:
      # Rejected
      acceptable = 'rejected'
      self.raiseADebug('Rejecting step {}'.format(self._stepCounter[traj]))
    self._stepTracker[traj]['opt'] = (rlz, info)
    converged = self._updateConvergence(traj, rlz, self._prevRlz, acceptable) 
    # Find Next State
    newPoint = self._nextNeighbour(self._prevRlz,fraction)
    try:
      suggested, modded = self._handleExplicitConstraints(newPoint, currentPoint, 'opt')
    except NoConstraintResolutionFound:
      # we've tried everything, but we just can't hack it
      self.raiseAMessage('Optimizer "{}" trajectory {} was unable to continue due to functional or boundary constraints.'
                         .format(self.name, traj))
      self._closeTrajectory(traj, 'converge', 'no constraint resolution', opt[self._objectiveVar])
      return    
    #self._updateSolutionExport(traj, rlz, T,fraction,self._stepCounter[traj])
    self._submitRun(newPoint, traj, self._stepCounter[traj])

  def checkConvergence(self, traj,new,old):
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
    # self.raiseADebug(self.convFormat.format(name='same point',
    #                                         conv=str(converged),
    #                                         got=sum(same),
    #                                         req=len(same)))
    return converged

  def _checkForImprovement(self, new, old):
    """
      Determine if the new value is sufficient improved over the old.
      @ In, new, float, new optimization value
      @ In, old, float, previous optimization value
      @ Out, improved, bool, True if "sufficiently" improved or False if not.
    """
    pass

  def _checkAcceptability(self, traj, opt, optVal):
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
      oldVal = self._collectOptValue(old)
      # check if same point
      self.raiseADebug(' ... change: {d: 1.3e} new: {n: 1.6e} old: {o: 1.6e}'
                      .format(d=optVal-oldVal, o=oldVal, n=optVal))
      # if this is an opt point rerun, accept it without checking.
      if self._acceptRerun[traj]:
        acceptable = 'rerun'
        self._acceptRerun[traj] = False
        self._stepRecommendations[traj] = 'shrink' # FIXME how much do we really want this?
      elif all(opt[var] == old[var] for var in self.toBeSampled):
        # this is the classic "same point" trap; we accept the same point, and check convergence later
        acceptable = 'accepted'
      else:
        if self._acceptabilityCriterion(oldVal,optVal)>randomUtils.random(dim=1, samples=1):
          acceptable = 'accepted'
        else:
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
    kB = 1.380657799e-23
    if self.T == None:
      self.T = self.T0
    if optVal <= self.currentObjective:
      prob = 1
    else:
      deltaE = optVal - self.currentObjective
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
    delta = mathUtils.relativeDiff(self._collectOptValue(o2), self._collectOptValue(o1))
    converged = abs(delta) < self._convergenceCriteria['objective']
    self.raiseADebug(self.convFormat.format(name='objective',
                                            conv=str(converged),
                                            got=delta,
                                            req=self._convergenceCriteria['objective']))
    return converged
  
  def _updatePersistence(self, traj, converged, optVal):
    """
      Update persistence tracking state variables
      @ In, traj, identifier
      @ In, converged, bool, convergence check result
      @ In, optVal, float, new optimal value
      @ Out, None
    """
    pass

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
      @ In, old, dict, previous optimal point (to resubmit)
    """
    pass
  
  def _applyFunctionalConstraints(self, suggested, previous):
    """
      fixes functional constraints of variables in "point" -> DENORMED point expected!
      @ In, suggested, dict, potential point to apply constraints to
      @ In, previous, dict, previous opt point in consideration
      @ Out, point, dict, adjusted variables
      @ Out, modded, bool, whether point was modified or not
    """
    # right now I do not handle functional Constraints 
    point = suggested
    modded = False
    return point, modded


  ##################
  # Static Methods #
  ##################
  @staticmethod
  def funcname(parameter_list):
    pass

  #################
  # Class Methods #
  #################
  @classmethod
  def funcname(parameter_list):
    pass

  ###########
  # Utility Methods #
  ###########
  def _checkStepReady(self, traj):
    """
      Checks if enough information has been collected to proceed with optimization
      @ In, traj, int, identifier for trajectory of interest
      @ Out, ready, bool, True if all required data has been collected
    """
    # need to make sure opt point, grad points are all present
    # tracker = self._stepTracker[traj]
    # if tracker['opt'] is None:
    #   return False
    return True

  # def _resolveNewOptPoint(self,traj,rlz,optVal,info):  
  #   pass
  
  # def _objectiveFunction(self,x,model):
  #   """
  #   Method to compute the black-box objective function
  #   In the literature of simulated annealing
  #   the cose function (objective function) is often called
  #   System energy E(x)
  #   @ In, x, vector, design vector (configuaration in SA literature)
  #   @ Out, E(x), scalar, energy at x 
  #   """   
  #   #return self.E(x)
  #   return model(x)

  def _acceptanceCriterion(self, oldObjective, newObjective, T, kB=1.380657799e-23):
    if newObjective < oldObjective:
      return 1
    else:
      deltaE = (newObjective - oldObjective) # Should it be absolute?
      prob = np.exp(-deltaE/(kB*T))
      return prob  

  def _temperature(self, fraction):
    return max(0.01,min(1,1-fraction))

  def _coolingSchedule(self, iter, T0, type='exponential', alpha = 0.94, beta = 0.1,d=10):
    if T0 == None:
      T0 = 1e4
    if type == 'linear':
      return T0 - iter * beta
    elif type == 'exponential':
      return alpha ** iter * T0
    elif type == 'logarithmic':
      return T0/(np.log10(iter + d))
    # elif type == 'exponential':
    #   return  (T1/T0) ** iter * T0 # Add T1  
    else:
      raise NotImplementedError('Type not implemented.')
  

  def _nextNeighbour(self, rlz, fraction=1):
    """ Perturb x to find the next random neighbour"""
    nextNeighbour = {}
    #rlz2 = {}
    for var in rlz.keys():
      if var != self._objectiveVar:
        amp = (fraction) / 10.
        #print(self._variableBounds[var])
        delta = (-amp/2.)+ amp * randomUtils.random(dim=1, samples=1)
        #delta = randomUtils.random(dim=1, samples=1) -0.5
        nextNeighbour[var] = rlz[var] + delta
        # Clip it to the normalized bound ~ U[0,1]
        #nextNeighbour[var] = max(min(nextNeighbour[var],1.0),0.0)
        self._currentInfo['delta_'+var] = delta
        #self._currentInfo['amp_'+var] = amp
    #self._updateSolutionExport(self._trajCounter,rlz2,True)
    return nextNeighbour
    #amp = (self.ub - self.lb) * fraction / 10
    #delta = (-amp/2)+ amp * np.random.randn(len(x))
    
    #return x + delta

  # def _anneal(self,rlz):
  #   state = []
  #   traj = self._traj
  #   for var in rlz.keys():
  #     if var in self.toBeSampled:
  #       state.append(rlz[var])
  #   #state = np.array(state)
  #   cost = self._collectOptValue(rlz)
  #   states, costs = [state], [cost]
  #   for step in range(1,self.limit):
  #     if cost <= self._convergenceCriteria['objective']:
  #       continue
  #     fraction = step / float(self.limit)
  #     T0 = self._temperature(fraction)
  #     T = self._coolingSchedule(step, T0, type='exponential', alpha = 0.94, beta = 0.1,d=10)
  #     # start new step
  #     newState = self._nextNeighbour(state,self._lb,self._ub,fraction)
  #     self._stepCounter[traj] += 1
  #     self.raiseADebug('New State reached, Checking for acceptance ...')
  #     newPoint = {}
  #     for ind,var in enumerate(self.toBeSampled):
  #       newPoint[var] = newState[ind]
  #     # #opt, _ = self._stepTracker[traj]['opt']
  #     # #prefix = job.getMetadata()['prefix']
  #     # #info = self.getIdentifierFromPrefix(prefix, pop=True)
  #     # _, full = self._targetEvaluation.realization(matchDict={'prefix': prefix})
  #     # rlz = dict((var, full[var]) for var in (list(self.toBeSampled.keys()) + [self._objectiveVar] + list(self.dependentSample.keys())))
  #     # rlz = self.normalizeData(rlz)      
  #     # print('DEBUGG starting from opt point:', self.denormalizeData(newState))
  #     # # get new step information
  #     # self.raiseADebug(' ... found new proposed opt point ...')
  #     # print('DEBUGG ... ... proposed:', self.denormalizeData(newState))
  #     # self._initializeStep(self._traj)
  #     # self.raiseADebug('Taking step {} for traj {} ...'.format(self._stepCounter[self._traj], self._traj))
  #     # ## initial tests show it's not a big deal for small systems
  #     # self.raiseADebug(' ... current opt point:', newState)
  #     # # initialize step
  #     # # rlz = dict((var, full[var]) for var in (list(self.toBeSampled.keys()) + [self._objectiveVar] + list(self.dependentSample.keys())))
  #     # # optVal = self._collectOptValue(rlz)
  #     # # rlz = self.normalizeData(rlz)
  #     #self._submissionQueue.append((newPoint))
  #     self._submitRun(newPoint, self._traj, self._stepCounter[self._traj])      
    
  #     new_cost = self._collectOptValue()
  #     # if self.acceptanceCriterion(cost, new_cost, T) > np.random.rand():
  #     if self._checkAcceptability(state, new_cost) > np.random.rand():
  #       state, cost = newState, new_cost
  #       states.append(state)
  #       costs.append(cost)
  #       print("  ==> Accept it!")
  #     else:
  #       print("  ==> Reject it...")
  #     return state, self.objectiveFunction(state,model), states, costs  

  ##############
  # Destructor #      
  ##############
  def __del__(self):
    print('simulatedAnnealing() has been destroyed')

def E(x):
  """
  $$(\vec(x)-5)^{T}\vec(x)-5$$
  """
  return (x-5) @ (x-5)

def Q(x):
  x1 = x[0]
  x2 = x[1]
  obj = 0.2 + x1**2 + x2**2 - 0.1*math.cos(6.0*3.1415*x1) - 0.1*math.cos(6.0*3.1415*x2)
  return obj

def beale(x):
  return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]*x[1])**2 + (2.625 - x[0] + x[0]*x[1]*x[1]*x[1])**2

if __name__ == "__main__":
  S1 = simulatedAnnealing()
  d = 2
  S1.currentPoint = np.random.randn(d)#np.array([0.5,1.5])
  lb,ub = -10,10
  S1.lb,S1.ub = lb,ub
  model = E
  Tol = 1e-4
  state, obj, states, costs = S1.anneal(model,Tol)
  
  #state, obj, states, costs = simulatedAnnealing(np.array([0.5,1.5]),lb,ub,objectiveFunction = model(np.array([0.5,1.5]))).anneal(model,Tol)
  print(state,obj)
  hist = np.array(states).reshape(-1,d)
  
  # Create a contour plot
  plt.figure()
  # Specify contour lines
  #lines = range(0,int(max(costs)),5)
  # Plot contours
  # Start location
  x_start = hist[0,:]

  # Design variables at mesh points
  i1 = np.arange(lb, ub, 0.01)
  i2 = np.arange(lb, ub, 0.01)
  x1m, x2m = np.meshgrid(i1, i2)
  costm = np.zeros(x1m.shape)
  for i in range(x1m.shape[0]):
      for j in range(x1m.shape[1]):
          xm = np.array([x1m[i][j],x2m[i][j]])
          costm[i][j] = model(xm)
  CS = plt.contour(x1m, x2m, costm)#,lines)
  # Label contours
  plt.clabel(CS, inline=1, fontsize=10)
  # Add some text to the plot
  plt.title('Contour Plot for Objective Functions')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.plot(hist[:,0],hist[:,1],'m-x')
  plt.grid()
  plt.savefig('contourPlot.png')
  plt.show()

  fig = plt.figure()
  ax1 = fig.add_subplot(211)
  ax1.plot(costs,'r.-')
  ax1.legend(['Objective'])
  ax1.set_xlabel('# iterations')
  ax1.grid()
  ax2 = fig.add_subplot(212)
  ax2.plot(hist[:,0],'r.')
  ax2.plot(hist[:,1],'b-')
  ax2.legend(['x1','x2'])
  ax2.set_xlabel('# iterations')
  ax2.grid()
  plt.show()
  # Save the figure as a PNG
  plt.savefig('iterationHistory.png')
