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
  This module contains the Simultaneous Perturbation Stochastic Approximation Optimization strategy

  Created on June 16, 2016
  @ author: chenj, alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
import numpy as np
import scipy
from itertools import cycle
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .GradientBasedOptimizer import GradientBasedOptimizer
import Distributions
from utils import mathUtils,randomUtils,InputData
import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class SPSA(GradientBasedOptimizer):
  """
    Simultaneous Perturbation Stochastic Approximation Optimizer
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls
    """
    inputSpecification = super(SPSA,cls).getInputSpecification()

    # add additional parameters to "parameter"
    param = inputSpecification.popSub('parameter')
    param.addSub(InputData.parameterInputFactory('initialStepSize', contentType=InputData.FloatType, strictMode=True))
    param.addSub(InputData.parameterInputFactory('perturbationDistance', contentType=InputData.FloatType, strictMode=True))

    inputSpecification.addSub(param)
    return inputSpecification

  def __init__(self):
    """
      Default Constructor
    """
    GradientBasedOptimizer.__init__(self)
    self.paramDict['pertSingleGrad'] = 1
    self.currentDirection = None
    self.stochasticDistribution = None                        # Distribution used to generate perturbations
    self.stochasticEngine = None                              # Random number generator used to generate perturbations
    self.stochasticEngineForConstraintHandling = Distributions.returnInstance('Normal',self)
    self.stochasticEngineForConstraintHandling.mean, self.stochasticEngineForConstraintHandling.sigma = 0, 1
    self.stochasticEngineForConstraintHandling.upperBoundUsed, self.stochasticEngineForConstraintHandling.lowerBoundUsed = False, False
    self.stochasticEngineForConstraintHandling.initializeDistribution()

  def localInputAndChecks(self, xmlNode):
    """
      Local method for additional reading.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    GradientBasedOptimizer.localInputAndChecks(self, xmlNode)
    self.currentDirection   = None
    numValues = self._numberOfSamples()
    # set the initial step size
    ## use the hyperdiagonal of a unit hypercube with a side length equal to the user's provided initialStepSize * 1.0
    stepPercent = float(self.paramDict.get('initialStepSize', 0.05))
    self.paramDict['initialStepSize'] = mathUtils.hyperdiagonal(np.ones(numValues)*stepPercent)
    self.raiseADebug('Based on initial step size factor of "{:1.5e}", initial step size is "{:1.5e}"'
                         .format(stepPercent, self.paramDict['initialStepSize']))
    # set the perturbation distance
    ## if not given, default to 10% of the step size
    self.paramDict['pertDist'] = float(self.paramDict.get('perturbationDistance',0.01))
    self.raiseADebug('Perturbation distance is "{:1.5e}" percent of the step size'
                         .format(self.paramDict['pertDist']))

    self.constraintHandlingPara['innerBisectionThreshold'] = float(self.paramDict.get('innerBisectionThreshold', 1e-2))
    if not 0 < self.constraintHandlingPara['innerBisectionThreshold'] < 1:
      self.raiseAnError(IOError,'innerBisectionThreshold must be between 0 and 1; got',self.constraintHandlingPara['innerBisectionThreshold'])
    self.constraintHandlingPara['innerLoopLimit'] = float(self.paramDict.get('innerLoopLimit', 1000))

    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * (self.paramDict['pertSingleGrad']+1)

    # determine the number of indpendent variables (scalar and vectors included)
    stochDist = self.paramDict.get('stochasticDistribution', 'Hypersphere')
    if stochDist == 'Bernoulli':
      self.stochasticDistribution = Distributions.returnInstance('Bernoulli',self)
      self.stochasticDistribution.p = 0.5
      self.stochasticDistribution.initializeDistribution()
      # Initialize bernoulli distribution for random perturbation. Add artificial noise to avoid that specular loss functions get false positive convergence
      # FIXME there has to be a better way to get two random numbers
      self.stochasticEngine = lambda: [(0.5+randomUtils.random()*(1.+randomUtils.random()/1000.*randomUtils.randomIntegers(-1, 1, self))) if self.stochasticDistribution.rvs() == 1 else
                                   -1.*(0.5+randomUtils.random()*(1.+randomUtils.random()/1000.*randomUtils.randomIntegers(-1, 1, self))) for _ in range(numValues)]
    elif stochDist == 'Hypersphere':
      # TODO assure you can't get a "0" along any dimension! Need to be > 1e-15. Right now it's just highly unlikely.
      self.stochasticEngine = lambda: randomUtils.randPointsOnHypersphere(numValues) if numValues > 1 else [randomUtils.randPointsOnHypersphere(numValues)]
    else:
      self.raiseAnError(IOError, self.paramDict['stochasticEngine']+'is currently not supported for SPSA')

  def localLocalInitialize(self, solutionExport):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, a PointSet to hold the solution
      @ Out, None
    """
    self._endJobRunnable = (self._endJobRunnable*self.gradDict['pertNeeded'])+len(self.optTraj)
    # set up cycler for trajectories
    self.trajCycle = cycle(self.optTraj)
    # build up queue of initial runs
    for traj in self.optTraj:
      # for the first run, set the step size to the initial step size
      self.counter['lastStepSize'][traj] = self.paramDict['initialStepSize']
      # construct initial point for trajectory
      values = {}
      for var in self.getOptVars():
        # user-provided points
        values[var] = self.optVarsInit['initial'][var][traj]
        # assure points are within bounds; correct them if not
        values[var] = self._checkBoundariesAndModify(self.optVarsInit['upperBound'][var],
                                                     self.optVarsInit['lowerBound'][var],
                                                     self.optVarsInit['ranges'][var],
                                                     values[var], 0.99, 0.01)
      # normalize initial point for this trajectory
      data = self.normalizeData(values)
      # store (unnormalized?) point in history
      self.updateVariableHistory(values,traj)
      # set up a new batch of runs on the new optimal point (batch size 1 unless more requested by user)
      self.queueUpOptPointRuns(traj,data)
      # set up grad point near initial point
      pertPoints = self._createPerturbationPoints(traj,data)
      # set up storage structure for results
      self._setupNewStorage(traj)

  ###############
  # Run Methods #
  ###############
  def localStillReady(self, ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, variable indicating whether the caller is prepared for another input.
    """
    # accept unreadiness from another source if applicable
    if not ready:
      return ready
    if any(len(self.submissionQueue[t]) for t in self.optTraj):
      return True
    return False

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    GradientBasedOptimizer.localGenerateInput(self,model,oldInput)
    # find something to submit
    for _ in self.optTraj:
      # get next trajectory in line, which assures each gets fair treatment in submissions
      traj = next(self.trajCycle)
      # if this trajectory has a run to submit, populate the submission dictionaries
      if len(self.submissionQueue[traj]):
        prefix, point = self.getQueuedPoint(traj)
        for var in self.getOptVars():
          self.values[var] = point[var]
        self.inputInfo['prefix'] = prefix
        self.inputInfo['trajID'] = traj+1
        self.inputInfo['varsUpdate'] = self.counter['varsUpdate'][traj]
        # if we found a submission, cease looking for submissions
        return
    # if no submissions were found, then we shouldn't have flagged ourselves as Ready or there's a bigger issue!
    self.raiseAnError(RuntimeError,'Attempted to generate an input but there are none queued to provide!')

  ###################
  # Utility Methods #
  ###################
  def angleBetween(self, traj, d1, d2):
    """
      Evaluate the angle between the two dictionaries of vars (d1 and d2) by means of the dot product. Unit: degree
      @ In, traj, int, trajectory label for whom we are working
      @ In, d1, dict, first vector
      @ In, d2, dict, second vector
      @ Out, angleD, float, angle between d1 and d2 with unit of degree
    """
    nVar = len(self.getOptVars())
    v1, v2 = np.zeros(shape=[nVar,]), np.zeros(shape=[nVar,])
    for cnt, var in enumerate(self.getOptVars()):
      v1[cnt], v2[cnt] = copy.deepcopy(d1[var]), copy.deepcopy(d2[var])
    angle = np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
    if np.isnan(angle):
      if (v1 == v2).all():
        angle = 0.0
      else:
        angle = np.pi
    angleD = np.rad2deg(angle)
    return angleD

  def _bisectionForConstrainedInput(self,traj,varK,ak,vector):
    """
      Method to find the maximum fraction of the step size "ak" such that taking a step of size "ak" in
      the direction of "vector" starting from current input point "varK" will not violate the constraint function.
      @ In, traj, int, trajectory label for whom we are working
      @ In, varK, dictionary, current variable values
      @ In, ak, float or array, it is step size (gain) for variable update (if array, different gain for each variable)
      @ In, vector, dictionary, contains the gradient information as a unit vector
      @ Out, _bisectionForConstrainedInput, tuple(bool,dict), (indicating whether a fractional step is found, contains the fraction of step that satisfies constraint)
    """
    try:
      gain = ak[:]
    except (TypeError,IndexError):
      gain = [ak]*self._numberOfSamples() #technically incorrect, but missing ones will be *0 anyway just below here

    innerBisectionThreshold = self.constraintHandlingPara['innerBisectionThreshold']
    bounds = [0, 1.0]
    tempVarNew = {}
    frac = 0.5
    while np.absolute(bounds[1]-bounds[0]) >= innerBisectionThreshold:
      index = 0
      for var in self.getOptVars():
        numSamples = np.prod(self.variableShapes[var])
        new = np.zeros(numSamples)
        for i in range(numSamples):
          if numSamples > 1:
            new[i] = copy.copy(varK[var][i] - gain[index] * vector[var][i]*1.0*frac) # FIXME is this copy needed?
          else:
            new = copy.copy(varK[var]-gain[index]*vector[var]*1.0*frac) # FIXME is this copy needed?
          index += 1
        tempVarNew[var] = new

      satisfied,_ = self.checkConstraint(tempVarNew)
      if satisfied:
        bounds[0] = copy.deepcopy(frac)
        if np.absolute(bounds[1]-bounds[0]) >= innerBisectionThreshold:
          varKPlus = copy.deepcopy(tempVarNew)
          return True, varKPlus
        frac = copy.deepcopy(bounds[1]+bounds[0])/2.0
      else:
        bounds[1] = copy.deepcopy(frac)
        frac = copy.deepcopy(bounds[1]+bounds[0])/2.0
    return False, None

  def _checkBoundariesAndModify(self,upperBound,lowerBound,varRange,currentValue,pertUp,pertLow):
    """
      Method to check the boundaries and add a perturbation in case they are violated
      @ In, upperBound, float, the upper bound for the variable
      @ In, lowerBound, float, the lower bound for the variable
      @ In, varRange, float, the variable range
      @ In, currentValue, float, the current value
      @ In, pertUp, float, the perturbation to apply in case the upper bound is violated
      @ In, pertLow, float, the perturbation to apply in case the lower bound is violated
      @ Out, convertedValue, float, the modified value in case the boundaries are violated
    """
    # if variable is a vector of values, treat each independently
    if isinstance(currentValue,np.ndarray):
      convertedValue = np.asarray([self._checkBoundariesAndModify(upperBound,lowerBound,varRange,value,pertUp,pertLow) for value in currentValue])
    else:
      convertedValue = currentValue
      if currentValue > upperBound:
        convertedValue = pertUp*varRange + lowerBound
      elif currentValue < lowerBound:
        convertedValue = pertLow*varRange + lowerBound
    return convertedValue

  def clearCurrentOptimizationEffort(self,traj):
    """
      See base class.  Used to clear out current optimization information and start clean.
      For the SPSA, this means clearing out the perturbation points
      @ In, traj, int, index of trajectory being cleared
      @ Out, None
    """
    self.raiseADebug('Clearing current optimization efforts ...')
    self.counter ['perturbation'   ][traj] = 0
    self.counter ['gradientHistory'][traj] = [{},{}]
    self.counter ['gradNormHistory'][traj] = [0,0]
    #only clear non-opt points from pertPoints
    for i in self.perturbationIndices:
      self.gradDict['pertPoints'][traj][i] = 0
    self.convergeTraj[traj] = False
    self.status[traj] = {'process':'submitting grad eval points','reason':'found new opt point'}
    try:
      del self.counter['lastStepSize'][traj]
    except KeyError:
      pass
    try:
      del self.recommendToGain[traj]
    except KeyError:
      pass

  def _computeStepSize(self,paramDict,iterNum,traj):
    """
      Utility function to compute the step size
      @ In, paramDict, dict, dictionary containing information to compute gain parameter
      @ In, iterNum, int, current iteration index
      @ Out, new, float, current value for gain ak
    """
    #TODO FIXME is this a good idea?
    try:
      size = self.counter['lastStepSize'][traj]
    except KeyError:
      size = paramDict['initialStepSize']
    # modify step size based on the history of the gradients used
    frac = self.fractionalStepChangeFromGradHistory(traj)
    new = size*frac
    self.raiseADebug('step gain size for traj "{}" iternum "{}": {:1.3e} (root {:1.2e} frac {:1.2e})'.format(traj,iterNum,new,size,frac))
    self.counter['lastStepSize'][traj] = new
    return new

  def _computePerturbationDistance(self,traj,paramDict,iterNum):
    """
      Utility function to compute the perturbation distance (distance from opt point to grad point)
      @ In, traj, int, integer label for current trajectory
      @ In, paramDict, dict, dictionary containing information to compute gain parameter
      @ In, iterNum, int, current iteration index
      @ Out, distance, float, current value for gain ck
    """
    # perturbation point should be a percent of the intended step
    pct = paramDict['pertDist']
    distance = pct * self.counter['lastStepSize'][traj]
    return distance

  def _createPerturbationPoints(self, traj, optPoint, submit=True):
    """
      Creates perturbation points based on a provided NORMALIZED data point
      @ In, traj, int, integer label for current trajectory
      @ In, optPoint, dict, current optimal point near which to calculate gradient
      @ In, submit, bool, optional, if True then submit perturbation points to queue
      @ Out, points, list(dict), perturbation points
    """
    points = []
    distance = self._computePerturbationDistance(traj,self.paramDict,self.counter['varsUpdate'][traj]+1)
    for i in self.perturbationIndices:
      direction = self._getPerturbationDirection(i)
      point = {}
      index = 0
      for var in self.getOptVars():
        size = np.prod(self.variableShapes[var])
        if size > 1:
          new = np.zeros(size)
          for v, origVal in enumerate(optPoint[var]):
            new[v] = origVal + distance*direction[index]
            new[v] = self._checkBoundariesAndModify(1.0, 0.0, 1.0, new[v], 0.9999, 0.0001)
            index += 1
          point[var] = new
        else:
          val = optPoint[var] + distance*direction[index]
          val = self._checkBoundariesAndModify(1.0, 0.0, 1.0, val, 0.9999, 0.0001)
          index += 1
          point[var] = val
      points.append(point)
      if submit:
        prefix = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],i)
        self.submissionQueue[traj].append({'inputs':point, 'prefix':prefix})
    return points

  def _generateVarsUpdateConstrained(self,traj,ak,gradient,varK):
    """
      Method to generate input for model to run, considering also that the input satisfies the constraint
      @ In, traj, int, trajectory label for whom we are generating variables with constraint consideration
      @ In, ak, float or array, it is gain for variable update (if array, different gain for each variable)
      @ In, gradient, dictionary, contains the gradient information for variable update
      @ In, varK, dictionary, current variable values (normalized)
      @ Out, varKPlus, dictionary, variable values for next iteration.
      @ Out, modded, bool, if True the point was modified by the constraint
    """
    varKPlus = {}
    gain = [ak]*self._numberOfSamples() #technically too many entries, but unneeded ones will be *0 anyway just below here
    gain = np.asarray(gain)
    index = 0
    for var in self.getOptVars():
      numSamples = np.prod(self.variableShapes[var])
      if numSamples == 1:
        new = varK[var]-gain[index]*gradient.get(var,0.0)
        index += 1
      else:
        new = np.zeros(numSamples)
        for i in range(numSamples):
          new[i] = varK[var][i] - gain[index] * gradient.get(var,[0.0]*i)[i]
          index += 1
      varKPlus[var] = new
    satisfied, activeViolations = self.checkConstraint(self.denormalizeData(varKPlus))
    if satisfied:
      return varKPlus, False
    # else if not satisfied ...
    # check if the active constraints are the boundary ones. In this case, try to project the gradient at an angle
    modded = False
    if len(activeViolations['internal']) > 0:
      self.raiseADebug('Attempting to fix constraint violation with gradient projection ...')
      modded = True
      projectedOnBoundary= {}
      for var,under,over in activeViolations['internal']:
        if np.prod(self.variableShapes[var]) == 1:
          if np.sum(over) > 0:
            projectedOnBoundary[var] = self.optVarsInit['upperBound'][var]
          elif np.sum(under) > 0:
            projectedOnBoundary[var] = self.optVarsInit['lowerBound'][var]
          gradient[var] = 0.0
        else:
          projectedOnBoundary[var] = self.denormalizeData({var:varKPlus[var]})[var]
          projectedOnBoundary[var][under] = self.optVarsInit['lowerBound'][var]
          projectedOnBoundary[var][over] = self.optVarsInit['upperBound'][var]
          gradient[var][np.logical_or(under,over)] = 0.0
      varKPlus.update(self.normalizeData(projectedOnBoundary))
      newNormWithoutComponents = self.calculateMultivectorMagnitude(gradient.values())
      for var in gradient.keys():
        gradient[var] = gradient[var]/newNormWithoutComponents if newNormWithoutComponents != 0.0 else gradient[var]

    if len(activeViolations['external']) == 0:
      return varKPlus, modded

    self.raiseADebug('Attempting to fix constraint violation by shortening gradient vector ...')
    # Try to find varKPlus by shorten the gradient vector
    self.raiseADebug('Trajectory "{}" hit constraints ...'.format(traj))
    self.raiseADebug('  Attempting to shorten step length ...')
    foundVarsUpdate, varKPlus = self._bisectionForConstrainedInput(traj,varK, ak, gradient)
    if foundVarsUpdate:
      self.raiseADebug('   ... successfully found new point by shortening length.')
      return varKPlus, True

    self.raiseADebug('Attempting to fix constraint violation by rotating towards orthogonal ...')
    # Try to find varKPlus by rotate the gradient towards its orthogonal, since we consider the gradient as perpendicular
    # with respect to the constraints hyper-surface
    self.raiseADebug('  Attempting instead to rotate trajectory ...')
    innerLoopLimit = self.constraintHandlingPara['innerLoopLimit']
    if innerLoopLimit < 0:
      self.raiseAnError(IOError, 'Limit for internal loop for constraint handling should be nonnegative')
    loopCounter = 0
    foundPendVector = False
    # search for the perpendicular vector
    while not foundPendVector and loopCounter < innerLoopLimit:
      loopCounter += 1
      # randomly choose the index of a variable to be the dependent? pivot
      depVarPos = randomUtils.randomIntegers(0,len(self.getOptVars())-1,self)
      # if that variable is multidimensional, pick a dimension -> this is not precisely equal probability of picking, but that should be okay.
      varSize = np.prod(self.variableShapes[var])
      if varSize > 1:
        depVarIdx = randomUtils.randomIntegers(0,varSize-1,self)
      pendVector = {}
      npDot = 0
      for varID, var in enumerate(self.getOptVars()):
        varSize = np.prod(self.variableShapes[var])
        if varSize == 1:
          pendVector[var] = self.stochasticEngineForConstraintHandling.rvs() if varID != depVarPos else 0.0
          npDot += pendVector[var]*gradient[var]
        else:
          for i in range(varSize):
            pendVector[var][i] = self.stochasticEngineForConstraintHandling.rvs() if (varID != depVarPos and depVarIdx != i) else 0.0
          npDot += np.sum(pendVector[var]*gradient[var])
      # TODO does this need to be in a separate loop or can it go with above?
      for varID, var in enumerate(self.getOptVars()):
        if varID == depVarPos:
          varSize = np.prod(self.variableShapes[var])
          if varSize == 1:
            pendVector[var] = -npDot/gradient[var]
          else:
            pendVector[var][depVarIdx] = -npDot/gradient[var][depVarIdx]

      r  = self.calculateMultivectorMagnitude([  gradient[var] for var in self.getOptVars()])
      r /= self.calculateMultivectorMagnitude([pendVector[var] for var in self.getOptVars()])
      for var in self.getOptVars():
        pendVector[var] = copy.deepcopy(pendVector[var])*r

      varKPlus = {}
      index = 0
      for var in self.getOptVars():
        varSize = np.prod(self.variableShapes[var])
        new = np.zeros(varSize)
        for i in range(varSize):
          if varSize == 1:
            new = copy.copy(varK[var]-gain[index]*pendVector[var]*1.0)
          else:
            new[i] = copy.copy(varK[var][i]-gain[index]*pendVector[var][i]*1.0)
          index += 1
        varKPlus[var] = new

      foundPendVector, activeConstraints = self.checkConstraint(self.denormalizeData(varKPlus))
      if not foundPendVector:
        foundPendVector, varKPlus = self._bisectionForConstrainedInput(traj,varK, gain, pendVector)
      gain = gain/2.

    if foundPendVector:
      lenPendVector = 0
      for var in self.getOptVars():
        lenPendVector += np.sum(pendVector[var]**2)
      lenPendVector = np.sqrt(lenPendVector)

      rotateDegreeUpperLimit = 2
      while self.angleBetween(traj,gradient, pendVector) > rotateDegreeUpperLimit:
        sumVector = {}
        lenSumVector = 0
        for var in self.getOptVars():
          sumVector[var] = gradient[var] + pendVector[var]
          lenSumVector += np.sum(sumVector[var]**2)

        tempTempVarKPlus = {}
        index = 0
        for var in self.getOptVars():
          sumVector[var] = copy.deepcopy(sumVector[var]/np.sqrt(lenSumVector)*lenPendVector)
          varSize = np.prod(self.variableShapes[var])
          new = np.zeros(varSize)
          for i in range(varSize):
            if varSize == 1:
              new = copy.copy(varK[var]-gain[index]*sumVector[var]*1.0)
            else:
              new[i] = copy.copy(varK[var][i]-gain[index]*sumVector[var][i]*1.0)
            index += 1
          tempTempVarKPlus[var] = new
        satisfied, activeConstraints = self.checkConstraint(self.denormalizeData(tempTempVarKPlus))
        if satisfied:
          varKPlus = copy.deepcopy(tempTempVarKPlus)
          pendVector = copy.deepcopy(sumVector)
        else:
          gradient = copy.deepcopy(sumVector)
      self.raiseADebug('   ... successfully found new point by rotating trajectory.')
      return varKPlus, True
    varKPlus = varK
    self.raiseADebug('   ... did not successfully find new point.')
    return varKPlus, False

  def _getAlgorithmState(self,traj):
    """
      Returns values specific to this algorithm such that it could pick up again relatively easily from here.
      @ In, traj, int, the trajectory being saved
      @ Out, state, dict, keys:values this algorithm cares about saving for this trajectory
    """
    state = {}
    state['lastStepSize']    = copy.deepcopy(self.counter['lastStepSize'   ].get(traj,None))
    state['gradientHistory'] = copy.deepcopy(self.counter['gradientHistory'].get(traj,None))
    state['recommendToGain'] = copy.deepcopy(self.recommendToGain           .get(traj,None))
    return state

  def _getPerturbationDirection(self,perturbationIndex):
    """
      This method is aimed to get the perturbation direction (i.e. in this case the random perturbation versor)
      @ In, perturbationIndex, int, the perturbation index (stored in self.perturbationIndices)
      @ Out, direction, list, the versor for each optimization dimension
    """
    if perturbationIndex == self.perturbationIndices[0]:
      direction = self.stochasticEngine()
      self.currentDirection = direction
    else:
      # in order to perform the de-noising we keep the same perturbation direction and we repeat the evaluation multiple times
      direction = self.currentDirection
    return direction

  def localEvaluateGradient(self, traj):
    """
      Local method to evaluate gradient.
      @ In, traj, int, the trajectory id
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    # this method used to take a gradient estimation. Nothing actually used it, though. - PWT, 2018-10
    # denoising has already been performed, so get the results
    opt = self.realizations[traj]['denoised']['opt'][0]   # opt point is only one point
    pert = self.realizations[traj]['denoised']['grad'][0] # SPSA CURRENTLY only has one grad point
    gradient = {}
    # difference in objective variable
    lossDiff = mathUtils.diffWithInfinites(pert[self.objVar], opt[self.objVar])
    # we only need the +/- 1, we don't need the gradient value at all.
    lossDiff = 1.0 if lossDiff > 0.0 else -1.0
    # force gradient descent
    if self.optType == 'max':
      lossDiff *= -1.0
    # difference in input variables
    for var in self.getOptVars():
      dh = pert[var] - opt[var]
      # keep dimensionality consistent, so at least 1D
      gradient[var] = np.atleast_1d(lossDiff * dh)
    return gradient

  def _newOptPointAdd(self, gradient, traj):
    """
      This local method add a new opt point based on the gradient
      @ In, gradient, dict, dictionary containing the gradient
      @ In, traj, int, trajectory
      @ Out, varKPlus, dict, new point that has been queued (or None if no new points should be run for this traj)
    """
    stepSize = self._computeStepSize(self.paramDict, self.counter['varsUpdate'][traj], traj)
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
    varK = dict((var,self.counter['recentOptHist'][traj][0][var]) for var in self.getOptVars())
    varKPlus,modded = self._generateVarsUpdateConstrained(traj, stepSize, gradient, varK)
    #check for redundant paths
    if len(self.optTrajLive) > 1 and self.counter['solutionUpdate'][traj] > 0:
      removed = self._removeRedundantTraj(traj, varKPlus)
    else:
      removed = False
    if removed:
      return None
    #if the new point was modified by the constraint, reset the step size
    if modded:
      self.counter['lastStepSize'][traj] = self.paramDict['initialStepSize']
      self.raiseADebug('Resetting step size for trajectory',traj,'due to hitting constraints')
    self.queueUpOptPointRuns(traj,varKPlus)
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = varKPlus
    return varKPlus

  def _setAlgorithmState(self,traj,state):
    """
      @ In, traj, int, the trajectory being saved
      @ In, state, dict, keys:values this algorithm cares about saving for this trajectory
      @ Out, None
    """
    if state is None:
      return
    if state['lastStepSize'] is not None:
      self.counter['lastStepSize'][traj] = state['lastStepSize']
    if state['gradientHistory'] is not None:
      self.counter['gradientHistory'][traj] = state['gradientHistory']
    if state['recommendToGain'] is not None:
      self.recommendToGain[traj] = state['recommendToGain']
