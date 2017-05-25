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
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
import numpy as np
from numpy import linalg as LA
import scipy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .GradientBasedOptimizer import GradientBasedOptimizer
import Distributions
from utils import mathUtils
import SupervisedLearning
#Internal Modules End--------------------------------------------------------------------------------

class SPSA(GradientBasedOptimizer):
  """
    Simultaneous Perturbation Stochastic Approximation Optimizer
  """
  def __init__(self):
    """
      Default Constructor
    """
    GradientBasedOptimizer.__init__(self)
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
    self.paramDict['alpha'] = float(self.paramDict.get('alpha', 0.602))
    self.paramDict['gamma'] = float(self.paramDict.get('gamma', 0.101))
    self.paramDict['A']     = float(self.paramDict.get('A', self.limit['mdlEval']/10.))
    self.paramDict['a']     = self.paramDict.get('a', None)
    self.paramDict['c']     = float(self.paramDict.get('c', 0.005))
    #FIXME the optimization parameters should probably all operate ONLY on normalized data!
    #  -> perhaps the whole optimizer should only work on optimized data.

    #FIXME normalizing doesn't seem to have the desired effect, currently; it makes the step size very small (for large scales)
    #if "a" was defaulted, use the average scale of the input space.
    #This is the suggested value from the paper, missing a 1/gradient term since we don't know it yet.
    if self.paramDict['a'] is None:
      self.paramDict['a'] = mathUtils.hyperdiagonal(np.ones(len(self.optVars))) # the features are always normalized
      self.raiseAMessage('Defaulting "a" gradient parameter to',self.paramDict['a'])
    else:
      self.paramDict['a'] = float(self.paramDict['a'])

    self.constraintHandlingPara['innerBisectionThreshold'] = float(self.paramDict.get('innerBisectionThreshold', 1e-2))
    self.constraintHandlingPara['innerLoopLimit'] = float(self.paramDict.get('innerLoopLimit', 1000))

    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2

    if self.paramDict.get('stochasticDistribution', 'Bernoulli') == 'Bernoulli':
      self.stochasticDistribution = Distributions.returnInstance('Bernoulli',self)
      self.stochasticDistribution.p = 0.5
      self.stochasticDistribution.initializeDistribution()
      # Initialize bernoulli distribution for random perturbation. Add artificial noise to avoid that specular loss functions get false positive convergence
      # FIXME there has to be a better way to get two random numbers
      self.stochasticEngine = lambda: [(0.5+Distributions.random()*(1.+Distributions.random()/1000.*Distributions.randomIntegers(-1, 1, self))) if self.stochasticDistribution.rvs() == 1 else
                                   -1.*(0.5+Distributions.random()*(1.+Distributions.random()/1000.*Distributions.randomIntegers(-1, 1, self))) for _ in range(self.nVar)]
      #self.stochasticEngine = lambda: [1.0+(Distributions.random()/1000.0)*Distributions.randomIntegers(-1, 1, self) if self.stochasticDistribution.rvs() == 1 else
      #                                -1.0+(Distributions.random()/1000.0)*Distributions.randomIntegers(-1, 1, self) for _ in range(self.nVar)]
    else:
      self.raiseAnError(IOError, self.paramDict['stochasticEngine']+'is currently not supported for SPSA')

  def localLocalInitialize(self, solutionExport = None):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2
    self._endJobRunnable        = (self._endJobRunnable*self.gradDict['pertNeeded'])+len(self.optTraj)

  def localStillReady(self, ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, variable indicating whether the caller is prepared for another input.
    """
    self.nextActionNeeded = (None,None) #prevents carrying over from previous run
    #get readiness from parent
    ready = ready and GradientBasedOptimizer.localStillReady(self,ready,convergence)
    #if not ready, just return that
    if not ready:
      return ready
    for _ in range(len(self.optTrajLive)):
      # despite several attempts, this is the most elegant solution I've found to assure each
      #   trajectory gets even treatment.
      traj = self.optTrajLive.pop(0)
      self.optTrajLive.append(traj)
      #see if trajectory needs starting
      if self.counter['varsUpdate'][traj] not in self.optVarsHist[traj].keys():
        self.nextActionNeeded = ('start new trajectory',traj)
        break #return True
      # see if there are points needed for evaluating a gradient, pick one
      elif self.counter['perturbation'][traj] < self.gradDict['pertNeeded']:
        self.nextActionNeeded = ('add new grad evaluation point',traj)
        break #return True
      else:
        # since all evaluation points submitted, check if we have enough collected to evaluate a gradient
        evalNotFinish = False
        for pertID in range(1,self.gradDict['pertNeeded']+1):
          if not self._checkModelFinish(traj,self.counter['varsUpdate'][traj],pertID)[0]:#[0]:
            evalNotFinish = True
            break
        if not evalNotFinish:
          # enough evaluations are done to calculate this trajectory's gradient
          #evaluate the gradient TODO don't actually evaluate it, until we get Andrea's branch merged in
          self.nextActionNeeded = ('evaluate gradient',traj)
          break #return True
    # if we did not find an action, we're not ready to provide an input
    if self.nextActionNeeded[0] is None:
      self.raiseADebug('Not ready to provide a sample yet.')
      return False
    else:
      self.raiseADebug('Next action needed: "%s" on trajectory "%i"' %self.nextActionNeeded)
      return True

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
    convertedValue = currentValue
    if currentValue >= upperBound:
      convertedValue = pertUp*varRange + lowerBound
    if currentValue <= lowerBound:
      convertedValue = pertLow*varRange + lowerBound
    return convertedValue

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    GradientBasedOptimizer.localGenerateInput(self,model,oldInput)
    action, traj = self.nextActionNeeded
    #"action" and "traj" are set in localStillReady
    #"action" is a string of the next action needed by the optimizer in order to move forward
    #"traj" is the trajectory that is in need of the action

    if action == 'start new trajectory':
      self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
      for var in self.optVars:
        self.values[var] = self.optVarsInit['initial'][var][traj]
        #if exceeding bounds, bring value within 1% of range
        self.values[var] = self._checkBoundariesAndModify(self.optVarsInit['upperBound'][var],
                                                          self.optVarsInit['lowerBound'][var],
                                                          self.optVarsInit['ranges'][var],self.values[var],0.99,0.01)
      data = self.normalizeData(self.values)
      self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = copy.deepcopy(data)
      # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
      self.inputInfo['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],'v')

    elif action == 'add new grad evaluation point':
      self.counter['perturbation'][traj] += 1
      if self.counter['perturbation'][traj] == 1:
        # Generate all the perturbations at once, then we can submit them one at a time
        self.gradDict['pertPoints'][traj] = {}
        ck = self._computeGainSequenceCk(self.paramDict,self.counter['varsUpdate'][traj]+1)
        varK = copy.deepcopy(self.optVarsHist[traj][self.counter['varsUpdate'][traj]])

        if self.gradDict['numIterForAve'] > 1:
          # In order to converge on the average of the objective variable, one of the
          # perturbation is performed on the variable at iteration i
          # In this way, we can compute the average and denoise the signal
          samePointPerturbation = True
        else:
          samePointPerturbation = False
        for ind in range(self.gradDict['numIterForAve']):
          self.gradDict['pertPoints'][traj][ind] = {}
          delta = self.stochasticEngine()
          for varID, var in enumerate(self.optVars):
            if var not in self.gradDict['pertPoints'][traj][ind].keys():
              p1 = np.asarray([varK[var]+ck*delta[varID]*1.0]).reshape((1,))
              if samePointPerturbation:
                p2 = np.asarray(varK[var]).reshape((1,))
              else:
                p2 = np.asarray([varK[var]-ck*delta[varID]*1.0]).reshape((1,))
                #p2 = np.asarray([varK[var]-ck*delta[varID]*1.0]).reshape((1,))
              p1[0] = self._checkBoundariesAndModify(1.0, 0.0, 1.0,p1[0],0.9999,0.0001)
              p2[0] = self._checkBoundariesAndModify(1.0, 0.0, 1.0,p2[0],0.9998,0.0002)
              #sanity check: p1 != p2
              if p1 == p2:
                self.raiseAnError(RuntimeError,'In choosing gradient evaluation points, the same point was chosen twice for variable "%s"!' %var)
              self.gradDict['pertPoints'][traj][ind][var] = np.concatenate((p1, p2))


      # get one of the perturbations to run
      loc1 = self.counter['perturbation'][traj] % 2
      loc2 = np.floor(self.counter['perturbation'][traj] / 2) if loc1 == 1 else np.floor(self.counter['perturbation'][traj] / 2) - 1
      tempOptVars = {}
      for var in self.optVars:
        tempOptVars[var] = self.gradDict['pertPoints'][traj][loc2][var][loc1]
      tempOptVarsDenorm = copy.deepcopy(self.denormalizeData(tempOptVars))
      for var in self.optVars:
        self.values[var] = tempOptVarsDenorm[var]
      # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation)
      self.inputInfo['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],self.counter['perturbation'][traj])

    elif action == 'evaluate gradient':
      # evaluation completed for gradient evaluation
      self.counter['perturbation'][traj] = 0
      self.counter['varsUpdate'][traj] += 1
      gradient = self.evaluateGradient(self.gradDict['pertPoints'][traj], traj)
      ak = self._computeGainSequenceAk(self.paramDict,self.counter['varsUpdate'][traj],traj) # Compute the new ak
      self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
      varK = copy.deepcopy(self.optVarsHist[traj][self.counter['varsUpdate'][traj]-1])
      # FIXME here is where adjustments to the step size should happen
      #TODO this is part of a future request.  Commented for now.
      #get central response for this trajectory: how?? TODO FIXME
      #centralResponseIndex = self._checkModelFinish(traj,self.counter['varsUpdate'][traj]-1,'v')[1]
      #self.estimateStochasticity(gradient,self.gradDict['pertPoints'][traj][self.counter['varsUpdate'][traj]-1],varK,centralResponseIndex) #TODO need current point too!
      varKPlus,modded = self._generateVarsUpdateConstrained(ak,gradient,varK)
      #if the new point was modified by the constraint, reset the step size
      if modded:
        del self.counter['lastStepSize'][traj]
        self.raiseADebug('Resetting step size for trajectory',traj,'due to hitting constraints')
      varKPlusDenorm = self.denormalizeData(varKPlus)
      for var in self.optVars:
        self.values[var] = copy.deepcopy(varKPlusDenorm[var])
        self.optVarsHist[traj][self.counter['varsUpdate'][traj]][var] = copy.deepcopy(varKPlus[var])
      # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
      #again, this is a copied line of code, so we should extract it if possible
      self.inputInfo['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],'v')

      # remove redundant trajectory
      if len(self.optTrajLive) > 1 and self.counter['solutionUpdate'][traj] > 0:
        self._removeRedundantTraj(traj, self.optVarsHist[traj][self.counter['varsUpdate'][traj]])

    #unrecognized action
    else:
      self.raiseAnError(RuntimeError,'Unrecognized "action" in localGenerateInput:',action)

  def estimateStochasticity(self,gradient,perturbedPoints,centralPoint,centralResponseIndex):
    """
      Uses the gradient and central point to estimate the expected values of the reponse
      at each of the perturbed points.  The difference between actual and expected will
      give a low-order estimate of the standard deviation of the system noise (aka "c").
      @ In, gradient, dict, {var:value} for each input, the estimated gradient
      @ In, perturbedPoints, dict, {var:[values], var:[values], response:[values]}
      @ In, centralPoint, dict, the central point in the optimize search (but not the response!! we need the response!
      @ In, centralResponseIndex, int, index at which central evaluation can be found
      @ Out, c, float, estimate of standard deviation
    """
    centralResponse = self.mdlEvalHist.getRealization(centralResponseIndex)['outputs'][self.objVar]
    numPerturbed = len(perturbedPoints.values()[0])
    inVars = gradient.keys()
    origin = np.array(centralPoint.values())
    gradVal = gradient.values()
    #calculate the differences between gradient-based estimates and actual evaluations
    differences = []
    for n in range(numPerturbed):
      newPoint = np.array(list(perturbedPoints[var][n] for var in inVars))
      delta = newPoint - origin
      expectedResponse = sum(gradVal*delta) + centralResponse
      difference = centralResponse-expectedResponse
      differences.append(difference)
    c = mathUtils.hyperdiagonal(differences)

  def _updateParameters(self):
    """
      Uses information about the gradient and optimizer trajectory to update parameters
      Updated parameters include [a,A,alpha,c,gamma]
    """
    pass #future tool
    #update A <-- shouldn't be needed since A is an early-life stabilizing parameter
    #update alpha <-- if we're moving basically in the same direction, don't damp!
    #update a <-- increase or decrease step size based on gradient information
    #  determine the minimum desirable step size at early calcualtion (1/10 of average
    #update c <-- estimate stochasticity of the response; if low, "c" tends to 0
    #update gamma <-- distance between samples to determine gradient.  Scales with step size?

  def _generateVarsUpdateConstrained(self,ak,gradient,varK):
    """
      Method to generate input for model to run, considering also that the input satisfies the constraint
      @ In, ak, float or array, it is gain for variable update (if array, different gain for each variable)
      @ In, gradient, dictionary, contains the gradient information for variable update
      @ In, varK, dictionary, current variable values
      @ Out, tempVarKPlus, dictionary, variable values for next iteration.
      @ Out, modded, bool, if True the point was modified by the constraint
    """
    tempVarKPlus = {}
    try:
      gain = ak[:]
    except (TypeError,IndexError):
      gain = [ak]*len(self.optVars)
    gain = np.asarray(gain)
    for index,var in enumerate(self.optVars):
      tempVarKPlus[var] = copy.copy(varK[var]-gain[index]*gradient[var]*1.0)
    satisfied, activeConstraints = self.checkConstraint(tempVarKPlus)
    #satisfied, activeConstraints = self.checkConstraint(self.denormalizeData(tempVarKPlus))
    if satisfied:
      return tempVarKPlus, False
    # else if not satisfied ...
    # check if the active constraints are the boundary ones. In case, project the gradient
    modded = False
    if len(activeConstraints['internal']) > 0:
      projectedOnBoundary= {}
      for activeConstraint in activeConstraints['internal']:
        projectedOnBoundary[activeConstraint[0]] = activeConstraint[1]
      tempVarKPlus.update(self.normalizeData(projectedOnBoundary))
      modded = True
    if len(activeConstraints['external']) == 0:
      return tempVarKPlus, modded

    # Try to find varKPlus by shorten the gradient vector
    foundVarsUpdate, tempVarKPlus = self._bisectionForConstrainedInput(varK, ak, gradient)
    if foundVarsUpdate:
      return tempVarKPlus, True

    # Try to find varKPlus by rotate the gradient towards its orthogonal, since we consider the gradient as perpendicular
    # with respect to the constraints hyper-surface
    innerLoopLimit = self.constraintHandlingPara['innerLoopLimit']
    if innerLoopLimit < 0:
      self.raiseAnError(IOError, 'Limit for internal loop for constraint handling shall be nonnegative')
    loopCounter = 0
    foundPendVector = False
    while not foundPendVector and loopCounter < innerLoopLimit:
      loopCounter += 1
      depVarPos = Distributions.randomIntegers(0,self.nVar-1,self)
      pendVector = {}
      npDot = 0
      for varID, var in enumerate(self.optVars):
        pendVector[var] = self.stochasticEngineForConstraintHandling.rvs() if varID != depVarPos else 0.0
        npDot += pendVector[var]*gradient[var]
      for varID, var in enumerate(self.optVars):
        if varID == depVarPos:
          pendVector[var] = -npDot/gradient[var]

      r = LA.norm(np.asarray([gradient[var] for var in self.optVars]))/LA.norm(np.asarray([pendVector[var] for var in self.optVars]))
      for var in self.optVars:
        pendVector[var] = copy.deepcopy(pendVector[var])*r

      tempVarKPlus = {}
      for index, var in enumerate(self.optVars):
        tempVarKPlus[var] = copy.copy(varK[var]-gain[index]*pendVector[var]*1.0)
      foundPendVector, activeConstraints = self.checkConstraint(tempVarKPlus)
      if not foundPendVector:
        foundPendVector, tempVarKPlus = self._bisectionForConstrainedInput(varK, gain, pendVector)
      gain = gain/2.

    if foundPendVector:
      lenPendVector = 0
      for var in self.optVars:
        lenPendVector += pendVector[var]**2
      lenPendVector = np.sqrt(lenPendVector)

      rotateDegreeUpperLimit = 2
      while self.angleBetween(gradient, pendVector) > rotateDegreeUpperLimit:
        sumVector, lenSumVector = {}, 0
        for var in self.optVars:
          sumVector[var] = gradient[var] + pendVector[var]
          lenSumVector += sumVector[var]**2

        tempTempVarKPlus = {}
        for index, var in enumerate(self.optVars):
          sumVector[var] = copy.deepcopy(sumVector[var]/np.sqrt(lenSumVector)*lenPendVector)
          tempTempVarKPlus[var] = copy.copy(varK[var]-gain[index]*sumVector[var]*1.0)
        satisfied, activeConstraints = self.checkConstraint(tempTempVarKPlus)
        if satisfied:
          tempVarKPlus = copy.deepcopy(tempTempVarKPlus)
          pendVector = copy.deepcopy(sumVector)
        else:
          gradient = copy.deepcopy(sumVector)
      return tempVarKPlus, True
    tempVarKPlus = varK
    return tempVarKPlus, True

  def _bisectionForConstrainedInput(self,varK,ak,vector):
    """
      Method to find the maximum fraction of 'vector' that, when using as gradient, the input can satisfy the constraint
      @ In, varK, dictionary, current variable values
      @ In, ak, float or array, it is gain for variable update (if array, different gain for each variable)
      @ In, vector, dictionary, contains the gradient information for variable update
      @ Out, _bisectionForConstrainedInput, tuple(bool,dict), (indicating whether a fraction vector is found, contains the fraction of gradient that satisfies constraint)
    """
    try:
      gain = ak[:]
    except (TypeError,IndexError):
      gain = [ak]*len(self.optVars)

    innerBisectionThreshold = self.constraintHandlingPara['innerBisectionThreshold']
    if innerBisectionThreshold <= 0 or innerBisectionThreshold >= 1:
      self.raiseAnError(ValueError, 'The innerBisectionThreshold shall be greater than 0 and less than 1')
    bounds = [0, 1.0]
    tempVarNew = {}
    frac = 0.5
    while np.absolute(bounds[1]-bounds[0]) >= innerBisectionThreshold:
      for index, var in enumerate(self.optVars):
        tempVarNew[var] = copy.copy(varK[var]-gain[index]*vector[var]*1.0*frac)
      satisfied, activeConstraints = self.checkConstraint(tempVarNew)
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

  def angleBetween(self, d1, d2):
    """
      Evaluate the angle between the two dictionaries of vars (d1 and d2) by means of the dot product. Unit: degree
      @ In, d1, dict, first vector
      @ In, d2, dict, second vector
      @ Out, angleD, float, angle between d1 and d2 with unit of degree
    """
    v1, v2 = np.zeros(shape=[self.nVar,]), np.zeros(shape=[self.nVar,])
    for cnt, var in enumerate(self.optVars):
      v1[cnt], v2[cnt] = copy.deepcopy(d1[var]), copy.deepcopy(d2[var])
    angle = np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
    if np.isnan(angle):
      if (v1 == v2).all():
        angle = 0.0
      else:
        angle = np.pi
    angleD = np.rad2deg(angle)
    return angleD

  def _computeGainSequenceCk(self,paramDict,iterNum):
    """
      Utility function to compute the ck coefficients (gain sequence ck)
      @ In, paramDict, dict, dictionary containing information to compute gain parameter
      @ In, iterNum, int, current iteration index
      @ Out, ck, float, current value for gain ck
    """
    c, gamma = paramDict['c'], paramDict['gamma']
    ck = c / (iterNum) ** gamma *1.0
    return ck

  def _computeGainSequenceAk(self,paramDict,iterNum,traj):
    """
      Utility function to compute the ak coefficients (gain sequence ak)
      @ In, paramDict, dict, dictionary containing information to compute gain parameter
      @ In, iterNum, int, current iteration index
      @ Out, ak, float, current value for gain ak
    """
    #TODO FIXME is this a good idea?
    try:
      ak = self.counter['lastStepSize'][traj]
    except KeyError:
      a, A, alpha = paramDict['a'], paramDict['A'], paramDict['alpha']
      ak = a / (iterNum + A) ** alpha
    # modify step size based on the history of the gradients used
    frac = self.fractionalStepChangeFromGradHistory(traj)
    ak *= frac
    self.raiseADebug('step gain size for traj "{}" iternum "{}": {}'.format(traj,iterNum,ak))
    self.counter['lastStepSize'][traj] = ak
    return ak
    # the line search with surrogate unfortunately does not work very well (we use it just at the begin of the search and after that
    # we switch to a decay constant strategy (above)). Another strategy needs to be find.
    #### OLD ###
    ## below is the line search methodology, which didn't prove as effective as we originally hoped.
    #if iterNum > 1 and iterNum <= int(self.limit['mdlEval']/50.0):
    #  # we use a line search algorithm for finding the best learning rate (using a surrogate)
    #  # if it fails, we use a decay rate (ak = a / (iterNum + A) ** alpha)
    #  objEvaluateROM = SupervisedLearning.returnInstance('SciKitLearn', self, **{'SKLtype':'neighbors|KNeighborsRegressor', 'Features':','.join(list(self.optVars)), 'Target':self.objVar, 'n_neighbors':5,'weights':'distance'})
    #  tempDict = copy.copy(self.mdlEvalHist.getParametersValues('inputs', nodeId = 'RecontructEnding'))
    #  tempDict.update(self.mdlEvalHist.getParametersValues('outputs', nodeId = 'RecontructEnding'))
    #  for key in tempDict.keys():
    #    tempDict[key] = np.asarray(tempDict[key])
    #  objEvaluateROM.train(tempDict)

    #  def f(x):
    #    """
    #      Method that just interface the evaluate method for the surrogate
    #      @ In, x, numpy.array, coordinate where to evaluate f
    #      @ Out, f, float, result
    #    """
    #    features = {}
    #    for cnt, value in enumerate(x):
    #      features[self.optVars[cnt]] = np.asarray(value)
    #    return objEvaluateROM.evaluate(features)[self.objVar]

    #  def fprime(x):
    #    """
    #      Method that just interface the computes the approximate derivatives using the surrogate
    #      @ In, x, numpy.array, coordinate where to evaluate f'
    #      @ Out, f, numpy.array, partial derivatives
    #    """
    #    return scipy.optimize.approx_fprime(x, f, self._computeGainSequenceCk(self.paramDict,self.counter['varsUpdate'][traj]+1))

    #  xK             = np.asarray([self.optVarsHist[traj][iterNum-1][key] for key in self.optVars])
    #  xKPrevious     = np.asarray([self.optVarsHist[traj][iterNum-2][key] for key in self.optVars])
    #  #xK             = np.asarray([self.denormalizeData(self.optVarsHist[traj][iterNum-1])[key] for key in self.optVars])
    #  #xKPrevious     = np.asarray([self.denormalizeData(self.optVarsHist[traj][iterNum-2])[key] for key in self.optVars])
    #  gradxK         = np.asarray([self.counter['gradientHistory'][traj][0][key] for key in self.optVars])#/self.counter['gradNormHistory'][traj][0]
    #  gradxKPrevious = np.asarray([self.counter['gradientHistory'][traj][1][key] for key in self.optVars])#/self.counter['gradNormHistory'][traj][1]
    #  alphaLineSearchCurrent  = scipy.optimize.line_search(f, fprime, xK, gradxK, amax=10.0)
    #  alphaLineSearchPrevious = scipy.optimize.line_search(f, fprime, xKPrevious, gradxKPrevious, amax=10.0)
    #  akCurrent, akPrevious = 0.0, 0.0
    #  if alphaLineSearchCurrent[-1] is not None:
    #    akCurrent = min(float(alphaLineSearchCurrent[0]),a)
    #  if alphaLineSearchPrevious[-1] is not None:
    #    akPrevious = min(float(alphaLineSearchPrevious[0]),a)
    #  newAk = (akCurrent+akPrevious)/2.
    #  print(ak,newAk)
    #  if newAk != 0.0:
    #    ak = newAk
    #return ak
