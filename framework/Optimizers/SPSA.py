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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .GradientBasedOptimizer import GradientBasedOptimizer
import Distributions
from utils import mathUtils
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
    self.paramDict['A']     = float(self.paramDict.get('A', self.limit['mdlEval']/10))
    self.paramDict['a']     = self.paramDict.get('a', None)
    self.paramDict['c']     = float(self.paramDict.get('c', 0.005))
    #FIXME the optimization parameters should probably all operate ONLY on normalized data!
    #  -> perhaps the whole optimizer should only work on optimized data.

    #FIXME normalizing doesn't seem to have the desired effect, currently; it makes the step size very small (for large scales)
    #if "a" was defaulted, use the average scale of the input space.
    #This is the suggested value from the paper, missing a 1/gradient term since we don't know it yet.
    if self.paramDict['a'] is None:
      self.paramDict['a'] = mathUtils.hyperdiagonal(self.optVarsInit['ranges'].values()) #should be normalized!
      self.raiseAMessage('Defaulting "a" gradient parameter to',self.paramDict['a'])
    else:
      self.paramDict['a'] = float(self.paramDict['a'])

    # Normalize the parameters...
    if self.gradDict['normalize']:
      maxVarRange = max(self.optVarsInit['ranges'].values())
      #tempMax = -1
      #for var in self.optVars:
      #  if self.optVarsInit['ranges'][var] > tempMax:
      #    tempMax = self.optVarsInit['ranges'][var]
      self.paramDict['c'] = copy.deepcopy(self.paramDict['c']/maxVarRange) #FIXME why are these deepcopied?
      self.paramDict['a'] = copy.deepcopy(self.paramDict['a']/(maxVarRange**2)) #FIXME why are these deepcopied? And why is this normalized to the square?

    self.constraintHandlingPara['innerBisectionThreshold'] = float(self.paramDict.get('innerBisectionThreshold', 1e-2))
    self.constraintHandlingPara['innerLoopLimit'] = float(self.paramDict.get('innerLoopLimit', 1000))

    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2

    if self.paramDict.get('stochasticDistribution', 'Bernoulli') == 'Bernoulli':
      self.stochasticDistribution = Distributions.returnInstance('Bernoulli',self)
      self.stochasticDistribution.p = 0.5
      self.stochasticDistribution.initializeDistribution()
      # Initialize bernoulli distribution for random perturbation. Add artificial noise to avoid that specular loss functions get false positive convergence
      self.stochasticEngine = lambda: [1.0+(Distributions.random()/1000.0)*Distributions.randomIntegers(-1, 1, self) if self.stochasticDistribution.rvs() == 1 else
                                      -1.0+(Distributions.random()/1000.0)*Distributions.randomIntegers(-1, 1, self) for _ in range(self.nVar)]
    else:
      self.raiseAnError(IOError, self.paramDict['stochasticEngine']+'is currently not supported for SPSA')

  def localLocalInitialize(self, solutionExport = None):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    self._endJobRunnable = 1
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2

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
          if not self._checkModelFinish(traj,self.counter['varsUpdate'][traj],pertID):#[0]:
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
        if self.values[var] >= self.optVarsInit['upperBound'][var]:
          self.values[var] = 0.99*(self.optVarsInit['ranges'][var]) + self.optVarsInit['lowerBound'][var]
          # self.values[var]-= 0.01*(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])
        if self.values[var] <= self.optVarsInit['lowerBound'][var]:
          self.values[var] = 0.01*(self.optVarsInit['ranges'][var]) + self.optVarsInit['lowerBound'][var]
      data = self.normalizeData(self.values) if self.gradDict['normalize'] else self.values
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
        for ind in range(self.gradDict['numIterForAve']):
          self.gradDict['pertPoints'][traj][ind] = {}
          delta = self.stochasticEngine()
          for varID, var in enumerate(self.optVars):
            if var not in self.gradDict['pertPoints'][traj][ind].keys():
              p1 = np.asarray([varK[var]+ck*delta[varID]*1.0]).reshape((1,))
              p2 = np.asarray([varK[var]-ck*delta[varID]*1.0]).reshape((1,))
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
      tempOptVarsDenorm = copy.deepcopy(self.denormalizeData(tempOptVars)) if self.gradDict['normalize'] else copy.deepcopy(tempOptVars)
      for var in self.optVars:
        self.values[var] = tempOptVarsDenorm[var]
      # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation)
      self.inputInfo['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],self.counter['perturbation'][traj])

    elif action == 'evaluate gradient':
      # evaluation completed for gradient evaluation
      self.counter['perturbation'][traj] = 0
      self.counter['varsUpdate'][traj] += 1

      ak = self._computeGainSequenceAk(self.paramDict,self.counter['varsUpdate'][traj]) # Compute the new ak
      gradient = self.evaluateGradient(self.gradDict['pertPoints'][traj], traj)
      self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
      varK = copy.deepcopy(self.optVarsHist[traj][self.counter['varsUpdate'][traj]-1])
      # FIXME here is where adjustments to the step size should happen
      #TODO this is part of a future request.  Commented for now.
      #get central response for this trajectory: how?? TODO FIXME
      #centralResponseIndex = self._checkModelFinish(traj,self.counter['varsUpdate'][traj]-1,'v')[1]
      #self.estimateStochasticity(gradient,self.gradDict['pertPoints'][traj][self.counter['varsUpdate'][traj]-1],varK,centralResponseIndex) #TODO need current point too!

      varKPlus = self._generateVarsUpdateConstrained(ak,gradient,varK)
      varKPlusDenorm = self.denormalizeData(varKPlus) if self.gradDict['normalize'] else varKPlus
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
      @ In, ak, float, it is gain for variable update
      @ In, gradient, dictionary, contains the gradient information for variable update
      @ In, varK, dictionary, current variable values
      @ Out, tempVarKPlus, dictionary, variable values for next iteration.
    """
    tempVarKPlus = {}
    for var in self.optVars:
      tempVarKPlus[var] = copy.copy(varK[var]-ak*gradient[var]*1.0)
    satisfied, activeConstraints = self.checkConstraint(tempVarKPlus)
    #satisfied, activeConstraints = self.checkConstraint(self.denormalizeData(tempVarKPlus))
    if satisfied:
      return tempVarKPlus
    else:
      # check if the active constraints are the boundary ones. In case, project the gradient
      if len(activeConstraints['internal']) > 0:
        projectedOnBoundary= {}
        for activeConstraint in activeConstraints['internal']:
          projectedOnBoundary[activeConstraint[0]] = activeConstraint[1]
        tempVarKPlus.update(self.normalizeData(projectedOnBoundary) if self.gradDict['normalize'] else projectedOnBoundary)
      if len(activeConstraints['external']) == 0:
        return tempVarKPlus

    # Try to find varKPlus by shorten the gradient vector
    foundVarsUpdate, tempVarKPlus = self._bisectionForConstrainedInput(varK, ak, gradient)
    if foundVarsUpdate:
      return tempVarKPlus

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
      for var in self.optVars:
        tempVarKPlus[var] = copy.copy(varK[var]-ak*pendVector[var]*1.0)
      foundPendVector, activeConstraints = self.checkConstraint(tempVarKPlus)
      if not foundPendVector:
        foundPendVector, tempVarKPlus = self._bisectionForConstrainedInput(varK, ak, pendVector)

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
        for var in self.optVars:
          sumVector[var] = copy.deepcopy(sumVector[var]/np.sqrt(lenSumVector)*lenPendVector)
          tempTempVarKPlus[var] = copy.copy(varK[var]-ak*sumVector[var]*1.0)
        satisfied, activeConstraints = self.checkConstraint(tempTempVarKPlus)
        if satisfied:
          tempVarKPlus = copy.deepcopy(tempTempVarKPlus)
          pendVector = copy.deepcopy(sumVector)
        else:
          gradient = copy.deepcopy(sumVector)

      return tempVarKPlus

    tempVarKPlus = varK
    return tempVarKPlus

  def _bisectionForConstrainedInput(self,varK,gain,vector):
    """
      Method to find the maximum fraction of 'vector' that, when using as gradient, the input can satisfy the constraint
      @ In, varK, dictionary, current variable values
      @ In, gain, float, it is gain for variable update
      @ In, vector, dictionary, contains the gradient information for variable update
      @ Out, _bisectionForConstrainedInput, tuple(bool,dict), (indicating whether a fraction vector is found, contains the fraction of gradient that satisfies constraint)
    """
    innerBisectionThreshold = self.constraintHandlingPara['innerBisectionThreshold']
    if innerBisectionThreshold <= 0 or innerBisectionThreshold >= 1:
      self.raiseAnError(ValueError, 'The innerBisectionThreshold shall be greater than 0 and less than 1')
    fracLowerLimit = 1e-2
    bounds = [0, 1.0]
    tempVarNew = {}
    frac = 0.5
    while np.absolute(bounds[1]-bounds[0]) >= innerBisectionThreshold:
      for var in self.optVars:
        tempVarNew[var] = copy.copy(varK[var]-gain*vector[var]*1.0*frac)
      satisfied, activeConstraints = self.checkConstraint(tempVarNew)
      if satisfied:
        bounds[0] = copy.deepcopy(frac)
        if np.absolute(bounds[1]-bounds[0]) < innerBisectionThreshold:
          if frac >= fracLowerLimit:
            varKPlus = copy.deepcopy(tempVarNew)
            return True, varKPlus
          break
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

  def _computeGainSequenceAk(self,paramDict,iterNum):
    """
      Utility function to compute the ak coefficients (gain sequence ak)
      @ In, paramDict, dict, dictionary containing information to compute gain parameter
      @ In, iterNum, int, current iteration index
      @ Out, ak, float, current value for gain ak
    """
    #This block is going to be used and formalized in the future
    #if iterNum > 1:
    #  traj = 0
    #  gradK     = self.counter['gradientHistory'][traj][0].values()
    #  gradPrevK = self.counter['gradientHistory'][traj][1].values()
    #  xK        =
    #  xPrevK    =
    #  deltaX    = np.asarray(xK) - np.asarray(xPrevK)
    #  gX        = gradK - gradPrevK
    #  ak        = (np.asarray(gX).T * np.asarray(deltaX))/(np.asarray(gX)*np.asarray(gX).T)
    a, A, alpha = paramDict['a'], paramDict['A'], paramDict['alpha']
    ak = a / (iterNum + A) ** alpha *1.0
    return ak
