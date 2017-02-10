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
    self.paramDict['a']     = float(self.paramDict.get('a', 0.16))
    self.paramDict['c']     = float(self.paramDict.get('c', 0.005))

    # Normalize the parameters...
    if self.gradDict['normalize']:
      tempMax = -1
      for var in self.optVars:
        if self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var] > tempMax:
          tempMax = self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var]
      self.paramDict['c'] = copy.deepcopy(self.paramDict['c']/tempMax)
      self.paramDict['a'] = copy.deepcopy(self.paramDict['a']/(tempMax**2))

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
      self.raiseAnError(IOError, self.paramDict['stochasticEngine']+'is currently not support for SPSA')

  def localLocalInitialize(self, solutionExport = None):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    self._endJobRunnable = 1
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2

  def localLocalStillReady(self, ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, variable indicating whether the caller is prepared for another input.
    """
    if convergence:             ready = False
    else:                       ready = True and ready
    return ready

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    GradientBasedOptimizer.localGenerateInput(self,model,oldInput)

    if self.counter['mdlEval'] <= len(self.optTraj): # Just started
      traj = self.optTrajLive.pop(0)
      self.optTrajLive.append(traj)
      self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
      for var in self.optVars:
        self.values[var] = self.optVarsInit['initial'][var][traj]
        if self.values[var] >=  self.optVarsInit['upperBound'][var]: self.values[var]-= 0.01*(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])
        if self.values[var] <=  self.optVarsInit['lowerBound'][var]: self.values[var]+= 0.01*(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])
      data = self.normalizeData(self.values) if self.gradDict['normalize'] else self.values
      self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = copy.deepcopy(data)
      # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
      self.inputInfo['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],'v')
    else:
      while True: # this while loop is needed to loop over all trajectories to find one that is ready for update.
        traj = self.optTrajLive.pop(0)
        self.optTrajLive.append(traj)
        if self.counter['perturbation'][traj] < self.gradDict['pertNeeded']:
          self.counter['perturbation'][traj] += 1
        else:
          self.readyVarsUpdate[traj] = True
        if not self.readyVarsUpdate[traj]: # Not ready to update decision variables; continue to perturb for gradient evaluation
          if self.counter['perturbation'][traj] == 1: # Generate all the perturbations at once
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
                  self.gradDict['pertPoints'][traj][ind][var] = np.concatenate((p1, p2))

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
          break
        else: # Enough gradient evaluation for decision variable update
          evalNotFinish = False
          for pertID in range(1,self.gradDict['pertNeeded']+1):
            if not self._checkModelFinish(traj,self.counter['varsUpdate'][traj],pertID)[0]:
              evalNotFinish = True
              break
          if evalNotFinish:  # evaluation not completed for gradient evaluation
            continue
          else:  # evaluation completed for gradient evaluation
            self.counter['perturbation'][traj] = 0
            self.counter['varsUpdate'][traj] += 1

            ak = self._computeGainSequenceAk(self.paramDict,self.counter['varsUpdate'][traj]) # Compute the new ak
            gradient = self.evaluateGradient(self.gradDict['pertPoints'][traj], traj)
            self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
            varK = copy.deepcopy(self.optVarsHist[traj][self.counter['varsUpdate'][traj]-1])

            varKPlus = self._generateVarsUpdateConstrained(ak,gradient,varK)
            varKPlusDenorm = self.denormalizeData(varKPlus) if self.gradDict['normalize'] else varKPlus
            for var in self.optVars:
              self.values[var] = copy.deepcopy(varKPlusDenorm[var])
              self.optVarsHist[traj][self.counter['varsUpdate'][traj]][var] = copy.deepcopy(varKPlus[var])
            # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
            self.inputInfo['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],'v')

            # remove redundant trajectory
            if len(self.optTrajLive) > 1 and self.counter['solutionUpdate'][traj] > 0:
              self._removeRedundantTraj(traj, self.optVarsHist[traj][self.counter['varsUpdate'][traj]])

            break

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
    if satisfied: return tempVarKPlus
    else:
      # check if the active constraints are the boundary ones. In case, project the gradient
      if len(activeConstraints['internal']) > 0:
        projectedOnBoundary= {}
        for activeConstraint in activeConstraints['internal']: projectedOnBoundary[activeConstraint[0]] = activeConstraint[1]
        tempVarKPlus.update(self.normalizeData(projectedOnBoundary) if self.gradDict['normalize'] else projectedOnBoundary)
      if len(activeConstraints['external']) == 0: return tempVarKPlus

    # Try to find varKPlus by shorten the gradient vector
    foundVarsUpdate, tempVarKPlus = self._bisectionForConstrainedInput(varK, ak, gradient)
    if foundVarsUpdate:
      return tempVarKPlus

    # Try to find varKPlus by rotate the gradient towards its orthogonal, since we consider the gradient as perpendicular
    # with respect to the constraints hyper-surface
    innerLoopLimit = self.constraintHandlingPara['innerLoopLimit']
    if innerLoopLimit < 0:   self.raiseAnError(IOError, 'Limit for internal loop for constraint handling shall be nonnegative')
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
    if innerBisectionThreshold <= 0 or innerBisectionThreshold >= 1: self.raiseAnError(ValueError, 'The innerBisectionThreshold shall be greater than 0 and less than 1')
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
      if (v1 == v2).all(): angle = 0.0
      else: angle = np.pi
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
#     if iterNum > 2:
#       traj = 0
#       gradK     = self.counter['gradientHistory'][traj][0].values()
#       gradPrevK = self.counter['gradientHistory'][traj][1].values()
#       xK        =
#       xPrevK    =
#       deltaX    = xK - xPrevK
#       gX        = gradK - gradPrevK
#       ak        = (np.asarray(gX).T * np.asarray(deltaX))/(np.asarray(gX)*np.asarray(gX).T)
    a, A, alpha = paramDict['a'], paramDict['A'], paramDict['alpha']
    ak = a / (iterNum + A) ** alpha *1.0
    return ak
