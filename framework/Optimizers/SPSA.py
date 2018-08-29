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
import scipy
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
    param.addSub(InputData.parameterInputFactory('gamma', contentType=InputData.FloatType, strictMode=True))
    param.addSub(InputData.parameterInputFactory('c'    , contentType=InputData.FloatType, strictMode=True))
    param.addSub(InputData.parameterInputFactory('a'    , contentType=InputData.FloatType, strictMode=True))
    param.addSub(InputData.parameterInputFactory('alpha', contentType=InputData.FloatType, strictMode=True))
    param.addSub(InputData.parameterInputFactory('A'    , contentType=InputData.FloatType, strictMode=True))

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
    self.paramDict['alpha'] = float(self.paramDict.get('alpha', 0.602))
    self.paramDict['gamma'] = float(self.paramDict.get('gamma', 0.101))
    self.paramDict['A']     = float(self.paramDict.get('A', self.limit['mdlEval']/10.))
    self.paramDict['a']     = self.paramDict.get('a', None)
    self.paramDict['c']     = float(self.paramDict.get('c', 0.005))
    #FIXME the optimization parameters should probably all operate ONLY on normalized data!
    #  -> perhaps the whole optimizer should only work on optimized data.

    numValues = self._numberOfSamples()

    #FIXME normalizing doesn't seem to have the desired effect, currently; it makes the step size very small (for large scales)
    #if "a" was defaulted, use the average scale of the input space.
    #This is the suggested value from the paper, missing a 1/gradient term since we don't know it yet.
    if self.paramDict['a'] is None:
      self.paramDict['a'] = mathUtils.hyperdiagonal(np.ones(numValues)) # the features are always normalized
      self.raiseAMessage('Defaulting "a" gradient parameter to',self.paramDict['a'])
    else:
      self.paramDict['a'] = float(self.paramDict['a'])

    self.constraintHandlingPara['innerBisectionThreshold'] = float(self.paramDict.get('innerBisectionThreshold', 1e-2))
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

  def _newOptPointAdd(self, gradient, traj):
    """
      This local method add a new opt point based on the gradient
      @ In, gradient, dict, dictionary containing the gradient
      @ In, traj, int, trajectory
      @ Out, None
    """
    ak = self._computeGainSequenceAk(self.paramDict,self.counter['varsUpdate'][traj],traj) # Compute the new ak
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
    varK = dict((var,self.counter['recentOptHist'][traj][0][var]) for var in self.getOptVars(traj))
    varKPlus,modded = self._generateVarsUpdateConstrained(traj,ak,gradient,varK)
    #check for redundant paths
    if len(self.optTrajLive) > 1 and self.counter['solutionUpdate'][traj] > 0:
      self._removeRedundantTraj(traj, varKPlus)
    #if the new point was modified by the constraint, reset the step size
    if modded:
      del self.counter['lastStepSize'][traj]
      self.raiseADebug('Resetting step size for trajectory',traj,'due to hitting constraints')
    self.queueUpOptPointRuns(traj,varKPlus)

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
    # loop over live trajectories and look for any actions that are ready to proceed
    # -> indicate the first ready action through the self.nextActionNeeded mechanic
    self.raiseADebug('Reviewing status of trajectories:')
    for traj in self.optTraj:
      self.raiseADebug('   Traj: "{:^n}": Process: "{:^30.30}", Reason: "{:^30.30}"'.format(traj,self.status[traj].get('process','None'),self.status[traj].get('reason','None'),n=len(str(max(self.optTraj)))))
    for _ in range(len(self.optTrajLive)):
      # despite several attempts, this is the most elegant solution I've found to assure each
      #   trajectory gets even treatment.
      traj = self.optTrajLive.pop(0)
      self.optTrajLive.append(traj)
      self.raiseADebug('Checking readiness for traj "{}" ...'.format(traj))
      process = self.status[traj]['process']
      reason = self.status[traj]['reason']

      # still in the process of submitting new points for evaluating the gradient
      if process == 'submitting grad eval points':
        self.nextActionNeeded = ('add new grad evaluation point',traj)
        break

      # still in the process of submitting new opt point evaluations (for stochastic denoising)
      elif process == 'submitting new opt points':
        if reason == 'just started':
          self.nextActionNeeded = ('start new trajectory',traj)
          break
        elif reason in ['seeking new opt point','received recommended point']:
          self.nextActionNeeded = ('add more opt point evaluations',traj)
          break
        elif reason in 'failed run':
          self.nextActionNeeded = ('add more opt point evaluations',traj)
          if len(self.submissionQueue[traj]) == 0:
            gradient = self.counter['gradientHistory'][traj][0]
            self._newOptPointAdd(gradient,traj)
        else:
          self.raiseAnError(RuntimeError,'unexpected reason for submitting new opt points:',reason)

      # all gradient evaluation points submitted, but waiting for them to be collected
      elif process == 'collecting grad eval points':
        # check to see if the grad evaluation points have all been collected
        evalsFinished = True
        for pertID in range(self.gradDict['pertNeeded']):
          if not self._checkModelFinish(traj,self.counter['varsUpdate'][traj],pertID)[0]:
            evalsFinished = False
            break
        # if grad eval pts are finished, then evaluate the gradient
        if evalsFinished:
          # collect output values for perturbed points
          #for i in range(1,self.gradDict['numIterForAve']*2,2):
          for i in self.perturbationIndices:
            evalIndex = self._checkModelFinish(traj,self.counter['varsUpdate'][traj],i)[1]
            outval = float(self.mdlEvalHist.realization(index=evalIndex)[self.objVar])
            self.gradDict['pertPoints'][traj][i]['output'] = outval
          # reset per-opt-point counters, forward the varsUpdate
          self.counter['perturbation'][traj] = 0
          self.counter['varsUpdate'][traj] += 1
          # evaluate the gradient
          gradient = self.evaluateGradient(self.gradDict['pertPoints'][traj],traj)
          # establish a new point, if found; FIXME otherwise?
          if len(self.submissionQueue[traj]) == 0:
            self._newOptPointAdd(gradient,traj)
            # if trajectory was killed for redundancy, continue on to check next trajectory for readiness
            if self.status[traj]['reason'] == 'removed as redundant':
              continue # loops back to the next opt traj
          self.nextActionNeeded = ('add more opt point evaluations',traj)
          self.status[traj]['process'] = 'submitting new opt points'
          self.status[traj]['reason'] = 'seeking new opt point'
          break
        # otherwise, we're not ready to sample yet
        else:
          self.raiseADebug('    Traj "{}": Waiting on collection of gradient evaluation points.'.format(traj))
          continue

      # all opt point evaluations have been submitted, but waiting for them to be collected
      elif process == 'collecting new opt points':
        self.raiseADebug('    Traj "{}": Waiting on collection of new optimization point.'.format(traj))
        continue

      # trajectory has already converged, so there's no work to do for it.
      elif reason == 'converged':
        self.raiseADebug('    Traj "{}": Trajectory is marked as converged.'.format(traj))
        continue

      # trajectory didn't converge, but was killed because of redundancy
      elif reason == 'removed as redundant':
        self.raiseADebug('    Traj "{}": Trajectory is removed for redundancy.'.format(traj))
        continue
      # unknown status
      else:
        self.raiseAnError(RuntimeError,'Unrecognized status:'.format(traj),self.status[traj])
    # end loop through trajectories looking for new actions

    # if we did not find an action, we're not ready to provide an input
    if self.nextActionNeeded[0] is None:
      self.raiseADebug('Not ready to provide a sample yet.')
      return False
    else:
      self.raiseADebug('Next action needed: "%s" on trajectory "%i"' %self.nextActionNeeded)
      return True

  def localEvaluateGradient(self, optVarsValues, traj,  gradient = None):
    """
      Local method to evaluate gradient.
      @ In, optVarsValues, dict, dictionary containing perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ In, traj, int, the trajectory id
      @ In, gradient, dict, optional, dictionary containing gradient estimation by the caller.
                                      gradient should have the form {varName: gradEstimation}
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    gradArray = {}
    # number of gradient evaluations to consider (denoising or resampling)
    numGrads = self.gradDict['numIterForAve']
    # prepopulate array of collected gradient
    for var in self.getOptVars(traj=traj):
      gradArray[var] = np.zeros(numGrads,dtype=object)
    # Evaluate gradient at each point
    for i in range(numGrads):
      opt  = optVarsValues[i]            #the latest opt point
      pert = optVarsValues[i + numGrads] #the perturbed point
      # calculate grad(F) wrt each input variable
      # fix infinities!
      lossDiff = mathUtils.diffWithInfinites(pert['output'],opt['output'])
      #cover "max" problems
      # TODO it would be good to cover this in the base class somehow, but in the previous implementation this
      #   sign flipping was only called when evaluating the gradient.
      #   Perhaps the sign should flip when evaluating the next point to take, instead of forcing gradient descent
      if self.optType == 'max':
        lossDiff *= -1.0
      for var in self.getOptVars(traj=traj):
        # NOTE: gradient is calculated in normalized space
        dh = pert['inputs'][var] - opt['inputs'][var]
        # a sample so close cannot be taken without violating minimum step, so this check should not be necessary (left for reference)
        #if abs(dh) < 1e-15:
        #  self.raiseAnError(RuntimeError,'While calculating the gradArray a "dh" of zero was found for var:',var)
        gradArray[var][i] = lossDiff/dh
    gradient = {}
    for var in self.getOptVars(traj=traj):
      mean = gradArray[var].mean()
      gradient[var] = np.atleast_1d(mean)
    return gradient

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    GradientBasedOptimizer.localGenerateInput(self,model,oldInput)
    action, traj = self.nextActionNeeded
    #store traj as active for sampling
    self.inputInfo['trajID'] = traj+1
    self.inputInfo['varsUpdate'] = self.counter['varsUpdate'][traj]
    #"action" and "traj" are set in localStillReady
    #"action" is a string of the next action needed by the optimizer in order to move forward
    #"traj" is the trajectory that is in need of the action

    if action == 'start new trajectory':
      if len(self.submissionQueue[traj]) == 0:
        self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
        values = {}
        for var in self.getOptVars(traj=traj):
          values[var] = self.optVarsInit['initial'][var][traj]
          #if exceeding bounds, bring value within 1% of range
          values[var] = self._checkBoundariesAndModify(self.optVarsInit['upperBound'][var],
                                                            self.optVarsInit['lowerBound'][var],
                                                            self.optVarsInit['ranges'][var],values[var],0.99,0.01)
        # record this suggested new optimal point (in normalized input space)
        data = self.normalizeData(values)
        self.updateVariableHistory(values,traj)
        # set up a new batch of runs on the new optimal point (batch size 1 unless more requested by user)
        self.queueUpOptPointRuns(traj,data)
      #"submit" the next queued point
      prefix,point = self.getQueuedPoint(traj)
      for var in self.getOptVars(traj=traj):
        self.values[var] = point[var]
      self.inputInfo['prefix'] = prefix
      # if all iterations have been submitted, the optimizer is now in collection mode.
      if len(self.submissionQueue[traj]) == 0:
        self.status[traj]['process'] = 'collecting new opt points'

    elif action == 'add new grad evaluation point':
      #increment submission number
      self.counter['perturbation'][traj] += 1
      #if this is the first perturbation, prep all the perturbation (aka gradient evaluation) points we need to run
      #note that currently we use the opt point as half of the perturbation point, so each gradient eval will be between
      #   the opt point and a different perturbed point
      if self.counter['perturbation'][traj] == 1:
        # Generate all the perturbations at once, then we can submit them one at a time
        ck = self._computeGainSequenceCk(self.paramDict,self.counter['varsUpdate'][traj]+1)
        varK = dict((var,self.counter['recentOptHist'][traj][0][var]) for var in self.getOptVars(traj))
        #check the submission queue is empty; otherwise something went wrong # TODO this is a sanity check, might be removed for efficiency
        assert(len(self.submissionQueue[traj])==0)
        for i in self.perturbationIndices:
          direction = self._getPerturbationDirection(i, traj)
          point = {}
          index = 0
          for var in self.getOptVars(traj=traj):
            size = np.prod(self.variableShapes[var])
            if size > 1:
              new = np.zeros(size)
              for v,origVal in enumerate(varK[var]):
                new[v] = origVal + ck*direction[index]
                new[v] = self._checkBoundariesAndModify(1.0, 0.0, 1.0, new[v], 0.9999, 0.0001)
                index += 1
              point[var] = new
            else:
              val = varK[var] + ck*direction[index]
              index += 1
              val = self._checkBoundariesAndModify(1.0, 0.0, 1.0, val, 0.9999, 0.0001)
              point[var] = val
          #create identifier
          prefix = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],i)
          #queue it up
          self.submissionQueue[traj].append({'inputs':point,'prefix':prefix})
      #end if-first-time conditional
      #get a queued entry to run
      entry = self.submissionQueue[traj].popleft()
      prefix = entry['prefix']
      point = entry['inputs']
      self.gradDict['pertPoints'][traj][int(prefix.split('_')[-1])] = {'inputs':point}#self.normalizeData(point)}
      point = self.denormalizeData(point)
      for var in self.getOptVars(traj=traj):
        self.values[var] = point[var]
      # use 'prefix' to locate the input sent out, as <traj>_<varUpdate>_<eval number>, as 0_0_2
      self.inputInfo['prefix'] = prefix
      # if all required points are submitted, switch into collection mode
      if len(self.submissionQueue[traj]) == 0:
        self.status[traj]['process'] = 'collecting grad eval points'

    elif action == 'add more opt point evaluations':
      #take a sample from the queue
      prefix,point = self.getQueuedPoint(traj)
      #prep the point for running
      for var in self.getOptVars(traj=traj):
        self.values[var] = point[var]
      self.updateVariableHistory(self.values,traj)
      # use 'prefix' to locate the input sent out
      self.inputInfo['prefix'] = prefix
      # if all points submitted (after current one gets submitted), switch to collection mode
      if len(self.submissionQueue[traj]) == 0:
        self.status[traj]['process'] = 'collecting new opt points'

    #unrecognized action
    else:
      self.raiseAnError(RuntimeError,'Unrecognized "action" in localGenerateInput:',action)
    self.raiseADebug('Queuing run "{}"'.format(self.inputInfo['prefix']))

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
    for var in self.getOptVars(traj=traj):
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
      depVarPos = randomUtils.randomIntegers(0,len(self.getOptVars(traj=traj))-1,self)
      # if that variable is multidimensional, pick a dimension -> this is not precisely equal probability of picking, but that should be okay.
      varSize = np.prod(self.variableShapes[var])
      if varSize > 1:
        depVarIdx = randomUtils.randomIntegers(0,varSize-1,self)
      pendVector = {}
      npDot = 0
      for varID, var in enumerate(self.getOptVars(traj=traj)):
        varSize = np.prod(self.variableShapes[var])
        if varSize == 1:
          pendVector[var] = self.stochasticEngineForConstraintHandling.rvs() if varID != depVarPos else 0.0
          npDot += pendVector[var]*gradient[var]
        else:
          for i in range(varSize):
            pendVector[var][i] = self.stochasticEngineForConstraintHandling.rvs() if (varID != depVarPos and depVarIdx != i) else 0.0
          npDot += np.sum(pendVector[var]*gradient[var])
      # TODO does this need to be in a separate loop or can it go with above?
      for varID, var in enumerate(self.getOptVars(traj=traj)):
        if varID == depVarPos:
          varSize = np.prod(self.variableShapes[var])
          if varSize == 1:
            pendVector[var] = -npDot/gradient[var]
          else:
            pendVector[var][depVarIdx] = -npDot/gradient[var][depVarIdx]

      r  = self.calculateMultivectorMagnitude([  gradient[var] for var in self.getOptVars(traj=traj)])
      r /= self.calculateMultivectorMagnitude([pendVector[var] for var in self.getOptVars(traj=traj)])
      for var in self.getOptVars(traj=traj):
        pendVector[var] = copy.deepcopy(pendVector[var])*r

      varKPlus = {}
      index = 0
      for var in self.getOptVars(traj=traj):
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
      for var in self.getOptVars(traj=traj):
        lenPendVector += np.sum(pendVector[var]**2)
      lenPendVector = np.sqrt(lenPendVector)

      rotateDegreeUpperLimit = 2
      while self.angleBetween(traj,gradient, pendVector) > rotateDegreeUpperLimit:
        sumVector = {}
        lenSumVector = 0
        for var in self.getOptVars(traj=traj):
          sumVector[var] = gradient[var] + pendVector[var]
          lenSumVector += np.sum(sumVector[var]**2)

        tempTempVarKPlus = {}
        index = 0
        for var in self.getOptVars(traj=traj):
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
    if innerBisectionThreshold <= 0 or innerBisectionThreshold >= 1: #FIXME REWORK this is an input check, not a runtime check
      self.raiseAnError(ValueError, 'The innerBisectionThreshold should be greater than 0 and less than 1')
    bounds = [0, 1.0]
    tempVarNew = {}
    frac = 0.5
    while np.absolute(bounds[1]-bounds[0]) >= innerBisectionThreshold:
      index = 0
      for var in self.getOptVars(traj=traj):
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
    for cnt, var in enumerate(self.getOptVars(traj=traj)):
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
      @ Out, new, float, current value for gain ak
    """
    #TODO FIXME is this a good idea?
    try:
      ak = self.counter['lastStepSize'][traj]
    except KeyError:
      a, A, alpha = paramDict['a'], paramDict['A'], paramDict['alpha']
      ak = a / (iterNum + A) ** alpha
    # modify step size based on the history of the gradients used
    frac = self.fractionalStepChangeFromGradHistory(traj)
    new = ak*frac
    self.raiseADebug('step gain size for traj "{}" iternum "{}": {:1.3e} (root {:1.2e} frac {:1.2e})'.format(traj,iterNum,new,ak,frac))
    self.counter['lastStepSize'][traj] = new
    return new

  def _getAlgorithmState(self,traj):
    """
      Returns values specific to this algorithm such that it could pick up again relatively easily from here.
      #REWORK functionally a getstate/setstate for multilevel
      @ In, traj, int, the trajectory being saved
      @ Out, state, dict, keys:values this algorithm cares about saving for this trajectory
    """
    state = {}
    state['lastStepSize']    = copy.deepcopy(self.counter['lastStepSize'   ].get(traj,None))
    state['gradientHistory'] = copy.deepcopy(self.counter['gradientHistory'].get(traj,None))
    state['recommendToGain'] = copy.deepcopy(self.recommendToGain           .get(traj,None))
    return state

  def _setAlgorithmState(self,traj,state):
    """
      #REWORK functionally a getstate/setstate for multilevel
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

  def _getPerturbationDirection(self,perturbationIndex, traj):
    """
      This method is aimed to get the perturbation direction (i.e. in this case the random perturbation versor)
      @ In, perturbationIndex, int, the perturbation index (stored in self.perturbationIndices)
      @ In, traj, int, the trajectory id
      @ Out, direction, list, the versor for each optimization dimension
    """
    if perturbationIndex == self.perturbationIndices[0]:
      direction = self.stochasticEngine()
      self.currentDirection = direction
    else:
      # in order to perform the de-noising we keep the same perturbation direction and we repeat the evaluation multiple times
      direction = self.currentDirection
    return direction
