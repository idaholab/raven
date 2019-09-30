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
  This module contains the Finite Difference Gradient Optimization strategy

  Created on Sept 10, 2017
  @ author: alfoa
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
from numpy import linalg as LA
from numpy import (atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze,
                   asarray, sqrt, Inf, asfarray, isinf)
import scipy
from scipy.optimize import minpack2
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .SPSA import SPSA
from utils import mathUtils,randomUtils
#Internal Modules End--------------------------------------------------------------------------------

class ConjugateGradient(SPSA):
  """
    Finite Difference Gradient Optimizer
    This class currently inherits from the SPSA (since most of the gradient based machinery is there).
    TODO: Move the SPSA machinery here (and GradientBasedOptimizer) and make the SPSA just take care of the
    random perturbation approach
  """
  ##########################
  # Initialization Methods #
  ##########################
  def __init__(self):
    """
      Default Constructor
    """
    SPSA.__init__(self)

    self.isave = np.zeros((2,), np.intc)
    self.dsave = np.zeros((13,), float)
    self.task = b'START'
    self.maxiter = 10

    # self.maxiter = self.limit['mdlEval']
    self.gtol=1e-08
    self.eps=1.4901161193847656e-08
    self.disp=True
    self.maxiter=2000
    self.return_all=True
    self.localmaxiter=100
    self.xk = None
    self.gfk = None
    self.pk = None
    self.k =0
    self.old_fval = None
    self.old_old_fval = None
    self.allvecs = None
    self.gnorm = None
    self.sigma_3 = 0.01
    self.deltak = None
    self.derphi0 = None
    self.alpha1 =  None
    self.stp = None

  def localInputAndChecks(self, xmlNode):
    """
      Local method for additional reading.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    SPSA.localInputAndChecks(self, xmlNode)
    # need extra eval for central Diff, using boolean in math

    self.useCentralDiff = True
    self.paramDict['pertSingleGrad'] = 2 * len(self.fullOptVars)
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * (self.paramDict['pertSingleGrad']+1)
    self.raiseADebug('Central differencing activated!')

  ###############
  # Run Methods #
  ###############
  def localFinalizeActualSampling(self,jobObject,model,myInput):
      """
        Overwrite only if you need something special at the end of each run....
        This function is used by optimizers that need to collect information from the just ended run
        @ In, jobObject, instance, an instance of a Runner
        @ In, model, Model, instance of a RAVEN model
        @ In, myInput, list, the generating input
        @ Out, None
      """
      #collect first output

      if self.task[:5] == b'START':
        prefix = jobObject.getMetadata()['prefix']
        traj, step, identifier = [int(x) for x in prefix.split('_')]
        self.raiseADebug('Collected sample "{}"'.format(prefix))
        category, number, _, cdId = self._identifierToLabel(identifier)
        done, index = self._checkModelFinish(str(traj), str(step), str(identifier))
        number = number + (cdId * len(self.fullOptVars))
        self.realizations[traj]['collect'][category][number].append(index)
        if len(self.realizations[traj]['collect'][category][number]) == self.realizations[traj]['need']:
          outputs = self._averageCollectedOutputs(self.realizations[traj]['collect'][category][number])
          self.realizations[traj]['denoised'][category][number] = outputs
          if category == 'opt':
            converged = self._finalizeOptimalCandidate(traj,outputs)
          else:
            converged = False
          optDone = bool(len(self.realizations[traj]['denoised']['opt'][0]))
          gradDone = all( len(self.realizations[traj]['denoised']['grad'][i]) for i in range(self.paramDict['pertSingleGrad']))
          if not converged and optDone and gradDone:
            optCandidate = self.normalizeData(self.realizations[traj]['denoised']['opt'][0])
            if self.writeSolnExportOn == 'every':
              self.writeToSolutionExport(traj, optCandidate, self.realizations[traj]['accepted'])
            # whether we wrote to solution export or not, update the counter
            self.counter['solutionUpdate'][traj] += 1
            self.counter['varsUpdate'][traj] += 1
            ## since accepted, update history
            try:
              self.counter['recentOptHist'][traj][1] = copy.deepcopy(self.counter['recentOptHist'][traj][0])
            except KeyError:
              # this means we don't have an entry for this trajectory yet, so don't copy anything
              pass
            # store realization of most recent developments
            self.counter['recentOptHist'][traj][0] = optCandidate
            indexmap = self.getOptVars()
            # change xk into array with index matching the index map
            xk = dict((var,self.realizations[traj]['denoised']['opt'][0][var]) for var in indexmap)
            self.xk = np.asarray(list(xk.values()))
            gfk = dict((var,self.localEvaluateGradient(traj)[var][0]) for var in indexmap)
            self.gfk = np.asarray(list(gfk.values()))
            self.old_fval = self.realizations[traj]['denoised']['opt'][0][self.objVar]
            self.old_old_fval = self.old_fval + np.linalg.norm(self.gfk) / 2
            self.allvecs = [self.xk]
            self.pk = -(self.gfk)
            self.gnorm = np.amax(np.abs(self.gfk))
            # first step

            self.deltak = np.dot(self.gfk, self.gfk)
            self.derphi0 = np.dot(self.gfk, self.pk)
            self.alpha1 = min(1.0, 1.01*2*(self.old_fval - self.old_old_fval)/self.derphi0)

            phi1 = self.old_fval
            derphi1 = self.derphi0
            self.stp, _, _, self.task = minpack2.dcsrch(self.alpha1, phi1, derphi1, ftol=1e-4, gtol=0.4,
                                                    xtol=1e-14, task = self.task, stpmin=1e-100,
                                                    stpmax=1e100, isave = self.isave , dsave=self.dsave)
            if self.task[:2] == b'FG':
              self.alpha1 = self.stp
              grad= dict((var,self.gfk[ind]/(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])) for ind,var in enumerate(indexmap))
              new = self._newOptPointAdd(grad, traj)
              if new is not None:
                # add new gradient points
                self._createPerturbationPoints(traj, new)
              # reset storage
              self._setupNewStorage(traj)
            else:
              self.stp = None
              self.raiseAnError(ValueError, 'Not able to calculate the first step')

      elif self.task[:2] == b'FG':
        prefix = jobObject.getMetadata()['prefix']
        traj, step, identifier = [int(x) for x in prefix.split('_')]
        self.raiseADebug('Collected sample "{}"'.format(prefix))
        category, number, _, cdId = self._identifierToLabel(identifier)
        done, index = self._checkModelFinish(str(traj), str(step), str(identifier))
        number = number + (cdId * len(self.fullOptVars))
        self.realizations[traj]['collect'][category][number].append(index)
        if len(self.realizations[traj]['collect'][category][number]) == self.realizations[traj]['need']:
          outputs = self._averageCollectedOutputs(self.realizations[traj]['collect'][category][number])
          print('outputoutputoutput',outputs)
          self.realizations[traj]['denoised'][category][number] = outputs
          if category == 'opt':
            converged = self._finalizeOptimalCandidate(traj,outputs)
          else:
            converged = False
          optDone = bool(len(self.realizations[traj]['denoised']['opt'][0]))
          gradDone = all( len(self.realizations[traj]['denoised']['grad'][i]) for i in range(self.paramDict['pertSingleGrad']))
          if not converged and optDone and gradDone:
            optCandidate = self.normalizeData(self.realizations[traj]['denoised']['opt'][0])
            if self.writeSolnExportOn == 'every':
              print('everyeveryevery')
              self.writeToSolutionExport(traj, optCandidate, self.realizations[traj]['accepted'])
            # whether we wrote to solution export or not, update the counter
            self.counter['solutionUpdate'][traj] += 1
            self.counter['varsUpdate'][traj] += 1
            ## since accepted, update history
            try:
              self.counter['recentOptHist'][traj][1] = copy.deepcopy(self.counter['recentOptHist'][traj][0])
            except KeyError:
              # this means we don't have an entry for this trajectory yet, so don't copy anything
              pass
            indexmap = self.getOptVars()
            newGrad = dict((var,self.localEvaluateGradient(traj)[var][0]) for var in indexmap)
            newGrad = np.asarray(list(newGrad.values()))
            # after step

            phi1 = self.realizations[traj]['denoised']['opt'][0][self.objVar]
            derphi1 = np.dot(newGrad, self.pk)
            print('youyouchecking self.alpha1,phi1,derphi1',self.alpha1,phi1,derphi1)
            self.stp, _, _, self.task = minpack2.dcsrch(self.alpha1, phi1, derphi1, ftol=1e-4, gtol=0.4,
                                                    xtol=1e-14, task = self.task, stpmin=1e-100,
                                                    stpmax=1e100, isave = self.isave , dsave=self.dsave)

            if self.task[:2] == b'FG':
              self.alpha1 = self.stp
              grad= dict((var,self.gfk[ind]/(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])) for ind,var in enumerate(indexmap))
              new = self._newOptPointAdd(grad, traj)
              if new is not None:
                # add new gradient points
                self._createPerturbationPoints(traj, new)
              # reset storage
              self._setupNewStorage(traj)
            elif self.task[:11] == b'CONVERGENCE' :
              self.task = b'START'
              self.alpha1 = self.stp
              grad= dict((var,self.gfk[ind]/(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])) for ind,var in enumerate(indexmap))
              new = self._newOptPointAdd(grad, traj)
              if new is not None:
                # add new gradient points
                self._createPerturbationPoints(traj, new)
              # reset storage
              self._setupNewStorage(traj)
            elif self.task[:7] == b'WARNING' :
              self.raiseAWarning('Desired error not necessarily achieved due to precision loss.')
            else:
              self.stp = None
              print(self.task)
              self.raiseAnError(ValueError, 'Not able to calculate the fw step')

      else:
        print(self.task)
        self.raiseAnError(ValueError, 'Not able to calculate the do what')
        pass


  ###################
  # Utility Methods #
  ###################
  def _getPerturbationDirection(self, perturbationIndex, step = None):
    """
      This method is aimed to get the perturbation direction (i.e. in this case the random perturbation versor)
      @ In, perturbationIndex, int, the perturbation index (stored in self.perturbationIndices)
      @ In, step, int, the step index, zero indexed
      @ Out, direction, list, the versor for each optimization dimension
    """
    _, varId, denoId, cdId = self._identifierToLabel(perturbationIndex)
    direction = np.zeros(len(self.getOptVars())).tolist()
    if cdId == 0:
      direction[varId] = 1.0
    else:
      direction[varId] = -1.0

    self.currentDirection = direction
    return direction

  def localEvaluateGradient(self, traj):
    """
      Local method to evaluate gradient.
      @ In, traj, int, the trajectory id
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    gradient = {}
    gi = {}
    inVars = self.getOptVars()
    opt = self.realizations[traj]['denoised']['opt'][0]
    allGrads = self.realizations[traj]['denoised']['grad']
    for g,pert in enumerate(allGrads):
      varId = g % len(inVars)
      var = inVars[varId]
      lossDiff = mathUtils.diffWithInfinites(pert[self.objVar],opt[self.objVar])
      # unlike SPSA, keep the loss diff magnitude so we get the exact right direction
      if self.optType == 'max':
        lossDiff *= -1.0
      dh = pert[var] - opt[var]
      if abs(dh) < 1e-15:
        self.raiseADebug('Checking Var:',var)
        self.raiseADebug('Opt point   :',opt)
        self.raiseADebug('Grad point  :',pert)
        self.raiseAnError(RuntimeError,'While calculating the gradArray a "dh" very close to zero was found for var:',var)
      if gradient.get(var) == None:
        gi[var] = 0
        gradient[var] = np.atleast_1d(lossDiff / dh)
      else:
        gi[var] += 1
        gradient[var] = (gradient[var] + np.atleast_1d(lossDiff / dh))/(gi[var]  + 1)
    return gradient

  def _newOptPointAdd(self, gradient, traj):
    """
      This local method add a new opt point based on the gradient
      @ In, gradient, dict, dictionary containing the gradient
      @ In, traj, int, trajectory
      @ Out, varKPlus, dict, new point that has been queued (or None if no new points should be run for this traj)
    """
    stepSize = self.stp
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
    varK = dict((var,self.counter['recentOptHist'][traj][0][var]) for var in self.getOptVars())
    print('jz is looking into this traj, stepSize, gradient, varK', traj, stepSize, gradient, varK)
    varKPlus,modded = self._generateVarsUpdateConstrained(traj, stepSize, gradient, varK)
    print('what is this what is this',varKPlus)    #check for redundant paths
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

  def _createPerturbationPoints(self, traj, optPoint, submit=True):
    """
      Creates perturbation points based on a provided NORMALIZED data point
      @ In, traj, int, integer label for current trajectory
      @ In, optPoint, dict, current optimal point near which to calculate gradient
      @ In, submit, bool, optional, if True then submit perturbation points to queue
      @ Out, points, list(dict), perturbation points
    """
    print('trajtrajtrajtraj, optPoint, submit',traj, optPoint, submit)
    points = []

    distance = self.paramDict['pertDist'] * self.counter['lastStepSize'][traj]
    # distance = self._computePerturbationDistance(traj,self.paramDict,self.counter['varsUpdate'][traj]+1)
    # print(' distance self.perturbationIndices',distance,self.perturbationIndices distance 2...8)
    for i in self.perturbationIndices:
      direction = self._getPerturbationDirection(i, step = self.counter['varsUpdate'][traj])
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
    print('points',points)

    return points


  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    self.readyVarsUpdate = {traj:False for traj in self.optTrajLive}
    # GradientBasedOptimizer.localGenerateInput(self,model,oldInput)
    # print('model, self.optTraj, self.trajCycle',model,self.optTraj,self.trajCycle)
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
        print('insisde localGenerateInput')#,self.inputInfo)
        # if we found a submission, cease looking for submissions
        return
    # if no submissions were found, then we shouldn't have flagged ourselves as Ready or there's a bigger issue!
    self.raiseAnError(RuntimeError,'Attempted to generate an input but there are none queued to provide!')

  def getQueuedPoint(self,traj,denorm=True):
    """
      Pops the first point off the submission queue (or errors if empty).  By default denormalized the point before returning.
      @ In, traj, int, the trajectory from whose queue we should obtain an entry
      @ In, denorm, bool, optional, if True the input data will be denormalized before returning
      @ Out, prefix, #_#_#
      @ Out, point, dict, {var:val}
    """
    try:
      entry = self.submissionQueue[traj].popleft()
    except IndexError:
      self.raiseAnError(RuntimeError,'Tried to get a point from submission queue of trajectory "{}" but it is empty!'.format(traj))
    prefix = entry['prefix']
    point = entry['inputs']
    print('inside getQueuedPoint')
    if denorm:
      point = self.denormalizeData(point)
    return prefix,point




  def polakRibierePowellStep(self, alpha, gfkp1=None):
    print('polak_ribiere_powell_step')
    xkp1 = xk + alpha * pk
    if gfkp1 is None:
      # gfkp1 = myfprime(xkp1)
      gfkp1 = approx_fprime(xkp1, f, self.eps)
    yk = gfkp1 - gfk
    beta_k = max(0, np.dot(yk, gfkp1) / deltak)
    pkp1 = -gfkp1 + beta_k * pk
    gnorm = np.amax(np.abs(gfkp1))
    return (alpha, xkp1, pkp1, gfkp1, gnorm)

  def descentCondition(self, alpha, xkp1, fp1, gfkp1):
    print('descent_condition')
    print(alpha, xkp1, fp1, gfkp1)
    # Polak-Ribiere+ needs an explicit check of a sufficient
    # descent condition, which is not guaranteed by strong Wolfe.
    #
    # See Gilbert & Nocedal, "Global convergence properties of
    # conjugate gradient methods for optimization",
    # SIAM J. Optimization 2, 21 (1992).
    cached_step[:] = self.polakRibierePowellStep(alpha, gfkp1)
    alpha, xk, pk, gfk, gnorm = cached_step

    # Accept step if it leads to convergence.
    if gnorm <= gtol:
      print('acetptingac')
      return True

    # Accept step if sufficient descent condition applies.
    return np.dot(pk, gfk) <= -sigma_3 * np.dot(gfk, gfk)

  def _finalizeOptimalCandidate(self,traj,outputs):
    """
      Once all the data for an opt point has been collected:
       - determine convergence
       - determine redundancy
       - determine acceptability
       - queue new points (if rejected)
      @ In, traj, int, the trajectory we are currently considering
      @ In, outputs, dict, denoised new optimal point
      @ Out, converged, bool, if True then indicates convergence has been reached
    """
    # check convergence and check if new point is accepted (better than old point)
    accepted = self._updateConvergenceVector(traj, self.counter['solutionUpdate'][traj], outputs)
    # if converged, we can wrap up this trajectory
    if self.convergeTraj[traj]:
      # end any excess gradient evaluation jobs
      self.cancelJobs([self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],i) for i in self.perturbationIndices])
      return True #converged
    # if not accepted, we need to scrap this run and set up a new one
    if accepted:
      # store acceptance for later
      self.realizations[traj]['accepted'] = accepted
    else:
      # update solution export
      optCandidate = self.normalizeData(self.realizations[traj]['denoised']['opt'][0])
      ## only write here if we want to write on EVERY optimizer iteration (each new optimal point)
      if self.writeSolnExportOn == 'every':
        self.writeToSolutionExport(traj, optCandidate, self.realizations[traj]['accepted'])
      # whether we wrote to solution export or not, update the counter
      self.counter['solutionUpdate'][traj] += 1
      self.counter['varsUpdate'][traj] += 1
    return False #not converged
