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
  This module contains the Conjugate Gradient Optimization strategy

  Created on Sept 17, 2019
  @ author: ZHOUJ2
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
from scipy.optimize import minpack2
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .FiniteDifference import FiniteDifference
from utils import mathUtils,randomUtils
#Internal Modules End--------------------------------------------------------------------------------

class ConjugateGradient(FiniteDifference):
  """
    Conjugate Gradient Optimizer
    This class currently inherits from the FiniteDifference (since most of the gradient based machinery is there).
  """
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
    prefix = jobObject.getMetadata()['prefix']
    traj, step, identifier = [int(x) for x in prefix.split('_')]
    self.raiseADebug('Collected sample "{}"'.format(prefix))
    category, number, _, cdId = self._identifierToLabel(identifier)
    done, index = self._checkModelFinish(str(traj), str(step), str(identifier))
    if not done:
      self.raiseAnError(RuntimeError,'Trying to collect "{}" but identifies as not done!'.format(prefix))
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

        if self.counter['task'][traj][:5] == b'START':
          ## since accepted, update history
          try:
            self.counter['recentOptHist'][traj][1] = copy.deepcopy(self.counter['recentOptHist'][traj][0])
          except KeyError:
            # this means we don't have an entry for this trajectory yet, so don't copy anything
            pass
          # store realization of most recent developments
          self.counter['recentOptHist'][traj][0] = optCandidate
          # change xk into array with index matching the index map
          xk = dict((var,self.realizations[traj]['denoised']['opt'][0][var]) for var in self.getOptVars())
          self.counter['xk'][traj] = np.asarray(list(xk.values()))
          self.counter['oldGradK'][traj] = self.counter['gfk'][traj]
          gfk = dict((var,self.localEvaluateGradient(traj,gradHist = True)[var][0]) for var in self.getOptVars())
          self.counter['gfk'][traj] = np.asarray(list(gfk.values()))
          if self.useGradHist and self.counter['oldFVal'][traj]:
            self.counter['oldOldFVal'][traj] = self.counter['oldFVal'][traj]
            self.counter['oldFVal'][traj] = self.realizations[traj]['denoised']['opt'][0][self.objVar]
            self.counter['gNorm'][traj] = self.polakRibierePowellStep(traj,self.counter['lastStepSize'][traj], self.counter['gfk'][traj])
          else:
            self.counter['oldFVal'][traj] = self.realizations[traj]['denoised']['opt'][0][self.objVar]
            self.counter['oldOldFVal'][traj] = self.counter['oldFVal'][traj] + np.linalg.norm(self.counter['gfk'][traj]) / 2
            self.counter['pk'][traj] = -(self.counter['gfk'][traj])
            self.counter['gNorm'][traj] = np.amax(np.abs(self.counter['gfk'][traj]))

          # first step
          self.counter['deltaK'][traj] = np.dot(self.counter['gfk'][traj], self.counter['gfk'][traj])
          self.counter['derPhi0'][traj] = np.dot(self.counter['gfk'][traj], self.counter['pk'][traj])
          self.counter['alpha'][traj] = min(1.0, 1.01*2*(self.counter['oldFVal'][traj] - self.counter['oldOldFVal'][traj])/self.counter['derPhi0'][traj])

          phi1 = self.counter['oldFVal'][traj]
          derPhi1 = self.counter['derPhi0'][traj]
        else:
          newGrad = dict((var,self.localEvaluateGradient(traj, gradHist = True)[var][0]) for var in self.getOptVars())
          newGrad = np.asarray(list(newGrad.values()))
          # after step

          phi1 = self.realizations[traj]['denoised']['opt'][0][self.objVar]
          derPhi1 = np.dot(newGrad, self.counter['pk'][traj])

        self.counter['lastStepSize'][traj], self.counter['newFVal'][traj], _, self.counter['task'][traj] = minpack2.dcsrch(self.counter['alpha'][traj], phi1, derPhi1, ftol=1e-4, gtol=0.4,
                                                xtol=1e-14, task = self.counter['task'][traj], stpmin=1e-100,
                                                stpmax=1e100, isave = self.counter['iSave'][traj] , dsave=self.counter['dSave'][traj])
        # return of the line search can be those results
        # If task = 'FG' then evaluate the function and derivative at step and call dcsrch again
        # If task = 'CONV' then the search is successful. Store this point start a new point
        # If task = 'WARN' then the subroutine is not able to satisfy the convergence conditions. Counted as converged and  resubmit the same point for the presistance
        # If task = 'ERROR' then there is an error in the input arguments

        if self.counter['task'][traj][:2] == b'FG':
          pass # No need to do anything because the value have already been updated
        elif self.counter['task'][traj][:11] == b'CONVERGENCE' :
          self.raiseADebug('Local minimal reached, start new line search')
          self.counter['task'][traj] = b'START' # start a new search

        elif self.counter['task'][traj][:7] == b'WARNING' :
          self.raiseAWarning(self.counter['task'][traj][9:].decode().lower())
          self.counter['persistence'][traj] += 1 # counted as one converged point
          if self.counter['persistence'][traj] >= self.convergencePersistence:
            self.raiseAMessage(' ... Trajectory "{}" converged {} times consecutively!'.format(traj,self.counter['persistence'][traj]))
            self.convergeTraj[traj] = True
            self.removeConvergedTrajectory(traj)
          else:
            self.raiseAMessage(' ... converged Traj "{}" {} times, required persistence is {}.'.format(traj,self.counter['persistence'][traj],self.convergencePersistence))
        else:
          self.raiseAWarning('Not able to calculate the forward step')
          self.counter['persistence'][traj] += 1
          if self.counter['persistence'][traj] >= self.convergencePersistence:
            self.raiseAMessage(' ... Trajectory "{}" converged {} times consecutively!'.format(traj,self.counter['persistence'][traj]))
            self.convergeTraj[traj] = True
            self.removeConvergedTrajectory(traj)
          else:
            self.raiseAMessage(' ... converged Traj "{}" {} times, required persistence is {}.'.format(traj,self.counter['persistence'][traj],self.convergencePersistence))
        self.counter['alpha'][traj] = self.counter['lastStepSize'][traj]
        grad = dict((var,self.counter['gfk'][traj][ind]/(self.optVarsInit['upperBound'][var]-self.optVarsInit['lowerBound'][var])) for ind,var in enumerate(self.getOptVars()))
        try:
          self.counter['gradNormHistory'][traj][1] = self.counter['gradNormHistory'][traj][0]
        except IndexError:
          pass # don't have a history on the first pass
        self.counter['gradNormHistory'][traj][0] = grad

        new = self._newOptPointAdd(grad, traj)
        if new is not None:
          # add new gradient points
          self._createPerturbationPoints(traj, new)
        # reset storage
        self._setupNewStorage(traj)


  ###################
  # Utility Methods #
  ###################
  def _newOptPointAdd(self, gradient, traj):
    """
      This local method add a new opt point based on the gradient
      @ In, gradient, dict, dictionary containing the gradient
      @ In, traj, int, trajectory
      @ Out, varKPlus, dict, new point that has been queued (or None if no new points should be run for this traj)
    """
    stepSize = self.counter['lastStepSize'][traj]
    self.optVarsHist[traj][self.counter['varsUpdate'][traj]] = {}
    varK = dict((var,self.counter['recentOptHist'][traj][0][var]) for var in self.getOptVars())
    varKPlus,modded = self._generateVarsUpdateConstrained(traj, stepSize, gradient, varK)
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
    points = []
    distance = self.paramDict['pertDist'] * max(self.counter['lastStepSize'][traj],self.paramDict['initialStepSize'])

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
    return points

  def polakRibierePowellStep(self, traj,alpha, gfkp1=None):
    """
      Method to update the conjugate direction with steepest direction
      @ In, traj, int, the trajectory we are currently considering
      @ In, alpha, float, step size, or None if no suitable step was found
      @ In, gfkp1, numpy.ndarray, optional, [1, #variables] gradient value for xk (xk being the current parameter estimate).
      @ Out, gNorm, float, norm of the gradient.
    """
    xkp1 = self.counter['xk'][traj]
    yk = gfkp1 - self.counter['oldGradK'][traj]
    betaK = max(0, np.dot(yk, gfkp1) / self.counter['deltaK'][traj])
    self.counter['pk'][traj] = -gfkp1 + betaK * self.counter['pk'][traj]
    gNorm = np.amax(np.abs(gfkp1))
    return gNorm


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
    accepted = self._updateConvergenceVector(traj, self.counter['solutionUpdate'][traj], outputs,conj=True)
    # if converged, we can wrap up this trajectory
    if self.convergeTraj[traj]:
      # end any excess gradient evaluation jobs
      self.cancelJobs([self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],i) for i in self.perturbationIndices])
      return True #converged
    # if not accepted, we need to scrap this run and set up a new one
    if accepted:
      # store acceptance for later
      self.realizations[traj]['accepted'] = accepted
      self.counter['recentOptHist'][traj][1] = self.normalizeData(self.realizations[traj]['denoised']['opt'][0])
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


  def finalizeSampler(self,failedRuns):
    """
      Method called at the end of the Step when no more samples will be taken.  Closes out optimizer.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    FiniteDifference.handleFailedRuns(self,failedRuns)
    # get the most optimal point among the trajectories
    bestValue = None
    bestTraj = None
    for traj in self.counter['recentOptHist'].keys():
      value = self.counter['recentOptHist'][traj][-1][self.objVar]
      self.raiseADebug('For trajectory "{}" the best value was'.format(traj+1),value)
      if bestTraj is None:
        bestTraj = traj
        bestValue = value
        continue
      if self.checkIfBetter(value,bestValue):
        bestTraj = traj
        bestValue = value
    # now have the best trajectory, so write solution export
    bestPoint = self.denormalizeData(self.counter['recentOptHist'][bestTraj][0])
    self.raiseADebug('The best overall trajectory ending was for trajectory "{}".'.format(bestTraj+1))
    self.raiseADebug('    The optimal location is at:')
    for v in self.getOptVars():
      self.raiseADebug('                {} = {}'.format(v,bestPoint[v]))
    self.raiseADebug('    The objective value there: {}'.format(bestValue))
    self.raiseADebug('====================')
    self.raiseADebug('| END OPTIMIZATION |')
    self.raiseADebug('====================')
    # _always_ re-add the last point to the solution export, but use a new varsUpdate value
    overwrite = {'varsUpdate': self.counter['varsUpdate'][traj]}

    self.writeToSolutionExport(bestTraj, self.normalizeData(bestPoint), True, overwrite=overwrite)
