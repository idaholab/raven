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
      # collect finished jobs
      prefix = jobObject.getMetadata()['prefix']
      traj, step, identifier = [int(x) for x in prefix.split('_')] # FIXME This isn't generic for any prefixing system
      self.raiseADebug('Collected sample "{}"'.format(prefix))
      failed = jobObject.getReturnCode() != 0
      if failed:
        self.raiseADebug(' ... sample "{}" FAILED. Cutting step and re-queueing.'.format(prefix))
        # since run failed, cut the step and requeue
        ## cancel any further runs at this point
        self.cancelJobs([self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],i) for i in range(self.perturbationIndices[-1])])
        self.recommendToGain[traj] = 'cut'
        grad = self.counter['gradientHistory'][traj][0]
        new = self._newOptPointAdd(grad, traj)
        if new is not None:
          self._createPerturbationPoints(traj, new)
        self._setupNewStorage(traj)
      else:
        # update self.realizations dictionary for the right trajectory
        # category: is this point an "opt" or a "grad" evaluations?
        # number is which variable is being perturbed, ie which dimention 0 indexed
        category, number, _, cdId = self._identifierToLabel(identifier)
        # done is whether the realization finished
        # index: where is it in the dataObject
        # find index of sample in the target evaluation data object
        done, index = self._checkModelFinish(str(traj), str(step), str(identifier))
        # sanity check
        if not done:
          self.raiseAnError(RuntimeError,'Trying to collect "{}" but identifies as not done!'.format(prefix))
        # store index for future use
        # number is the varID
        number = number + (cdId * len(self.fullOptVars))
        self.realizations[traj]['collect'][category][number].append(index)
        # check if any further action needed because we have all the points we need for opt or grad
        if len(self.realizations[traj]['collect'][category][number]) == self.realizations[traj]['need']:
          # get the output space (input space included as well)
          outputs = self._averageCollectedOutputs(self.realizations[traj]['collect'][category][number])
          # store denoised results
          self.realizations[traj]['denoised'][category][number] = outputs

          # if we just finished "opt", check some acceptance and convergence checking
          if category == 'opt':
            converged = self._finalizeOptimalCandidate(traj,outputs)
          else:
            converged = False
          # if both opts and grads are now done, then we can do an evaluation
          ## note that by now we've ALREADY accepted the point; if it was rejected, it would have been reset by now.
          optDone = bool(len(self.realizations[traj]['denoised']['opt'][0]))
          gradDone = all( len(self.realizations[traj]['denoised']['grad'][i]) for i in range(self.paramDict['pertSingleGrad']))


          if not converged and optDone and gradDone:
            print(self.realizations[traj]['denoised'])


            ###parameter
            gtol=1e-08
            norm= Inf
            eps=1.4901161193847656e-08
            disp=True
            maxiter=2000
            return_all=True
            indexmap = self.getOptVars()
            # change xk into array with index matching the index map
            xk = dict((var,self.realizations[traj]['denoised']['opt'][0][var]) for var in indexmap)
            xk = np.asarray(list(xk.values()))
            gfk = dict((var,self.localEvaluateGradient(traj)[var][0]) for var in indexmap)
            gfk = np.asarray(list(gfk.values()))

            k =0
            old_fval = self.realizations[traj]['denoised']['opt'][0][self.objVar]

            old_old_fval = old_fval + np.linalg.norm(gfk) / 2
            allvecs = [xk]
            pk = -gfk
            gnorm = np.amax(np.abs(gfk))
            sigma_3 = 0.01


            # print('this is a test jz')
            # print('need f')
            # print('need fprime')
            print('xk',xk)
            print('gfk',gfk)
            print('k=',k)
            print('old_fval =',old_fval)
            print('old_old_fval =',old_old_fval)
            # print([gfk])

            while (gnorm > gtol) and (k < maxiter):
              print('inside while')
              deltak = np.dot(gfk, gfk)
              print(deltak)
            # try:
              # alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
              #         _line_search_wolfe12(f, myfprime, xk, pk, gfk, old_fval,
              #                               old_old_fval, c2=0.4, amin=1e-100, amax=1e100,
              #                               extra_condition=descent_condition)
              # extra_condition = self.descentCondition(alpha, xkp1, fp1, gfkp1)
              # alpha_k, fc, gc, old_fval, old_old_fval, gfkp1  = line_search_wolfe1(f, fprime, xk, pk, gfk,
              #                                                   old_fval, old_old_fval,**kwargs)
              # newargs = args
              gradient = True
              gval = [gfk]
              derphi0 = np.dot(gfk, pk)
              # def phi(s):
              #   return f(xk + s*pk)
              # def derphi(s):
              #   gval[0] = fprime(xk + s*pk, *newargs)
              #   return np.dot(gval[0], pk)

              # stp, fval, old_fval = scalar_search_wolfe1(
              #                     phi, derphi, old_fval, old_old_fval, derphi0,
              #                     c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)
              alpha1 = min(1.0, 1.01*2*(old_fval - old_old_fval)/derphi0)
              phi1 = old_fval
              derphi1 = derphi0
              isave = np.zeros((2,), np.intc)
              dsave = np.zeros((13,), float)
              task = b'START'
              maxiter = 10
              print('ok here')
              for i in range(maxiter):
                print(alpha1)
                print(phi1)
                print(derphi1)
                stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1,
                                                  ftol=1e-4, gtol=0.4, xtol=1e-14, task = task,
                                                  stpmin=1e-100, stpmax=1e100, isave = isave , dsave=dsave)
                print('lulueluelueluleuleuleuleuleu',stp, phi1, derphi1, task[:2],dsave)
                i+=1
                # if task[:2] == b'FG':
                #   alpha1 = stp
                #   newx = xk + stp*pk
                #   phi1 = f(newx)
                #   derphi1 = np.dot(fprime(newx), pk)
                # else:
                #   break
              k += 100
              # except:
              #   print("An exception occurred")





          if not converged and optDone and gradDone:
            print('this is not a test jz')
            optCandidate = self.normalizeData(self.realizations[traj]['denoised']['opt'][0])
            print('not converged and optDone and gradDone optdenoised and normaled',self.realizations[traj]['denoised']['opt'][0],optCandidate)
            # update solution export
            ## only write here if we want to write on EVERY optimizer iteration (each new optimal point)
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
            # find the new gradient for this trajectory at the new opt point
            grad = self.evaluateGradient(traj)
            # grad = self.localEvaluateGradient(traj)
            # get a new candidate
            new = self._newOptPointAdd(grad, traj)

            if new is not None:
              # add new gradient points
              self._createPerturbationPoints(traj, new)
            # reset storage
            self._setupNewStorage(traj)

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
    stepSize = self._computeStepSize(self.paramDict, self.counter['varsUpdate'][traj], traj)
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

  def _computeStepSize(self,paramDict,iterNum,traj):
    """
      Utility function to compute the step size
      @ In, paramDict, dict, dictionary containing information to compute gain parameter
      @ In, iterNum, int, current iteration index
      @ Out, new, float, current value for gain ak
    """
    #TODO FIXME is this a good idea?

    print('jz is trying the stepsize')

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
        print('jzjzjzlueluekue',self.inputInfo)
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
    print('using this newnewnew')
    if denorm:
      point = self.denormalizeData(point)
    return prefix,point




  def polakRibierePowellStep(self, alpha, gfkp1=None):
    print('polak_ribiere_powell_step')
    xkp1 = xk + alpha * pk
    if gfkp1 is None:
      # gfkp1 = myfprime(xkp1)
      gfkp1 = approx_fprime(xkp1, f, epsilon)
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
