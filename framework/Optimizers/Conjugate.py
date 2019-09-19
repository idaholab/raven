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
      if var in gradient:
        gradient[var] += np.atleast_1d(lossDiff / dh)
      else:
        gradient[var] = np.atleast_1d(lossDiff / dh)
    print('jz is inside local gradient',gradient)
    return gradient


  def localFinalizeActualSampling(self,jobObject,model,myInput):
      """
        Overwrite only if you need something special at the end of each run....
        This function is used by optimizers that need to collect information from the just ended run
        @ In, jobObject, instance, an instance of a Runner
        @ In, model, Model, instance of a RAVEN model
        @ In, myInput, list, the generating input
        @ Out, None
      """
      #print(self.counter)
      # {'mdlEval': 3, 'varsUpdate': [1], 'recentOptHist': {0: [{'ans': 17.071067811865476, 'x': 0.75, 'y': 0.25}, {}]}, 'persistence': {0: 0}, 'perturbation': {0: 0}, 'gradientHistory': {0: [{'x': 0.9575438204512511, 'y': -0.28828775887231545}, {}]}, 'gradNormHistory': {0: [0.001414213562373151, 0.0]}, 'solutionUpdate': {0: 1}, 'lastStepSize': {0: 0.070710678118654766}}

      #
      # print(jobObject.getMetadata())
      # {'SampledVars': {'x': 0.36458285425512926, 'y': -0.45922995415366263}, 'SampledVarsPb': {}, 'crowDist': {}, 'prefix': '0_1_0', 'trajID': 1, 'varsUpdate': 1}
      print('samesamesame')
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

            optCandidate = self.normalizeData(self.realizations[traj]['denoised']['opt'][0])
            print('hehehehehehehehe',self.realizations[traj]['denoised']['opt'][0],optCandidate)
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