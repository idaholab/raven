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
  This module contains the Gradient Based Optimization strategy

  Created on June 16, 2016
  @author: chenj
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
import abc
import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Optimizer import Optimizer
from Assembler import Assembler
from utils import utils,cached_ndarray,mathUtils
#Internal Modules End--------------------------------------------------------------------------------

class GradientBasedOptimizer(Optimizer):
  """
    This is the base class for gradient based optimizer. The following methods need to be overridden by all derived class
    self.localLocalInputAndChecks(self, xmlNode)
    self.localLocalInitialize(self, solutionExport)
    self.localLocalGenerateInput(self,model,oldInput)
    self.localEvaluateGradient(self, optVarsValues, gradient = None)
  """
  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Optimizer.__init__(self)
    self.ableToHandelFailedRuns      = True            # is this optimizer able to handle failed runs?
    self.constraintHandlingPara      = {}              # Dict containing parameters for parameters related to constraints handling
    self.gradientNormTolerance       = 1.e-3           # tolerance on the L2 norm of the gradient
    self.gradDict                    = {}              # Dict containing information for gradient related operations
    self.gradDict['numIterForAve'  ] = 1               # Number of iterations for gradient estimation averaging
    self.gradDict['pertNeeded'     ] = 1               # Number of perturbation needed to evaluate gradient (globally, considering denoising)
    self.paramDict['pertSingleGrad'] = 1               # Number of perturbation needed to evaluate a single gradient
    self.gradDict['pertPoints'     ] = {}              # Dict containing normalized inputs sent to model for gradient evaluation
    self.readyVarsUpdate             = {}              # Bool variable indicating the finish of gradient evaluation and the ready to update decision variables
    self.counter['perturbation'    ] = {}              # Counter for the perturbation performed.
    self.counter['gradientHistory' ] = {}              # In this dict we store the gradient value (versor) for current and previous iterations {'trajectoryID':[{},{}]}
    self.counter['gradNormHistory' ] = {}              # In this dict we store the gradient norm for current and previous iterations {'trajectoryID':[float,float]}
    self.counter['varsUpdate'      ] = {}
    self.counter['solutionUpdate'  ] = {}
    self.counter['lastStepSize'    ] = {}              # counter to track the last step size taken, by trajectory
    self.convergeTraj                = {}
    self.convergenceProgress         = {}              #tracks the convergence progress, by trajectory
    self.trajectoriesKilled          = {}              # by traj, store traj killed, so that there's no mutual destruction
    self.recommendToGain             = {}              # recommended action to take in next step, by trajectory
    self.gainGrowthFactor            = 2.              # max step growth factor
    self.gainShrinkFactor            = 2.              # max step shrinking factor
    self.perturbationIndices         = []              # in this list we store the indeces that correspond to the perturbation. It is not ideal but it is quick and dirty now

    # register metadata
    self.addMetaKeys('trajID','varsUpdate','prefix')

  def localInputAndChecks(self, xmlNode):
    """
      Method to read the portion of the xml input that belongs to all gradient based optimizer only
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    convergence = xmlNode.find("convergence")
    if convergence is not None:
      #convergence criteria, the gradient threshold
      gradientThreshold = convergence.find("gradientThreshold")
      try:
        self.gradientNormTolerance = float(gradientThreshold.text) if gradientThreshold is not None else self.gradientNormTolerance
      except ValueError:
        self.raiseAnError(ValueError, 'Not able to convert <gradientThreshold> into a float.')
      #grain growth factor, the multiplier for going in the same direction
      gainGrowthFactor = convergence.find('gainGrowthFactor')
      try:
        self.gainGrowthFactor = float(gainGrowthFactor.text) if gainGrowthFactor is not None else 2.0
      except ValueError:
        self.raiseAnError(ValueError, 'Not able to convert <gainGrowthFactor> into a float.')
      #grain shrink factor, the multiplier for going in the opposite direction
      gainShrinkFactor = convergence.find('gainShrinkFactor')
      try:
        self.gainShrinkFactor = float(gainShrinkFactor.text) if gainShrinkFactor is not None else 2.0
      except ValueError:
        self.raiseAnError(ValueError, 'Not able to convert <gainShrinkFactor> into a float.')
      self.raiseADebug('Gain growth factor is set at',self.gainGrowthFactor)
      self.raiseADebug('Gain shrink factor is set at',self.gainShrinkFactor)
    self.gradDict['numIterForAve'] = int(self.paramDict.get('numGradAvgIterations', 1))

  def localInitialize(self,solutionExport):
    """
      Method to initialize settings that belongs to all gradient based optimizer
      @ In, solutionExport, DataObject, a PointSet to hold the solution
      @ Out, None
    """
    for traj in self.optTraj:
      self.gradDict['pertPoints'][traj]      = {}
      self.counter['perturbation'][traj]     = 0
      self.counter['varsUpdate'][traj]       = 0
      self.counter['solutionUpdate'][traj]   = 0
      self.counter['gradientHistory'][traj]  = [{},{}]
      self.counter['gradNormHistory'][traj]  = [0.0,0.0]
      self.counter['persistence'][traj]      = 0
      self.optVarsHist[traj]                 = {}
      self.readyVarsUpdate[traj]             = False
      self.convergeTraj[traj]                = False
      self.status[traj]                      = {'process':'submitting new opt points', 'reason':'just started'}
      self.counter['recentOptHist'][traj]    = [{},{}]
      self.trajectoriesKilled[traj]          = []
    # end job runnable equal to number of trajectory
    self._endJobRunnable = len(self.optTraj)
    # initialize index lists
    ## opt point evaluations are indices 0 through number of re-evaluation points
    self.optPointIndices = list(range(0,self.gradDict['numIterForAve']+1))
    ## perturbation evaluations are indices starting at the end of optPoint and going through all the rest
    self.perturbationIndices = list(range(self.gradDict['numIterForAve'],self.gradDict['numIterForAve']*(self.paramDict['pertSingleGrad']+1)))
    #specializing the self.localLocalInitialize()
    self.localLocalInitialize(solutionExport=solutionExport)

  @abc.abstractmethod
  def localLocalInitialize(self, solutionExport):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, a PointSet to hold the solution
      @ Out, None
    """
    pass

  def localStillReady(self,ready, convergence = False): #,lastOutput=None
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, boolean variable indicating whether the caller is prepared for another input.
    """
    #let this be handled at the local subclass level for now
    return ready

  def _checkModelFinish(self, traj, updateKey, evalID):
    """
      Determines if the Model has finished running an input and returned the output
      @ In, traj, int, traj on which the input is being checked
      @ In, updateKey, int, the id of variable update on which the input is being checked
      @ In, evalID, int or string, indicating the id of the perturbation (int) or its a variable update (string 'v')
      @ Out, _checkModelFinish, tuple(bool, int), (1,realization dictionary),
            (indicating whether the Model has finished the evaluation over input identified by traj+updateKey+evalID, the index of the location of the input in dataobject)
    """
    if len(self.mdlEvalHist) == 0:
      return (False,-1)
    lookFor = '{}_{}_{}'.format(traj,updateKey,evalID)
    index,match = self.mdlEvalHist.realization(matchDict = {'prefix':lookFor})
    # if no match, return False
    if match is None:
      return False,-1
    # otherwise, return index of match
    return True, index

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    self.readyVarsUpdate = {traj:False for traj in self.optTrajLive}

  def calculateMultivectorMagnitude(self,values):
    """
      Calculates the magnitude of vector "values", where values might be a combination of scalars and vectors (but not matrices [yet]).
      Calculates the magnitude as if "values" were flattened into a 1d array.
      @ In, values, list, values for which the magnitude will be calculated
      @ Out, mag, float, magnitude
    """
    # use np.linalg.norm (Frobenius norm) to calculate magnitude
    ## pre-normalise vectors, this is mathematically equivalent to flattening the vector first
    ## NOTE this assumes gradient vectors are 0 or 1 dimensional, not 2 or more! (vectors or scalars, not matrices)
    # TODO this could be sped up if we could avoid calling np.atleast_1d twice, but net slower if we loop first
    preMag = [np.linalg.norm(val) if len(np.atleast_1d(val))>1 else np.atleast_1d(val)[0] for val in values]
    ## then get the magnitude of the result, and return it
    return np.linalg.norm(preMag)

  def evaluateGradient(self, optVarsValues, traj):
    """
      Method to evaluate gradient based on perturbed points and model evaluations.
      @ In, optVarsValues, dict, dictionary containing perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ In, traj, int, the trajectory id
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    # store old values
    self.counter['gradientHistory'][traj][1] = copy.copy(self.counter['gradientHistory'][traj][0])
    self.counter['gradNormHistory'][traj][1] = copy.copy(self.counter['gradNormHistory'][traj][0])
    gradient = None # for now...most of the stuff in the localEvaluate can be performed here
    gradient = self.localEvaluateGradient(optVarsValues, traj, gradient)
    # we intend for gradient to give direction only, so get the versor
    ## NOTE this assumes gradient vectors are 0 or 1 dimensional, not 2 or more! (vectors or scalars, not matrices)
    gradientNorm = self.calculateMultivectorMagnitude(gradient.values())
    # store this norm, infinite or not
    self.counter['gradNormHistory'][traj][0] = gradientNorm
    #fix inf
    if gradientNorm == np.inf:
      # if there are infinites, then only infinites should remain, and they are +-1
      for v,var in enumerate(gradient.keys()):
        # first, set all non-infinites to 0, since they can't compete with infinites
        gradient[var][-np.inf < gradient[var] < np.inf] =  0.0
        # set +- infinites to +- 1 (arbitrary) since they're all equally important
        gradient[var][gradient[var] == -np.inf] = -1.0
        gradient[var][gradient[var] ==  np.inf] =  1.0
      # set up the new grad norm
      gradientNorm = self.calculateMultivectorMagnitude(gradient.values())
    # normalize gradient (if norm is zero, skip this)
    if gradientNorm != 0.0:
      for var in gradient.keys():
        gradient[var] = gradient[var]/gradientNorm
        # if float coming in, make it a float going out
        if len(gradient[var])==1:
          gradient[var] = float(gradient[var])
    # store gradient
    self.counter['gradientHistory'][traj][0] = gradient
    return gradient

  def _createEvaluationIdentifier(self,trajID,iterID,evalType):
    """
      Create evaluation identifier
      @ In, trajID, integer, trajectory identifier
      @ In, iterID, integer, iteration number (identifier)
      @ In, evalType, integer or string, evaluation type (v for variable update; otherwise id for gradient evaluation)
      @ Out, identifier, string, the evaluation identifier
    """
    identifier = str(trajID) + '_' + str(iterID) + '_' + str(evalType)
    return identifier

  def getPreviousIdentifierGivenCurrent(self,prefix):
    """
      Method to get the previous identifier given the current prefix
      @ In, prefix, str, the current identifier
      @ Out, previousPrefix, str, the previous identifier
    """

    traj, _, _ = prefix.split("_")
    traj       = int(traj)
    return self.counter['prefixHistory'][traj][-1]

  def localEvaluateGradient(self, optVarsValues, gradient = None):
    """
      Local method to evaluate gradient.
      @ In, optVarsValues, dict, dictionary containing perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ In, gradient, dict, optional, dictionary containing gradient estimation by the caller.
                                      gradient should have the form {varName: gradEstimation}
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    return gradient

  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, None
      @ Out, convergence, list, list of bool variable indicating whether the convergence criteria has been met for each trajectory.
    """
    convergence = True
    for traj in self.optTraj:
      if self.convergeTraj[traj] == False:
        convergence = False
        break
    return convergence

  def _updateConvergenceVector(self, traj, varsUpdate, currentLossVal):
    """
      Local method to update convergence vector.
      @ In, traj, int, identifier of the trajector to update
      @ In, varsUpdate, int, current variables update iteration number
      @ In, currentLossVal, float, current loss function value
      @ Out, None
    """
    # first, check if we're at varsUpdate 0 (first entry); if so, we are at our first point
    if varsUpdate == 0:
      # we don't have enough points to decide to accept or reject the new point, so accept it as the initial point
      self.raiseADebug('Accepting first point, since we have no rejection criteria.')
      self.status[traj]['reason'] = 'found new opt point'
      return

    #otherwise, we need to accept/reject point and check convergence
    currentInputDenorm = self.denormalizeData(self.optVarsHist[traj][self.counter['varsUpdate'][traj]])

    ## first, determine if we want to keep the new point
    # obtain the old loss value
    oldLossVal = self.counter['recentOptHist'][traj][0][self.objVar]
    # see if new point is better than old point
    newerIsBetter = self.checkIfBetter(currentLossVal,oldLossVal)
    # if this was a recommended preconditioning point, we should not be converged.
    pointFromRecommendation = self.status[traj]['reason'] == 'received recommended point'
    # if improved, keep it and move forward; otherwise, reject it and recommend cutting step size
    if newerIsBetter:
      self.status[traj]['reason'] = 'found new opt point'
      self.raiseADebug('Accepting potential opt point for improved loss value.  Diff: {}, New: {}, Old: {}'.format(abs(currentLossVal-oldLossVal),currentLossVal,oldLossVal))
      #TODO REWORK this belongs in the base class optimizer; grad shouldn't know about multilevel!!
      #  -> this parameter is how multilevel knows that a successful perturbation of an outer loop has been performed
      #  maybe implement a "acceptPoint" method in base class?
      self.mlActiveSpaceSteps[traj] += 1
    else:
      self.status[traj]['reason'] = 'rejecting bad opt point'
      self.raiseADebug('Rejecting potential opt point for worse loss value. old: "{}", new: "{}"'.format(oldLossVal,currentLossVal))
      # cut the next step size to hopefully stay in the valley instead of climb up the other side
      self.recommendToGain[traj] = 'cut'

    ## determine convergence
    if pointFromRecommendation:
      self.raiseAMessage('Setting convergence for Trajectory "{}" to "False" because of preconditioning.'.format(traj))
      converged = False
    else:
      self.raiseAMessage('Checking convergence for Trajectory "{}":'.format(traj))
      self.convergenceProgress[traj] = {} # tracks progress for grad norm, abs, rel tolerances
      converged = False                   # updated for each individual criterion using "or" (pass one, pass all)
      #printing utility
      printString = '    {:<21}: {:<5}'
      printVals = printString + ' (check: {:>+9.2e} < {:>+9.2e}, diff: {:>9.2e})'
      def printProgress(name,boolCheck,test,gold):
        """
          Consolidates a commonly-used print statement to prevent errors and improve readability.
          @ In, name, str, printed name of convergence check
          @ In, boolCheck, bool, boolean convergence results for this check
          @ In, test, float, value of check at current opt point
          @ In, gold, float, convergence threshold value
          @ Out, None
        """
        self.raiseAMessage(printVals.format(name,str(boolCheck),test,gold,abs(test-gold)))

      # "min step size" and "gradient norm" are both always valid checks, whether rejecting or accepting new point

      # min step size check
      try:
        lastStep = self.counter['lastStepSize'][traj]
        minStepSizeCheck = lastStep <= self.minStepSize
      except KeyError:
        #we reset the step size, so we don't have a value anymore
        lastStep = np.nan
        minStepSizeCheck = False
      printProgress('Min step size',minStepSizeCheck,lastStep,self.minStepSize)
      converged = converged or minStepSizeCheck

      # gradient norm
      if len(self.counter['gradientHistory'][traj][0]) > 0:
        gradNorm = self.counter['gradNormHistory'][traj][0]
        self.convergenceProgress[traj]['grad'] = gradNorm
        gradientNormCheck = gradNorm <= self.gradientNormTolerance
      else:
        gradNorm = np.nan
        gradientNormCheck = False
      printProgress('Gradient magnitude',gradientNormCheck,gradNorm,self.gradientNormTolerance)
      converged = converged or gradientNormCheck

      # if accepting new point, then "same coordinate" and "abs" and "rel" checks are also valid reasons to converge
      if newerIsBetter:
        #absolute tolerance
        absLossDiff = abs(mathUtils.diffWithInfinites(currentLossVal,oldLossVal))
        self.convergenceProgress[traj]['abs'] = absLossDiff
        absTolCheck = absLossDiff <= self.absConvergenceTol
        printProgress('Absolute Loss Diff',absTolCheck,absLossDiff,self.absConvergenceTol)
        converged = converged or absTolCheck

        #relative tolerance
        relLossDiff = mathUtils.relativeDiff(currentLossVal,oldLossVal)
        self.convergenceProgress[traj]['rel'] = relLossDiff
        relTolCheck = relLossDiff <= self.relConvergenceTol
        printProgress('Relative Loss Diff',relTolCheck,relLossDiff,self.relConvergenceTol)
        converged = converged or relTolCheck

        #same coordinate check
        sameCoordinateCheck = True
        for var,values in self.optVarsHist[traj][varsUpdate].items():
          # don't check constants, of course they're the same
          if var in self.constants:
            continue
          # differentiate vectors and scalars for checking
          if hasattr(values,'__len__'):
            if any(values != self.counter['recentOptHist'][traj][0][var]):
              sameCoordinateCheck = False
              break
          else:
            if values != self.counter['recentOptHist'][traj][0][var]:
              sameCoordinateCheck = False
              break
        self.raiseAMessage(printString.format('Same coordinate check',str(sameCoordinateCheck)))
        converged = converged or sameCoordinateCheck

    if converged:
      # update number of successful convergences
      self.counter['persistence'][traj] += 1
      # check if we've met persistence requirement; if not, keep going
      if self.counter['persistence'][traj] >= self.convergencePersistence:
        self.raiseAMessage(' ... Trajectory "{}" converged {} times consecutively!'.format(traj,self.counter['persistence'][traj]))
        self.convergeTraj[traj] = True
        self.removeConvergedTrajectory(traj)
      else:
        self.raiseAMessage(' ... converged Traj "{}" {} times, required persistence is {}.'.format(traj,self.counter['persistence'][traj],self.convergencePersistence))
    else:
      self.counter['persistence'][traj] = 0
      self.raiseAMessage(' ... continuing trajectory "{}".'.format(traj))

  def _removeRedundantTraj(self, trajToRemove, currentInput):
    """
      Local method to remove multiple trajectory
      @ In, trajToRemove, int, identifier of the trajector to remove
      @ In, currentInput, dict, the last variable on trajectory traj
      @ Out, None
    """
    # TODO replace this with a kdtree search
    removeFlag = False
    def getRemoved(trajThatSurvived, fullList=None):
      """
        Collect list of all the trajectories removed by this one, or removed by trajectories removed by this one, and etc
        @ In, trajThatSurvived, int, surviving trajectory that has potentially removed others
        @ In, fullList, list, optional, if included is the partial list to add to
        @ Out, fullList, list, list of all traj removed (explicitly or implicitly) by this one
      """
      if fullList is None:
        fullList = []
      removed = self.trajectoriesKilled[trajThatSurvived]
      fullList += removed
      for rm in removed:
        fullList = getRemoved(rm, fullList)
      return fullList
    #end function definition
    notEligibleToRemove = [trajToRemove] + getRemoved(trajToRemove)
    for traj in self.optTraj:
      #don't consider removal if comparing against itself,
      #  or a trajectory removed by this one, or a trajectory removed by a trajectory removed by this one (recursive)
      #  -> this prevents mutual destruction cases
      if traj not in notEligibleToRemove: #[trajToRemove] + self.trajectoriesKilled[trajToRemove]:
        #FIXME this can be quite an expensive operation, looping through each other trajectory
        for updateKey in self.optVarsHist[traj].keys():
          inp = copy.deepcopy(self.optVarsHist[traj][updateKey]) #FIXME deepcopy needed?
          if len(inp) < 1: #empty
            continue
          removeLocalFlag = True
          dist = self.calculateMultivectorMagnitude( [inp[var] - currentInput[var] for var in self.getOptVars(traj=trajToRemove)] )
          if dist < self.thresholdTrajRemoval:
            self.raiseADebug('Halting trajectory "{}" because it is following trajectory "{}"'.format(trajToRemove,traj))
            self.trajectoriesKilled[traj].append(trajToRemove)
            #TODO the trajectory to remove should be chosen more carefully someday, for example, the one that has the smallest steps or lower loss value currently
            removeFlag = True
            break
        if removeFlag:
          break

    if removeFlag:
      for trajInd, tr in enumerate(self.optTrajLive):
        if tr == trajToRemove:
          self.optTrajLive.pop(trajInd)
          self.status[trajToRemove] = {'process':'following traj '+str(traj),'reason':'removed as redundant'}
          break

  def localCheckConstraint(self, optVars, satisfaction = True):
    """
      Local method to check whether a set of decision variables satisfy the constraint or not
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ In, satisfaction, bool, optional, variable indicating how the caller determines the constraint satisfaction at the point optVars
      @ Out, satisfaction, bool, variable indicating the satisfaction of constraints at the point optVars
    """
    return satisfaction

  def _getJobsByID(self,traj):
    """
      Overwrite only if you need something special at the end of each run....
      This function is used by optimizers that need to collect information from the just ended run
      @ In, traj, int, ID of the trajectory for whom we collect jobs
      @ Out, solutionExportUpdatedFlag, bool, True if the solutionExport needs updating
      @ Out, solutionIndeces, list(int), location of updates within the full targetEvaluation data object
    """
    solutionUpdateList = []
    solutionIndeces = []
    # get all the opt point results (these are the multiple evaluations of the opt point)
    for i in range(self.gradDict['numIterForAve']):
      identifier = i
      solutionExportUpdatedFlag, index = self._checkModelFinish(traj, self.counter['solutionUpdate'][traj], str(identifier))
      solutionUpdateList.append(solutionExportUpdatedFlag)
      solutionIndeces.append(index)
    solutionExportUpdatedFlag = all(solutionUpdateList)
    return solutionExportUpdatedFlag,solutionIndeces

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """
      Overwrite only if you need something special at the end of each run....
      This function is used by optimizers that need to collect information from the just ended run
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
    """
    # for some reason, Ensemble Model doesn't preserve this information, so wrap this debug in a try:
    prefix = jobObject.getMetadata()['prefix']
    failed = jobObject.getReturnCode() != 0
    failedTrajectory = - 1
    if not failed:
      self.raiseADebug('Collected sample "{}"'.format(prefix))
    else:
      # failed trajectory
      failedTrajectory = int(prefix.split("_")[0])
    # TODO REWORK move this whole piece to Optimizer base class as much as possible
    if len(self.mdlEvalHist) != 0:
      for traj in self.optTraj:
        failedTraj = traj == failedTrajectory
        if self.counter['solutionUpdate'][traj] <= self.counter['varsUpdate'][traj]:
          # check whether solution export needs updating, and get indices of entries that need to be added
          solutionExportUpdatedFlag, indices = self._getJobsByID(traj)
          if solutionExportUpdatedFlag or failedTraj:
            #get evaluations (input,output) from the collection of all evaluations
            if not failedTraj:
              #TODO this might be faster for non-stochastic if we do an "if" here on gradDict['numIterForAve']
              #make a place to store distinct evaluation values
              outputs = dict((var,np.zeros(self.gradDict['numIterForAve'],dtype=object))
                  for var in self.solutionExport.getVars('output')
                  if var in self.mdlEvalHist.getVars('output'))
              # get output values corresponding to evaluations of the opt point
              # also add opt points to the grad perturbation list
              self.gradDict['pertPoints'][traj] = np.zeros((1+self.paramDict['pertSingleGrad'])*self.gradDict['numIterForAve'],dtype=dict)
              for i, index in enumerate(indices):
                # get the realization from the targetEvaluation
                vals = self.mdlEvalHist.realization(index=index)
                # place values TODO this could be vectorized significantly!
                for var in outputs.keys():
                  if hasattr(vals[var],'__len__') and len(vals[var]) == 1:
                    outputs[var][i] = float(vals[var])
                  else:
                    outputs[var][i] = vals[var]
                  if var == self.objVar:
                    self.gradDict['pertPoints'][traj][i] = {'inputs':self.normalizeData(dict((var,vals[var]) for var in self.mdlEvalHist.getVars('input'))),
                                                            'output':outputs[var][i]}
              # assumed output value is the mean of sampled values
              for var,vals in outputs.items():
                outputs[var] = vals.mean()
              currentObjectiveValue = outputs[self.objVar]#.mean()
              # check convergence
              # TODO REWORK move this to localStillReady, along with the gradient evaluation
              self._updateConvergenceVector(traj, self.counter['solutionUpdate'][traj], currentObjectiveValue)
            else:
              self.raiseAMessage('Rejecting opt point for trajectory "'+str(failedTrajectory)+'" since the model failed!')
              self.convergeTraj[traj]     = False
              self.status[traj]['reason'] =  'failed run'
              self.recommendToGain[traj]  = 'cut'
            if self.convergeTraj[traj]:
              self.status[traj] = {'process':None, 'reason':'converged'}
            else:
              # update status to submitting grad eval points
              if failedTraj:
                self.status[traj]['process'] = 'submitting new opt points'
              else:
                self.status[traj]['process'] = 'submitting grad eval points'
            # if rejecting bad point, keep the old point as the new point; otherwise, add the new one
            if self.status[traj]['reason'] not in  ['rejecting bad opt point','failed run']:
              try:
                self.counter['recentOptHist'][traj][1] = copy.deepcopy(self.counter['recentOptHist'][traj][0])
              except KeyError:
                # this means we don't have an entry for this trajectory yet, so don't copy anything
                pass
              # store realization of most recent developments
              rlz = {}
              rlz.update(self.optVarsHist[traj][self.counter['varsUpdate'][traj]])
              rlz.update(outputs)
              self.counter['recentOptHist'][traj][0] = rlz
              if traj not in self.counter['prefixHistory']:
                self.counter['prefixHistory'][traj] = []
              self.counter['prefixHistory'][traj].append(prefix)
            # update solution export
            #FIXME much of this should move to the base class!
            if not failedTraj:
              # only write here if we want to write on EVERY optimizer iteration (each new optimal point)
              if self.writeSolnExportOn == 'every':
                self.writeToSolutionExport(traj)
              # whether we wrote to solution export or not, update the counter
              self.counter['solutionUpdate'][traj] += 1
          else: #not ready to update solutionExport
            break

  def writeToSolutionExport(self,traj):
    """
      Standardizes how the solution export is written to.
      Uses data from "recentOptHist" and other counters to fill in values.
      @ In, traj, int, the trajectory for which an entry is being written
      @ Out, None
    """
    # create realization to add to data object
    rlz = {}
    badValue = -1 #value to use if we don't have a value # TODO make this accessible to user?
    recent = self.counter['recentOptHist'][traj][0]
    for var in self.solutionExport.getVars():
      # if this variable has indices, add them to the realization
      indexes = self.solutionExport.getDimensions(var)[var]
      if len(indexes):
        # use the prefix to find the right realization
        ## NOTE there will be a problem with unsynchronized histories!
        varUpdate = self.counter['solutionUpdate'][traj]
        # negative values wouldn't make sense
        varUpdate = max(0,varUpdate-1)
        prefix = '{}_{}_{}'.format(traj,varUpdate,0)
        _,match = self.mdlEvalHist.realization(matchDict = {'prefix':prefix})
        for index in indexes:
          rlz[index] = match[index]
      # CASE: what variable is asked for:
      # inputs, objVar, other outputs
      if var in recent.keys():
        new = self.denormalizeData(recent)[var]
      elif var in self.constants:
        new = self.constants[var]
      # custom counters: varsUpdate, trajID, stepSize
      elif var == 'varsUpdate':
        new = self.counter['solutionUpdate'][traj]
      elif var == 'trajID':
        new = traj+1 # +1 is for historical reasons, when histories were indexed on 1 instead of 0
      elif var == 'stepSize':
        try:
          new = self.counter['lastStepSize'][traj]
        except KeyError:
          new = badValue
      # variable-dependent information: gradients
      elif var.startswith( 'gradient_'):
        varName = var[9:]
        vec = self.counter['gradientHistory'][traj][0].get(varName,None)
        if vec is not None:
          new = vec*self.counter['gradNormHistory'][traj][0]
        else:
          new = badValue
      # convergence metrics
      elif var.startswith( 'convergenceAbs'):
        try:
          new = self.convergenceProgress[traj].get('abs',badValue)
        except KeyError:
          new = badValue
      elif var.startswith( 'convergenceRel'):
        try:
          new = self.convergenceProgress[traj].get('rel',badValue)
        except KeyError:
          new = badValue
      elif var.startswith( 'convergenceGrad'):
        try:
          new = self.convergenceProgress[traj].get('grad',badValue)
        except KeyError:
          new = badValue
      else:
        self.raiseAnError(IOError,'Unrecognized output request:',var)
      # format for realization
      rlz[var] = np.atleast_1d(new)
    self.solutionExport.addRealization(rlz)

  def fractionalStepChangeFromGradHistory(self,traj):
    """
      Uses the dot product between two successive gradients to determine a fractional multiplier for the step size.
      For instance, if the dot product is 1.0, we're consistently moving in a straight line, so increase step size.
      If the dot product is -1.0, we've gone forward and then backward again, so cut the step size down before moving again.
      If the dot product is 0.0, we're moving orthogonally, so don't change step size just yet.
      @ In, traj, int, the trajectory for whom we are creating a fractional step size
      @ Out, frac, float, the fraction by which to multiply the existing step size
    """
    # if we have a recommendation from elsewhere, take that first
    if traj in self.recommendToGain.keys():
      recommend = self.recommendToGain.pop(traj)
      if recommend == 'cut':
        frac = 1./self.gainShrinkFactor
      elif recommend == 'grow':
        frac = self.gainGrowthFactor
      else:
        self.raiseAnError(RuntimeError,'unrecognized gain recommendation:',recommend)
      self.raiseADebug('Based on recommendation, step size multiplier is:',frac)
      return frac
    # otherwise, no recommendation for this trajectory, so move on
    #if we don't have two evaluated gradients, just return 1.0
    grad1 = self.counter['gradientHistory'][traj][1]
    if len(grad1) == 0: # aka if grad1 is empty dict
      return 1.0
    #otherwise, do the dot product between the last two gradients
    grad0 = self.counter['gradientHistory'][traj][0]
    # scalar product
    ## NOTE assumes scalar or vector, not matrix, values
    prod = np.sum( [np.sum(grad0[key]*grad1[key]) for key in grad0.keys()] )
    #rescale from [-1, 1] to [1/g, g]
    if prod > 0:
      frac = self.gainGrowthFactor**prod
    else:
      frac = self.gainShrinkFactor**prod
    self.raiseADebug('Based on gradient history, step size multiplier is:',frac)
    return frac

  def queueUpOptPointRuns(self,traj,point):
    """
      Establishes a queue of runs, all on the point currently stored in "point", to satisfy stochastic denoising.
      @ In, traj, int, the trajectory who needs the queue
      @ In, point, dict, input space as {var:val} NORMALIZED
      @ Out, None
    """
    # TODO sanity check, this could be removed for efficiency later
    if len(self.submissionQueue[traj]) > 0:
      self.raiseAnError(RuntimeError,'Preparing to add opt evals to submission queue for trajectory "{}" but it is not empty: "{}"'.format(traj,self.submissionQueue[traj]))
    for i in range(self.gradDict['numIterForAve']):
      #entries into the queue are as {'inputs':{var:val}, 'prefix':runid} where runid is <traj>_<varUpdate>_<evalNumber> as 0_0_2
      nPoint = {'inputs':copy.deepcopy(point)} #deepcopy to prevent simultaneous alteration
      nPoint['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],i) # from 0 to self.gradDict['numIterForAve'] are opt point evals
      self.submissionQueue[traj].append(nPoint)

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
    if denorm:
      point = self.denormalizeData(point)
    return prefix,point

  def proposeNewPoint(self,traj,point):
    """
      See base class.  Used to set next recommended point to use for algorithm, overriding the gradient descent.
      @ In, traj, int, trajectory who gets proposed point
      @ In, point, dict, input space as dictionary {var:val}
      @ Out, None
    """
    Optimizer.proposeNewPoint(self,traj,point)
    self.counter['varsUpdate'][traj] += 1 #usually done when evaluating gradient, but we're bypassing that
    self.queueUpOptPointRuns(traj,self.recommendedOptPoint[traj])

  def finalizeSampler(self,failedRuns):
    """
      Method called at the end of the Step when no more samples will be taken.  Closes out optimizer.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    Optimizer.handleFailedRuns(self,failedRuns)
    # if writing soln export only on final, now is the time to do it
    if self.writeSolnExportOn == 'final':
      # get the most optimal point among the trajectories
      bestValue = None
      bestTraj = None
      for traj in self.counter['recentOptHist'].keys():
        value = self.counter['recentOptHist'][traj][0][self.objVar]
        self.raiseADebug('For trajectory "{}" the best value was'.format(traj),value)
        if bestTraj is None:
          bestTraj = traj
          bestValue = value
          continue
        if self.checkIfBetter(value,bestValue):
          bestTraj = traj
          bestValue = value
      # now have the best trajectory, so write solution export
      self.raiseADebug('The best overall trajectory ending was for trajectory "{}".'.format(bestTraj))
      self.writeToSolutionExport(bestTraj)
