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
    self.localLocalInitialize(self, solutionExport = None)
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
    self.constraintHandlingPara     = {}              # Dict containing parameters for parameters related to constraints handling
    self.gradientNormTolerance      = 1.e-3           # tolerance on the L2 norm of the gradient
    self.gradDict                   = {}              # Dict containing information for gradient related operations
    self.gradDict['numIterForAve' ] = 1               # Number of iterations for gradient estimation averaging
    self.gradDict['pertNeeded'    ] = 1               # Number of perturbation needed to evaluate gradient
    self.gradDict['pertPoints'    ] = {}              # Dict containing normalized inputs sent to model for gradient evaluation
    self.readyVarsUpdate            = {}              # Bool variable indicating the finish of gradient evaluation and the ready to update decision variables
    self.counter['perturbation'   ] = {}              # Counter for the perturbation performed.
    self.counter['gradientHistory'] = {}              # In this dict we store the gradient value (versor) for current and previous iterations {'trajectoryID':[{},{}]}
    self.counter['gradNormHistory'] = {}              # In this dict we store the gradient norm for current and previous iterations {'trajectoryID':[float,float]}
    self.counter['varsUpdate'     ] = {}
    self.counter['solutionUpdate' ] = {}
    self.counter['lastStepSize'   ] = {}              # counter to track the last step size taken, by trajectory
    self.convergeTraj = {}
    self.convergenceProgress        = {}              #tracks the convergence progress, by trajectory
    self.trajectoriesKilled         = {}              # by traj, store traj killed, so that there's no mutual destruction
    self.recommendToGain            = {}              # recommended action to take in next step, by trajectory

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

  def localInitialize(self,solutionExport=None):
    """
      Method to initialize settings that belongs to all gradient based optimizer
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    self.gradDict['numIterForAve'] = int(self.paramDict.get('numGradAvgIterations', 1))
    for traj in self.optTraj:
      self.gradDict['pertPoints'][traj]      = {}
      self.counter['perturbation'][traj]     = 0
      self.counter['varsUpdate'][traj]       = 0
      self.counter['solutionUpdate'][traj]   = 0
      self.counter['gradientHistory'][traj]  = [{},{}]
      self.counter['gradNormHistory'][traj]  = [0.0,0.0]
      self.optVarsHist[traj]                 = {}
      self.readyVarsUpdate[traj]             = False
      self.convergeTraj[traj]                = False
      self.status[traj]                      = {'process':'submitting new opt points', 'reason':'just started'}
      self.counter['recentOptHist'][traj]    = [{},{}]
      self.trajectoriesKilled[traj]          = []
    # end job runnable equal to number of trajectory
    self._endJobRunnable = len(self.optTraj)
    #specializing the self.localLocalInitialize()
    if solutionExport != None:
      self.localLocalInitialize(solutionExport=solutionExport)
    else:
      self.localLocalInitialize()

  @abc.abstractmethod
  def localLocalInitialize(self, solutionExport = None):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
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
    if self.mdlEvalHist.isItEmpty():
      return (False,-1)
    prefix = self.mdlEvalHist.getMetadata('prefix')
    for index, pr in enumerate(prefix):
      pr = pr.split(utils.returnIdSeparator())[-1].split('_')
      # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
      if pr[0] == str(traj) and pr[1] == str(updateKey) and pr[2] == str(evalID):
        return (True, index)
    return (False, -1)

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    self.readyVarsUpdate = {traj:False for traj in self.optTrajLive}

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
    gradArray = {}
    for var in self.getOptVars(traj=traj):
      gradArray[var] = np.zeros(2) #why are we initializing to this?
    # Evaluate gradient at each point
    #for pertIndex in optVarsValues.keys():
    for i in range(self.gradDict['numIterForAve']):
      opt  = optVarsValues[i*2]      #the latest opt point
      pert = optVarsValues[i*2 + 1] #the perturbed point
      #calculate grad(F) wrt each input variable
      lossDiff = pert['output'] - opt['output']
      #cover "max" problems
      # TODO it would be good to cover this in the base class somehow, but in the previous implementation this
      #   sign flipping was only called when evaluating the gradient.
      if self.optType == 'max':
        lossDiff *= -1.0
      for var in self.getOptVars(traj=traj):
        # gradient is calculated in normalized space
        dh = pert['inputs'][var] - opt['inputs'][var]
        if abs(dh) < 1e-15:
          self.raiseAnError(RuntimeError,'While calculating the gradArray a "dh" very close to zero was found for var:',var)
        gradArray[var] = np.append(gradArray[var], lossDiff/dh)
    gradient = {}
    for var in self.getOptVars(traj=traj):
      gradient[var] = gradArray[var].mean()
    # currently unused, allow subclasses to modify gradient evaluation
    gradient = self.localEvaluateGradient(optVarsValues, gradient)
    # we intend for gradient to give direction only
    gradientNorm = np.linalg.norm(gradient.values())
    if gradientNorm > 0.0:
      for var in gradient.keys():
        gradient[var] = gradient[var]/gradientNorm
    self.counter['gradientHistory'][traj][1] = copy.deepcopy(self.counter['gradientHistory'][traj][0])
    self.counter['gradientHistory'][traj][0] = gradient
    self.counter['gradNormHistory'][traj][1] = copy.deepcopy(self.counter['gradNormHistory'][traj][0])
    self.counter['gradNormHistory'][traj][0] = gradientNorm
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

  def _updateConvergenceVector(self, traj, varsUpdate, currentLossValue):
    """
      Local method to update convergence vector.
      @ In, traj, int, identifier of the trajector to update
      @ In, varsUpdate, int, current variables update iteration number
      @ In, currentLossValue, float, current loss function value
      @ Out, None
    """
    if not self.convergeTraj[traj]:
      if len(self.counter['gradientHistory'][traj][1]) < 1:
        self.status[traj]['reason'] = 'found new opt point'
      else:
        objectiveOutputs = np.zeros(self.gradDict['numIterForAve'])
        for i in range(self.gradDict['numIterForAve']): #evens are opt point evaluations
          index = i*2
          objectiveOutputs[i] = self.getLossFunctionGivenId(self._createEvaluationIdentifier(traj,varsUpdate-1,index))
        if any(np.isnan(objectiveOutputs)):
          self.raiseAnError(Exception,"the objective function evaluation for trajectory " +str(traj)+ "and iteration "+str(varsUpdate-1)+" has not been found!")
        oldVal = objectiveOutputs.mean() # TODO should this be from counter[recentOptPoints][traj]?
        # see if new point is better than old point
        newerIsBetter = self.checkIfBetter(currentLossValue,oldVal)
        varK = self.denormalizeData(self.optVarsHist[traj][self.counter['varsUpdate'][traj]])
        ## convergence values
        # gradient norm
        gradNorm  = self.counter['gradNormHistory'][traj][0]
        gradientNormCheck = gradNorm <= self.gradientNormTolerance
        # absolute loss value difference
        absDifference = abs(currentLossValue-oldVal)
        absoluteTolCheck = absDifference <= self.absConvergenceTol
        # relative loss value difference
        relativeDifference = mathUtils.relativeDiff(currentLossValue,oldVal)
        relativeTolCheck = relativeDifference <= self.relConvergenceTol
        # store progress for verbosity options
        self.convergenceProgress[traj] = {'abs':absDifference,'rel':relativeDifference,'grad':gradNorm}
        # safety check against multiple evaluations on the same point
        sameCoordinateCheck = set(self.optVarsHist[traj][varsUpdate].items()) == set(self.counter['recentOptHist'][traj][0]['inputs'].items()) #set(self.optVarsHist[traj][varsUpdate-1].items())
        # min step size check
        try:
          lastStep = self.counter['lastStepSize'][traj]
          minStepSizeCheck = lastStep <= self.minStepSize
        except KeyError:
          #we reset the step size, so we don't have a value anymore
          lastStep = np.nan
          minStepSizeCheck = False
        # screen outputs
        self.raiseAMessage("Trajectory: "+"%8i"% (traj)+      " | Iteration    : "+"%8i"% (varsUpdate)+ " | Loss function: "+"%8.2E"% (currentLossValue)+" |")
        self.raiseAMessage("Grad Norm : "+"%8.2E"% (gradNorm)+" | Relative Diff: "+"%8.2E"% (relativeDifference)+" | Abs Diff     : "+"%8.2E"% (absDifference)+" |")
        self.raiseAMessage("Step Size : "+"%8.2E"% (lastStep))
        self.raiseAMessage("Input Location :" +str(varK))
        ## set up status going forward
        # if new point is better, accept it and move forward
        if newerIsBetter:
          self.status[traj]['reason'] = 'found new opt point'
          self.raiseADebug('Accepting potential opt point for improved loss value')
          #TODO REWORK this belongs in the base class optimizer; grad shouldn't know about multilevel!!
          #  -> this parameter is how multilevel knows that a successful perturbation of an outer loop has been performed
          self.mlActiveSpaceSteps[traj] += 1
          converged = minStepSizeCheck or sameCoordinateCheck or gradientNormCheck or absoluteTolCheck or relativeTolCheck
        # if newer point is not better, we're keeping the old point, and sameCoordinate, absoluteTol, and relativeTol aren't applicable
        else:
          self.status[traj]['reason'] = 'rejecting bad opt point'
          self.raiseADebug('Rejecting potential opt point for worse loss value: "{}" vs "{}"'.format(oldVal,currentLossValue))
          # cut the next step size to hopefully stay in the valley instead of climb up the other side
          self.recommendToGain[traj] = 'cut'
          converged = gradientNormCheck or minStepSizeCheck
        if converged:
          reasons = []
          if sameCoordinateCheck:
            reasons.append("same coordinate")
          if gradientNormCheck:
            reasons.append("gradient norm")
          if absoluteTolCheck:
            reasons.append("absolute tolerance")
          if relativeTolCheck:
            reasons.append("relative tolerance")
          if minStepSizeCheck:
            reasons.append("minimum step size")
          self.raiseAMessage("Trajectory: "+"%8i"% (traj) +"   converged. Reasons: "+', '.join(reasons))
          self.convergeTraj[traj] = True
          self.removeConvergedTrajectory(traj)

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
          dist = np.sqrt(np.sum(list((inp[var] - currentInput[var])**2 for var in self.getOptVars())))
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
    # get all the even-valued results (these are the multiple evaluations of the opt point)
    for i in range(self.gradDict['numIterForAve']):
      identifier = i*2
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
    try:
      prefix = jobObject.getMetadata()['prefix']
      self.raiseADebug('Collected sample "{}"'.format(prefix))
    except TypeError:
      prefix = ''
    self.raiseADebug('Collected sample "{}"'.format(prefix))

    # TODO REWORK move this whole piece to Optimizer base class as much as possible
    if self.solutionExport != None and len(self.mdlEvalHist) > 0:
      for traj in self.optTraj:
        while self.counter['solutionUpdate'][traj] <= self.counter['varsUpdate'][traj]:
          solutionExportUpdatedFlag, indices = self._getJobsByID(traj)

          if solutionExportUpdatedFlag:
            #get evaluations (input,output) from the collection of all evaluations
            inputeval=self.mdlEvalHist.getParametersValues('inputs', nodeId = 'RecontructEnding')
            outputeval=self.mdlEvalHist.getParametersValues('outputs', nodeId = 'RecontructEnding')
            #TODO this might be faster for non-stochastic if we do an "if" here on gradDict['numIterForAve']
            #make a place to store distinct evaluation values
            objectiveOutputs = np.zeros(self.gradDict['numIterForAve'])
            # get output values corresponding to evaluations of the opt point
            # also add opt points to the grad perturbation list
            self.gradDict['pertPoints'][traj] = np.zeros(2*self.gradDict['numIterForAve'],dtype=dict)
            for i, index in enumerate(indices):
              objectiveOutputs[i] = outputeval[self.objVar][index]
              self.gradDict['pertPoints'][traj][i*2] = {'inputs':self.normalizeData(dict((k,v[index]) for k,v in inputeval.items())),
                                                        'output':objectiveOutputs[i]}
            # assumed output value is the mean of sampled values
            currentObjectiveValue = objectiveOutputs.mean()
            # check convergence
            # TODO REWORK move this to localStillReady, along with the gradient evaluation
            self._updateConvergenceVector(traj, self.counter['solutionUpdate'][traj], currentObjectiveValue)
            if self.convergeTraj[traj]:
              self.status[traj] = {'process':None, 'reason':'converged'}
            else:
              # if rejecting bad point, keep the old point as the new point
              if self.status[traj]['reason'] != 'rejecting bad opt point':
                try:
                  self.counter['recentOptHist'][traj][1] = copy.deepcopy(self.counter['recentOptHist'][traj][0])
                except KeyError:
                  # this means we don't have an entry for this trajectory yet, so don't copy anything
                  pass
                self.counter['recentOptHist'][traj][0] = {'inputs':self.optVarsHist[traj][self.counter['varsUpdate'][traj]],
                                                          'output':currentObjectiveValue}
              # update status to submitting grad eval points
              self.status[traj]['process'] = 'submitting grad eval points'

            # update solution export
            if 'trajID' not in self.solutionExport.getParaKeys('inputs'):
              self.raiseAnError(IOError, 'trajID is not in the <inputs> space of the solutionExport data object specified for this optimization step!  Please add it.')
            trajID = traj+1 # This is needed to be compatible with historySet object
            self.solutionExport.updateInputValue([trajID,'trajID'], traj)
            output = self.solutionExport.getParametersValues('outputs', nodeId = 'RecontructEnding').get(trajID,{})
            badValue = -1 #value to use if we don't have a value # TODO make this accessible to user?
            for var in self.solutionExport.getParaKeys('outputs'):
              old = copy.deepcopy(output.get(var, np.asarray([])))
              new = None #prevents accidental data copying
              if var in self.getOptVars():
                new = self.denormalizeData(self.counter['recentOptHist'][traj][0]['inputs'])[var] #inputeval[var][index]
              elif var == self.objVar:
                new = self.counter['recentOptHist'][traj][0]['output'] #currentObjectiveValue
              elif var == 'varsUpdate':
                new = [self.counter['solutionUpdate'][traj]]
              elif var == 'stepSize':
                try:
                  new = [self.counter['lastStepSize'][traj]]
                except KeyError:
                  new = badValue
              elif var.startswith( 'gradient_'):
                varName = var[10:]
                vec = self.counter['gradientHistory'][traj][0].get(varName,None)
                if vec is not None:
                  new = vec*self.counter['gradNormHistory'][traj][0]
                else:
                  new = badValue
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
              new = np.asarray(new)
              self.solutionExport.updateOutputValue([trajID,var],np.append(old,new))

            self.counter['solutionUpdate'][traj] += 1
          else: #not ready to update solutionExport
            break

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
    if len(grad1) < 1:
      return 1.0
    #otherwise, do the dot product between the last two gradients
    grad0 = self.counter['gradientHistory'][traj][0]
    prod = np.sum(list(grad0[key]*grad1[key] for key in grad0.keys()))
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
      nPoint['prefix'] = self._createEvaluationIdentifier(traj,self.counter['varsUpdate'][traj],i*2) # evens (including 0) are opt point evals
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

