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
  This module contains the Brute Force strategy for Knapsack 0-1 problems

  Created on Jan. 16, 2019
  @author: wangc, mandd
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
import abc
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .OptimizerBase import OptimizerBase
from Assembler import Assembler
from utils import utils,cached_ndarray,mathUtils
#Internal Modules End--------------------------------------------------------------------------------

class BruteForce(OptimizerBase):
  """
    Brute Force Method to solve Kapsack 0-1 problems
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls
    """
    inputSpecification = super(BruteForce,cls).getInputSpecification()

    inputSpecification.addSub(param)
    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    OptimizerBase.__init__(self)

  def localInputAndChecks(self, xmlNode):
    """
      Local method for additional reading.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """

  def localInitialize(self,solutionExport):
    """
      Method to initialize settings that belongs to all gradient based optimizer
      @ In, solutionExport, DataObject, a PointSet to hold the solution
      @ Out, None
    """

  def localStillReady(self, ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, variable indicating whether the caller is prepared for another input.
    """

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """

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
