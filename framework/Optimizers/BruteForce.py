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
from collections import deque
import itertools
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
    return inputSpecification

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    OptimizerBase.__init__(self)
    self.counter['varsUpdate'      ] = {}
    self.counter['solutionUpdate'  ] = {}
    # register metadata
    self.addMetaKeys('prefix')

  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, none,
      @ Out, convergence, bool, variable indicating whether the convergence criteria has been met.
    """
    pass

  def localInitialize(self,solutionExport):
    """
      Method to initialize settings that belongs to all gradient based optimizer
      @ In, solutionExport, DataObject, a PointSet to hold the solution
      @ Out, None
    """
    for traj in self.optTraj:
      self.counter['recentOptHist'][traj]    = [{},{}]
      self.counter['solutionUpdate'][traj]   = 0
      self.optVarsHist[traj]                 = {}
    # Currently we only accept one vector of input
    size = 0
    for var in self.getOptVars(traj=0):
      size += np.prod(self.variableShapes[var])
    self._generateSubmissionQueue(size=size)
    self.limit['mdlEval'] = len(self.submissionQueue[0])

  def _generateSubmissionQueue(self,size=1):
    """
    """
    values = map(list,list(itertools.product([0,1], repeat=size)))
    #create identifier
    values = deque([{'inputs':values[i],'prefix':self._createEvaluationIdentifier(trajID=0,iterID=0,evalType=i)} for i in range(len(values))])
    #queue it up
    self.submissionQueue[0] = values

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    traj = 0
    OptimizerBase.localGenerateInput(self,model,oldInput)
    #"submit" the next queued point
    prefix,point = self.getQueuedPoint(traj)
    # TODO point here is just the values
    # In Optimizer, point is a dict {var, value}, but for this case, we just use value for now
    for var in self.getOptVars(traj=traj):
      self.values[var] = point
    self.inputInfo['prefix'] = prefix

  def _getJobsByID(self,prefix):
    """
      Overwrite only if you need something special at the end of each run....
      This function is used by optimizers that need to collect information from the just ended run
      @ In, traj, int, ID of the trajectory for whom we collect jobs
      @ Out, solutionExportUpdatedFlag, bool, True if the solutionExport needs updating
      @ Out, solutionIndeces, list(int), location of updates within the full targetEvaluation data object
    """
    jobInfo = prefix.split("_")
    trajID, iterID, evalID = int(jobInfo[0]), int(jobInfo[1]), int(jobInfo[2])
    solutionIndeces = []
    solutionExportUpdatedFlag, index = self._checkModelFinish(trajID, iterID, evalID)
    solutionIndeces.append(index)
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
      self.raiseAnError(IOError, 'Failed runs, aborting RAVEN')
    # TODO REWORK move this whole piece to Optimizer base class as much as possible
    if len(self.mdlEvalHist) != 0:
      for traj in self.optTraj:
        if self.counter['solutionUpdate'][traj] <= self.counter['varsUpdate'][traj]:
          # check whether solution export needs updating, and get indices of entries that need to be added
          solutionExportUpdatedFlag, indices = self._getJobsByID(prefix)
          if solutionExportUpdatedFlag:
            outputs = dict((var,np.zeros(len(self.optTraj), dtype=object)) for var in self.solutionExport.getVars('output')
                      if var in self.mdlEvalHist.getVars('output'))
            for i, index in enumerate(indices):
              # get the realization from the targetEvaluation
              vals = self.mdlEvalHist.realization(index=index)
              # place values TODO this could be vectorized significantly!
              for var in outputs.keys():
                if hasattr(vals[var],'__len__') and len(vals[var]) == 1:
                  outputs[var][i] = float(vals[var])
                else:
                  outputs[var][i] = vals[var]
              currentObjectiveValue = outputs[self.objVar]
            try:
              self.counter['recentOptHist'][traj][1] = copy.deepcopy(self.counter['recentOptHist'][traj][0])
            except KeyError:
              # this means we don't have an entry for this trajectory yet, so don't copy anything
              pass
            # store realization of most recent developments
            self.counter['recentOptHist'][traj][0] = outputs
            if traj not in self.counter['prefixHistory']:
              self.counter['prefixHistory'][traj] = []
            self.counter['prefixHistory'][traj].append(prefix)
            # only write here if we want to write on EVERY optimizer iteration (each new optimal point)
            if self.writeSolnExportOn == 'every':
              self.writeToSolutionExport(traj)
            # whether we wrote to solution export or not, update the counter
            self.counter['solutionUpdate'][traj] += 1
          else: #not ready to update solutionExport
            break

  def writeToSolutionExport(self,traj=0):
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
        new = recent[var]
      elif var in self.constants:
        new = self.constants[var]
      # custom counters: varsUpdate, trajID, stepSize
      elif var == 'varsUpdate':
        new = self.counter['solutionUpdate'][traj]
      elif var == 'trajID':
        new = traj+1 # +1 is for historical reasons, when histories were indexed on 1 instead of 0
      else:
        self.raiseAnError(IOError,'Unrecognized output request:',var)
      # format for realization
      rlz[var] = np.atleast_1d(new)
    self.solutionExport.addRealization(rlz)

  def finalizeSampler(self,failedRuns):
    """
      Method called at the end of the Step when no more samples will be taken.  Closes out sampler for step.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    OptimizerBase.handleFailedRuns(self, failedRuns)
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
