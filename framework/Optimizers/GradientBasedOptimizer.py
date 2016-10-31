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
from sklearn.neighbors import NearestNeighbors
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Optimizer import Optimizer
from Assembler import Assembler
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
    self.ROMModelFinishCheck = NearestNeighbors(n_neighbors=1) # ROM to check whether result is returned from Model
    self.constraintHandlingPara = {}                  # Dict containing parameters for parameters related to constraints handling
    self.gradDict = {}                                # Dict containing information for gradient related operations
    self.gradDict['numIterForAve'] = 1                # Number of iterations for gradient estimation averaging
    self.gradDict['pertNeeded'] = 1                   # Number of perturbation needed to evaluate gradient
    self.gradDict['pertPoints'] = {}                  # Dict containing inputs sent to model for gradient evaluation
    self.counter['perturbation'] = {}                  # Counter for the perturbation performed.
    self.readyVarsUpdate = {}                       # Bool variable indicating the finish of gradient evaluation and the ready to update decision variables
    self.counter['varsUpdate'] = {}
    self.counter['solutionUpdate'] = {}
    self.convergeTraj = {}

  def localInputAndChecks(self, xmlNode):
    """
      Method to read the portion of the xml input that belongs to all gradient based optimizer only
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    self.gradDict['numIterForAve'] = int(self.paramDict.get('numGradAvgIterations', 1))
    for traj in self.optTraj:
      self.gradDict['pertPoints'][traj]       = {}
      self.counter['perturbation'][traj]      = 0
      self.counter['varsUpdate'][traj]        = 0
      self.counter['solutionUpdate'][traj]    = 0
      self.optVarsHist[traj]                  = {}
      self.readyVarsUpdate[traj]              = False
      self.convergeTraj[traj]                 = False

  def localInitialize(self,solutionExport=None):
    """
      Method to initialize settings that belongs to all gradient based optimizer
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    for traj in self.optTraj:
      self.gradDict['pertPoints'][traj] = {}

    #specializing the self.localLocalInitialize()
    if solutionExport != None : self.localLocalInitialize(solutionExport=solutionExport)
    else                      : self.localLocalInitialize()

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
    if ready == False:
      return ready # Return if we exceed the max iterations or converges...

    readyFlag = False
    for traj in self.optTrajLive:
      if self.counter['varsUpdate'][traj] < self.limit['varsUpdate']:
        readyFlag = True
    if readyFlag == False:
      ready = False
      return ready # Return not ready if all trajectories has more them permitted variable updates.  

    if self.mdlEvalHist.isItEmpty():
      for traj in self.optTrajLive:
        if self.counter['perturbation'][traj] < self.gradDict['pertNeeded']: # Return if we just initialize
          return ready
      ready = False # Waiting for the model output for gradient evaluation
    else:
      readyFlag = False
      for traj in self.optTrajLive:
        if self.counter['perturbation'][traj] < self.gradDict['pertNeeded']: # Return if we just initialize
#           ready = True
          readyFlag = True
          break
        else:
          evalNotFinish = False     
          for locI in range(self.gradDict['numIterForAve']):
            for locP in range(self.gradDict['pertPoints'][traj][locI][self.optVars[0]].size):
              optVars = {}
              for var in self.optVars:
                optVars[var] = self.gradDict['pertPoints'][traj][locI][var][locP]
              if not self._checkModelFinish(optVars):
                evalNotFinish = True
                break
            if evalNotFinish: break
          if evalNotFinish: 
            pass
          else: 
            readyFlag = True
            break
      if readyFlag: ready = True
      else:         ready = False
      
#       self.raiseADebug(self.counter['perturbation'][traj],self.gradDict['pertNeeded'])
#       self.raiseADebug(self.optTrajLive, ready)   
#       self.raiseADebug(readyFlag,readyFlag)
#       self.raiseAnError(IOError, 'test')
#        self.counter['perturbation'] >= self.gradDict['pertNeeded']:
#       if len(self.mdlEvalHist) % (self.gradDict['pertNeeded']+1): ready = False # Waiting for the model output for gradient evaluation

    ready = self.localLocalStillReady(ready, convergence)

    return ready

  def _checkModelFinish(self, optVarsValues):
    """
      Determines if the Model has finished running an input and returned the output
      @ In, optVarsValues, dict, dictionary containing the values of input to be checked. Only one input is allowed
      @ Out, _checkModelFinish, bool, indicating whether the Model has finished the evaluation over input of optVarsValues
    """
    if self.mdlEvalHist.isItEmpty():    return False
    
    tempDict = copy.copy(self.mdlEvalHist.getParametersValues('inputs', nodeId = 'RecontructEnding'))
    tempSamp = np.zeros(shape=(len(self.mdlEvalHist),self.nVar))
    tempX = np.zeros(shape=(1,self.nVar))
    for varInd, var in enumerate(self.optVars):
      tempSamp[:,varInd] = tempDict[var]
      tempX[0,varInd] = optVarsValues[var]
    self.ROMModelFinishCheck.fit(tempSamp)
    dist, ind = self.ROMModelFinishCheck.kneighbors(tempX, n_neighbors=1, return_distance=True)
    if dist == 0:         return True
    else:                 return False
      
    
  @abc.abstractmethod
  def localLocalStillReady(self, ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, boolean variable indicating whether the caller is prepared for another input.
    """
    pass

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    self.readyVarsUpdate = {traj:False for traj in self.optTrajLive}


#     self.localLocalGenerateInput(model,oldInput)
# 
#   @abc.abstractmethod
#   def localLocalGenerateInput(self,model,oldInput):
#     """
#       This class need to be overwritten since it is here that the magic of the optimizer happens.
#       After this method call the self.inputInfo should be ready to be sent to the model
#       @ In, model, model instance, it is the instance of a RAVEN model
#       @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
#       @ Out, None
#     """
#     pass

  def evaluateGradient(self, optVarsValues):
    """
      Method to evaluate gradient based on perturbed points and model evaluations.
      @ In, optVarsValues, dict, dictionary containing perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    gradArray = {}
    for var in self.optVars:                      gradArray[var] = np.ndarray((0,0))

    # Evaluate gradient at each point
    for pertIndex in optVarsValues.keys():
      tempDictPerturbed = optVarsValues[pertIndex]
      tempDictPerturbed['lossValue'] = copy.copy(self.lossFunctionEval(tempDictPerturbed))
      lossDiff = tempDictPerturbed['lossValue'][0] - tempDictPerturbed['lossValue'][1]
      for var in self.optVars:
        if tempDictPerturbed[var][0] != tempDictPerturbed[var][1]:
          gradArray[var] = np.append(gradArray[var], lossDiff/(tempDictPerturbed[var][0]-tempDictPerturbed[var][1])*1.0)

    gradient = {}
    for var in self.optVars:
      gradient[var] = gradArray[var].mean()

    gradient = self.localEvaluateGradient(optVarsValues, gradient)
    return gradient

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
      @ In, none,
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
    if self.convergeTraj[traj] == False:
      if varsUpdate > 1: 
        oldVal = copy.deepcopy(self.lossFunctionEval(self.optVarsHist[traj][varsUpdate-1]))
        if abs(currentLossValue-oldVal) < self.convergenceTol:
          self.convergeTraj[traj] = True
          for trajInd, tr in enumerate(self.optTrajLive):
            if tr == traj:
              self.optTrajLive.pop(trajInd)
    
#     convergeTraj = [False]*len(self.optTrajLive)
#     for trajInd, traj in enumerate(self.optTrajLive):
#       if self.counter['varsUpdate'][traj] >= 2:
#         optVal1 = copy.deepcopy(self.lossFunctionEval(self.optVarsHist[traj][self.counter['varsUpdate'][traj]]))
#         optVal2 = copy.deepcopy(self.lossFunctionEval(self.optVarsHist[traj][self.counter['varsUpdate'][traj]-1]))
#         if abs(optVal1-optVal2) < self.convergenceTol:
#           convergeTraj[trajInd] = True
#           self.optTrajLive.pop(trajInd)

  def localCheckConstraint(self, optVars, satisfaction = True):
    """
      Local method to check whether a set of decision variables satisfy the constraint or not
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ In, satisfaction, bool, optional, variable indicating how the caller determines the constraint satisfaction at the point optVars
      @ Out, satisfaction, bool, variable indicating the satisfaction of constraints at the point optVars
    """
    return satisfaction

  def localFinalizeActualSampling(self,jobObject,model,myInput):
    """
      Overwrite only if you need something special at the end of each run....
      This function is used by optimizers that need to collect information from the just ended run
      @ In, jobObject, instance, an instance of a JobHandler
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, myInput, list, the generating input
    """
# <<<<<<< HEAD
#       
# #       solutionExportUpdatedFlag = False
#       for traj in self.optTrajLive:
#         solutionExportUpdatedFlag = True
#         if self.counter['solutionUpdate'][traj] <= self.counter['varsUpdate'][traj] and self.counter['solutionUpdate'][traj] in self.optVarsHist[traj].keys():
#           for var in self.optVars:
# #             self.raiseADebug(traj, self.optVarsHist[traj], self.optTrajLive)
#             if self.optVarsHist[traj][self.counter['solutionUpdate'][traj]][var] != inputeval[var]:
#               solutionExportUpdatedFlag = False
#               break
#         else:
#           solutionExportUpdatedFlag = False
#         if solutionExportUpdatedFlag:
#           # check convergence
#           self._updateConvergenceVector(traj, self.counter['solutionUpdate'][traj], outputeval[self.objVar])
#           
#           # update solution export
#           for var in self.solutionExport.getParaKeys('inputs'):
#             if var in self.optVars:
#               self.solutionExport.updateInputValue(var,inputeval[var])
#           if 'varsUpdate' in self.solutionExport.getParaKeys('inputs'):
#             self.solutionExport.updateInputValue('varsUpdate', np.asarray([self.counter['solutionUpdate'][traj]]))
#           if 'traj' in self.solutionExport.getParaKeys('inputs'):
#             self.solutionExport.updateInputValue('traj', np.asarray([traj]))
#           for var in self.solutionExport.getParaKeys('outputs'):
#             if var == self.objVar:
#               self.solutionExport.updateOutputValue(var, outputeval[var])
#           self.counter['solutionUpdate'][traj] += 1
#           break
# 
# 
# 
# =======
    if self.solutionExport != None and len(self.mdlEvalHist) > 0:
      for traj in self.optTrajLive:        
        while self.counter['solutionUpdate'][traj] <= self.counter['varsUpdate'][traj]:
          solutionExportUpdatedFlag = False
          prefix = self.mdlEvalHist.getMetadata('prefix')
#           # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
#           pre = str(traj) + '_' + str(self.counter['varsUpdate'][traj]) + '_v_' + str(self.counter['mdlEval'])
          for index, pr in enumerate(prefix):
            pr = pr.split('|')[-1].split('_')
            # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
            if pr[0] == str(traj) and pr[1] == str(self.counter['solutionUpdate'][traj]) and pr[2] == 'v':
              solutionExportUpdatedFlag = True
              break
          
          if solutionExportUpdatedFlag:
            inputeval=self.mdlEvalHist.getParametersValues('inputs', nodeId = 'RecontructEnding')
            outputeval=self.mdlEvalHist.getParametersValues('outputs', nodeId = 'RecontructEnding')
            
            # check convergence
            self._updateConvergenceVector(traj, self.counter['solutionUpdate'][traj], outputeval[self.objVar][index])

            # update solution export
            if 'trajID' not in self.solutionExport.getParaKeys('inputs'):
              self.raiseAnError(ValueError, 'trajID is not in the input space of solutionExport')
            else:
              trajID = traj+1 # This is needed to be compatible with historySet object
              self.solutionExport.updateInputValue([trajID,'trajID'], traj)
#               self.raiseADebug(self.solutionExport.getParametersValues('inputs', nodeId = 'RecontructEnding'))
#               self.raiseADebug(self.solutionExport.getParaKeys('outputs'))
              
              tempOutput = self.solutionExport.getParametersValues('outputs', nodeId = 'RecontructEnding')
              
              tempTrajOutput = tempOutput.get(trajID, {})
              for var in self.solutionExport.getParaKeys('outputs'):
                if var in self.optVars:
                  tempTrajOutputVar = copy.deepcopy(tempTrajOutput.get(var, np.asarray([])))
                  self.solutionExport.updateOutputValue([trajID,var],np.append(tempTrajOutputVar,np.asarray(inputeval[var][index])))
                elif var == self.objVar:
                  tempTrajOutputVar = copy.deepcopy(tempTrajOutput.get(var, np.asarray([])))
                  self.solutionExport.updateOutputValue([trajID,var], np.append(tempTrajOutputVar,np.asarray(outputeval[var][index])))
              if 'varsUpdate' in self.solutionExport.getParaKeys('outputs'):
                tempTrajOutputVar = copy.deepcopy(tempTrajOutput.get('varsUpdate', np.asarray([])))
                self.solutionExport.updateOutputValue([trajID,'varsUpdate'], np.append(tempTrajOutputVar,np.asarray([self.counter['solutionUpdate'][traj]])))
#               self.raiseADebug(self.solutionExport.getParametersValues('inputs', nodeId = 'RecontructEnding'))
#               self.raiseADebug(self.solutionExport.getParametersValues('outputs', nodeId = 'RecontructEnding'))
              self.counter['solutionUpdate'][traj] += 1  

#               if traj not in tempOutput.keys():
#                 
#                 
#                 for var in self.solutionExport.getParaKeys('outputs'):
#                   if var in self.optVars:
#                     self.solutionExport.updateOutputValue([traj,var],np.asarray(inputeval[var][index]))
#                   elif var == self.objVar:
#                     self.solutionExport.updateOutputValue([traj,var], np.asarray(outputeval[var][index]))
#                 if 'varsUpdate' in self.solutionExport.getParaKeys('outputs'):
#                   self.solutionExport.updateOutputValue([traj,'varsUpdate'], np.asarray([self.counter['solutionUpdate'][traj]]))
#               else:
#                 
#                 
#                 pass
              
#               for var in self.solutionExport.getParaKeys('outputs'):
#                 if var in self.optVars:
#                   self.solutionExport.updateInputValue(var,inputeval[var][index])
#                 if 'varsUpdate' in self.solutionExport.getParaKeys('inputs'):
#                   self.solutionExport.updateInputValue('varsUpdate', np.asarray([self.counter['solutionUpdate'][traj]]))
#                 for var in self.solutionExport.getParaKeys('outputs'):
#                   if var == self.objVar:
#                     self.solutionExport.updateOutputValue(var, outputeval[var][index])

#               self.raiseAnError(IOError, 'sgege')
#               self.counter['solutionUpdate'][traj] += 1
              
##   Back up codes below for solutionExport as Pointset
#            # update solution export
#             for var in self.solutionExport.getParaKeys('inputs'):
#               if var in self.optVars:
#                 self.solutionExport.updateInputValue(var,inputeval[var][index])
#             if 'varsUpdate' in self.solutionExport.getParaKeys('inputs'):
#               self.solutionExport.updateInputValue('varsUpdate', np.asarray([self.counter['solutionUpdate'][traj]]))
#             for var in self.solutionExport.getParaKeys('outputs'):
#               if var == self.objVar:
#                 self.solutionExport.updateOutputValue(var, outputeval[var][index])
#             self.counter['solutionUpdate'][traj] += 1
##   Back up codes above for solutionExport as Pointset            
            
          else:
            break