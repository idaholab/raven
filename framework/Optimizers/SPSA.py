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
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .GradientBasedOptimizer import GradientBasedOptimizer
import Distributions
#Internal Modules End--------------------------------------------------------------------------------

class SPSA(GradientBasedOptimizer):
  """
    Simultaneous Perturbation Stochastic Approximation Optimizer
  """
  def __init__(self):
    """
      Default Constructor
    """
    GradientBasedOptimizer.__init__(self)
    self.stochasticDistribution = None                        # Distribution used to generate perturbations
    self.stochasticEngine = None                              # Random number generator used to generate perturbations

  def localLocalInputAndChecks(self, xmlNode):
    """
      Local method for additional reading.
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    self.gainParamDict['alpha'] = float(self.paramDict.get('alpha', 0.602))
    self.gainParamDict['gamma'] = float(self.paramDict.get('gamma', 0.101))
    self.gainParamDict['A'] = float(self.paramDict.get('A', self.limit['mdlEval']/10))
    self.gainParamDict['a'] = float(self.paramDict.get('a', 0.16))
    self.gainParamDict['c'] = float(self.paramDict.get('c', 0.005))

    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2

    if self.paramDict.get('stochasticDistribution', 'Bernoulli') == 'Bernoulli':
      self.stochasticDistribution = Distributions.returnInstance('Bernoulli',self)
      self.stochasticDistribution.p = 0.5
      self.stochasticDistribution.initializeDistribution()
      self.stochasticEngine = lambda: [1.0 if self.stochasticDistribution.rvs() == 1 else -1.0 for _ in range(self.nVar)] # Initialize bernoulli distribution for random perturbation
    else:
      self.raiseAnError(IOError, self.paramDict['stochasticEngine']+'is currently not support for SPSA')

  def localLocalInitialize(self, solutionExport = None):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    self._endJobRunnable = 1 # batch mode currently not implemented for SPSA
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2

  def localLocalStillReady(self, ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, variable indicating whether the caller is prepared for another input.
    """
    if convergence:             ready = False
    else:                       ready = True and ready
    return ready

  def localLocalGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    if self.counter['mdlEval'] == 1: # Just started
      self.optVarsHist[self.counter['varsUpdate']] = {}
      for var in self.optVars:
        self.values[var] = self.optVarsInit['initial'][var]
        self.optVarsHist[self.counter['varsUpdate']][var] = copy.copy(self.values[var])

    elif not self.readyVarsUpdate: # Not ready to update decision variables; continue to perturb for gradient evaluation
      if self.counter['perturbation'] == 1: # Generate all the perturbations at once
        self.gradDict['pertPoints'] = {}
        ck = self._computeGainSequenceCk(self.gainParamDict,self.counter['varsUpdate']+1)
        varK = copy.deepcopy(self.optVarsHist[self.counter['varsUpdate']])
        for ind in range(self.gradDict['numIterForAve']):
          self.gradDict['pertPoints'][ind] = {}
          delta = self.stochasticEngine()
          for varID, var in enumerate(self.optVars):
            if var not in self.gradDict['pertPoints'][ind].keys():
              p1 = np.asarray([varK[var]+ck*delta[varID]*1.0]).reshape((1,))
              p2 = np.asarray([varK[var]-ck*delta[varID]*1.0]).reshape((1,))
              self.gradDict['pertPoints'][ind][var] = np.concatenate((p1, p2))

      loc1 = self.counter['perturbation'] % 2
      loc2 = np.floor(self.counter['perturbation'] / 2) if loc1 == 1 else np.floor(self.counter['perturbation'] / 2) - 1
      for var in self.optVars:
        self.values[var] = copy.deepcopy(self.gradDict['pertPoints'][loc2][var][loc1])

    else: # Enough gradient evaluation for decision variable update
      ak = self._computeGainSequenceAk(self.gainParamDict,self.counter['varsUpdate']) # Compute the new ak
      gradient = self.evaluateGradient(self.gradDict['pertPoints'])

      self.optVarsHist[self.counter['varsUpdate']] = {}
      varK = copy.deepcopy(self.optVarsHist[self.counter['varsUpdate']-1])
      varKPlus = {}
      
      varKPlus = self._generateVarsUpdateConstrained(ak,gradient,varK)
       
      for var in self.optVars:  
        self.values[var] = copy.deepcopy(varKPlus[var])
        self.optVarsHist[self.counter['varsUpdate']][var] = copy.deepcopy(self.values[var])

  def _generateVarsUpdateConstrained(self,ak,gradient,varK):
    tempVarKPlus = {} 
    foundVarsUpdate = False
    for var in self.optVars:
      tempVarKPlus[var] = copy.copy(varK[var]-ak*gradient[var]*1.0)
    
    if self.checkConstraint(tempVarKPlus):                  foundVarsUpdate = True
    
    # Try to find varKPlus by shorten the gradient vector
    if not foundVarsUpdate:
      foundVarsUpdate, tempVarKPlus = self._bisectionForConstrainedInput(varK, ak, gradient)

    # Try to find varKPlus by rotate the gradient towards its orthogonal           
    if not foundVarsUpdate:
      pass
    
    if not foundVarsUpdate:
      tempVarKPlus = {}
      for var in self.optVars:
        tempVarKPlus[var] = copy.copy(varK[var]-ak*gradient[var]*1.0)
#       self.raiseAnError(ValueError, 'varKPlus not found')    
    
    return tempVarKPlus

  def _bisectionForConstrainedInput(self,varK,gain,vector):
    paraBoundError = 1e-2
    paraFracLowerLimit = 1e-2
    bounds = [0, 1.0]
    tempVarNew = {}
    frac = 0.5
    while np.absolute(bounds[1]-bounds[0]) >= paraBoundError:
      for var in self.optVars:  
        tempVarNew[var] = copy.copy(varK[var]-gain*vector[var]*1.0*frac)
          
      if self.checkConstraint(tempVarNew):
        bounds[0] = copy.deepcopy(frac)
        if np.absolute(bounds[1]-bounds[0]) < paraBoundError:
          if frac >= paraFracLowerLimit:
            varKPlus = copy.deepcopy(tempVarNew)
            return True, varKPlus
          break
        frac = copy.deepcopy(bounds[1]+bounds[0])/2
      else:
        bounds[1] = copy.deepcopy(frac) 
        frac = copy.deepcopy(bounds[1]+bounds[0])/2
    return False, None    
     

  def localEvaluateGradient(self, optVarsValues, gradient = None):
    """
      Local method to evaluate gradient.
      @ In, optVarsValues, dict, Dictionary containing perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ In, gradient, dict, optional, dictionary containing gradient estimation by the caller.
                                      gradient should have the form {varName: gradEstimation}
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    return gradient

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

  def _computeGainSequenceAk(self,paramDict,iterNum):
    """
      Utility function to compute the ak coefficients (gain sequence ak)
      @ In, paramDict, dict, dictionary containing information to compute gain parameter
      @ In, iterNum, int, current iteration index
      @ Out, ak, float, current value for gain ak
    """
    a, A, alpha = paramDict['a'], paramDict['A'], paramDict['alpha']
    ak = a / (iterNum + A) ** alpha *1.0
    return ak

  def localCheckConvergence(self, convergence = False):
    """
      Local method to check convergence.
      @ In, convergence, bool, optional, variable indicating how the caller determines the convergence.
      @ Out, convergence, bool, variable indicating whether the convergence criteria has been met.
    """
    return convergence
  
  def localCheckConstraint(self, optVars, satisfaction = True):
    """
      Local method to check whether a set of decision variables satisfy the constraint or not
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ In, satisfaction, bool, optional, variable indicating how the caller determines the constraint satisfaction at the point optVars
      @ Out, satisfaction, bool, variable indicating the satisfaction of constraints at the point optVars
    """
    return satisfaction  
