"""
  This module contains the Finite Difference Optimization sampling strategy

  Created on June 16, 2016
  @author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from collections import OrderedDict
import sys
import os
import copy
import abc
import numpy as np
import json
from operator import mul,itemgetter
from collections import OrderedDict
from functools import reduce
from scipy import spatial
from scipy.interpolate import InterpolatedUnivariateSpline
import xml.etree.ElementTree as ET
import itertools
from math import ceil
from collections import OrderedDict
from sklearn import neighbors
from sklearn.utils.extmath import cartesian

if sys.version_info.major > 2: import pickle
else: import cPickle as pickle
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .GradientBasedOptimizer import GradientBasedOptimizer
from Assembler import Assembler
import Distributions
import DataObjects
import TreeStructure as ETS
import SupervisedLearning
import pyDOE as doe
import Quadratures
import OrthoPolynomials
import IndexSets
import Models
import PostProcessors
import MessageHandler
import GridEntities
from AMSC_Object import AMSC_Object
#Internal Modules End--------------------------------------------------------------------------------

class SPSA(GradientBasedOptimizer):
  def __init__(self):
    GradientBasedOptimizer.__init__(self)
    self.stochasticDistribution = None
    self.stochasticEngine = None

    
  def localLocalInputAndChecks(self, xmlNode):
    self.gainParamDict['alpha'] = self.paramDict.get('alpha', 0.602)
    self.gainParamDict['gamma'] = self.paramDict.get('gamma', 0.101)
    self.gainParamDict['A'] = self.paramDict.get('alpha', self.limit['mdlEval']/10)
    self.gainParamDict['a'] = self.paramDict.get('alpha', 0.16)
    self.gainParamDict['c'] = self.paramDict.get('alpha', 0.005)
    
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2
    
    if self.paramDict.get('stochasticDistribution', 'Bernoulli') == 'Bernoulli':
      self.stochasticDistribution = Distributions.returnInstance('Bernoulli',self)
      self.stochasticDistribution.p = 0.5
      self.stochasticDistribution.initializeDistribution()
      self.stochasticEngine = lambda: [1.0 if self.stochasticDistribution.rvs() == 1 else -1.0 for _ in range(self.nVar)] # Initialize bernoulli distribution for random perturbation
    else:
      self.raiseAnError(IOError, self.paramDict['stochasticEngine']+'is currently not support for SPSA')

  def localLocalInitialize(self, solutionExport = None):
    self._endJobRunnable = 1 # batch mode currently not implemented for SPSA
    self.gradDict['pertNeeded'] = 2 * self.gradDict['numIterForAve']
      
  
  def localLocalStillReady(self, ready, convergence = False):
    if convergence:             ready = False
    else:                       ready = True and ready
    return ready
    
  def localEvaluateGradient(self, optVarsValues, gradient = None):
    """
    optVarsValues are the perturbed parameter values for gradient estimation
    optVarsValues should have the form of {pertIndex: {varName: [varValue1 varValue2]}}
    Therefore, each optVarsValues[pertIndex] should return a dict of variable values that is sufficient for gradient 
    evaluation for at least one variable (depending on specific optimization algorithm) 
    """    
    return gradient
  
  def localLocalGenerateInput(self,model,oldInput):
       
    if self.counter['mdlEval'] == 1:
      self.optVarsHist[self.counter['varsUpdate']] = {}
      for var in self.optVars:
        self.values[var] = 0
        self.optVarsHist[self.counter['varsUpdate']][var] = copy.copy(self.values[var])
        
    elif not self.readyVarsUpdate: # Not ready to update decision variables; continue to perturb for gradient evaluation
      if self.counter['perturbation'] == 1:
        self.gradDict['pertPoints'] = {}
        ck = self._computeGainSequenceCk(self.gainParamDict,self.counter['varsUpdate']+1)
        varK = copy.deepcopy(self.optVarsHist[self.counter['varsUpdate']])
        for ind in range(self.gradDict['numIterForAve']):
          self.gradDict['pertPoints'][ind] = {}
          delta = self.stochasticEngine()
          for varID, var in enumerate(self.optVars):
            if var not in self.gradDict['pertPoints'][ind].keys():
              self.gradDict['pertPoints'][ind][var] = [varK[var]+ck*delta[varID]*1.0, varK[var]-ck*delta[varID]*1.0]
            
    
          
#         self.raiseADebug(self.gradDict['pertPoints'])
#         self.raiseAnError(IOError, 'ff')
      
      for var in self.optVars:
        self.values[var] = 0.1
        

      
    else: # Enough gradient evaluation for decision variable update
      ak = self._computeGainSequenceAk(self.gainParamDict,self.counter['varsUpdate']) # Compute the new ak
      gradient = self.evaluateGradient(self.gradDict['pertPoints'])
      
      self.optVarsHist[self.counter['varsUpdate']] = {}
      varK = copy.deepcopy(self.optVarsHist[self.counter['varsUpdate']-1])
      for var in self.optVars:
        self.values[var] = copy.copy(varK[var]-ak*gradient[var]*1.0)
        self.raiseADebug(gradient[var])
        self.optVarsHist[self.counter['varsUpdate']][var] = copy.copy(self.values[var])

      self.raiseADebug(self.values)
      self.raiseAnError(IOError, 's')
#     for var in self.optVars:
#       self.values[var] = 0.2

  def _computeGainSequenceCk(self,paramDict,iterNum):
    """
    Utility function to compute the ck coefficients (gain sequence ck)
    @ In, None
    @ Out, an iterator for the gain sequence ck
    """
    c, gamma = paramDict['c'], paramDict['gamma']
    ck = c / (iterNum) ** gamma *1.0
    return ck
  
  def _computeGainSequenceAk(self,paramDict,iterNum):
    """
    Utility function to compute the ak coefficients (gain sequence ak)
    @ In, None
    @ Out, an iterator for the gain sequence ak
    """
    a, A, alpha = paramDict['a'], paramDict['A'], paramDict['alpha']
    ak = a / (iterNum + A) ** alpha *1.0
    return ak
        
#   def _computeAka(self,paramDict):
#     """
#     Utility function to compute the parameter 'a' appears in the ak (gain sequence ak)
#     @ In, paramDict, dictionary containing some parameters for SPSA
#     @ Out, parameter 'a'
#     """
#     A, alpha = paramDict['A'], paramDict['alpha']
#     minThetaKey = min(self.thetakCurrent, key = self.thetakCurrent.get)
#     minCurrentThetakElement = self.thetakCurrent[minThetaKey]
#     if minCurrentThetakElement < 0:     self.raiseAnError(Exception, "theta value is negative!")
#     for  varIdGradient, key in enumerate(self.axisName):
#       if key == minThetaKey:    break
#     minTestGradientElement = np.abs(self.gradientEstimationCurrent[varIdGradient])
#     a = 0.1*minCurrentThetakElement*((self.counterParamUpdate + A) ** alpha) / minTestGradientElement
#     return a
            
  def localInitialize(self, solutionExport):
    """
      use this function to add initialization features to the derived class
      it is call at the beginning of each step
      @ In, None
      @ Out, None
    """
    pass 
  
  def localCheckConvergence(self, convergence = False):
    return convergence    