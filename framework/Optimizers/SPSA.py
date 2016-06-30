"""
  This module contains the Simultaneous Perturbation Stochastic Approximation Optimization strategy

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
    self.gainParamDict['alpha'] = self.paramDict.get('alpha', 0.602)
    self.gainParamDict['gamma'] = self.paramDict.get('gamma', 0.101)
    self.gainParamDict['A'] = self.paramDict.get('A', self.limit['mdlEval']/10)
    self.gainParamDict['a'] = self.paramDict.get('a', 0.16)
    self.gainParamDict['c'] = self.paramDict.get('c', 0.005)

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
      @ In, solutionExport, DataObject, optional
      @ Out, None
    """
    self._endJobRunnable = 1 # batch mode currently not implemented for SPSA
    self.gradDict['pertNeeded'] = self.gradDict['numIterForAve'] * 2

  def localLocalStillReady(self, ready, convergence = False):
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @In, ready, bool, boolean variable indicating whether the caller is prepared for another input.
      @In, convergence, boolean variable indicating whether the convergence criteria has been met.
      @Out, ready, bool, boolean variable indicating whether the caller is prepared for another input.
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
        self.values[var] = self.gradDict['pertPoints'][loc2][var][loc1]

    else: # Enough gradient evaluation for decision variable update
      ak = self._computeGainSequenceAk(self.gainParamDict,self.counter['varsUpdate']) # Compute the new ak
      gradient = self.evaluateGradient(self.gradDict['pertPoints'])

      self.optVarsHist[self.counter['varsUpdate']] = {}
      varK = copy.deepcopy(self.optVarsHist[self.counter['varsUpdate']-1])
      for var in self.optVars:
        self.values[var] = copy.copy(varK[var]-ak*gradient[var]*1.0)
        self.optVarsHist[self.counter['varsUpdate']][var] = copy.copy(self.values[var])

  def localEvaluateGradient(self, optVarsValues, gradient = None):
    """
      Local method to evaluate gradient.
      @In, optVarsValues, Dict containing perturbed points.
           optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
           Therefore, each optVarsValues[pertIndex] should return a dict of variable values that is sufficient for gradient
           evaluation for at least one variable (depending on specific optimization algorithm)
      @In, gradient, Dict containing gradient estimation by the caller. gradient should have the form {varName: gradEstimation}
      @Out, gradient, Dict containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    return gradient

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

  def localCheckConvergence(self, convergence = False):
    """
      Local method to check convergence.
      @In, convergence, boolean variable indicating how the caller determines the convergence.
      @Out, convergence, boolean variable indicating whether the convergence criteria has been met.
    """
    return convergence
