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

    
  def localLocalInputAndChecks(self, xmlNode):
    self.gainParamDict['alpha'] = self.paramDict.get('alpha', 0.602)
    self.gainParamDict['gamma'] = self.paramDict.get('gamma', 0.101)
    self.gainParamDict['A'] = self.paramDict.get('alpha', self.limit['mdlEval']/10)
    self.gainParamDict['a'] = self.paramDict.get('alpha', None)
    self.gainParamDict['c'] = self.paramDict.get('alpha', 0.005)
    
#     if 'alpha' not in self.gainDict.keys():       self.gainDict['alpha']    = 0.602
#     if 'gamma' not in self.gainDict.keys():       self.gainDict['gamma']    = 0.101
#     if 'A' not in self.gainDict.keys():           self.gainDict['A']        = self.limit['mdlEval']/10
#     if 'a' not in self.gainDict.keys():           self.gainDict['a']        = None
#     if 'c' not in self.gainDict.keys():           self.gainDict['c']        = 0.005
 
    
    pass # to be overwritten by subclass

  def localLocalInitialize(self, solutionExport = None):
    self._endJobRunnable = 1 # batch mode currently not implemented for SPSA
    self.numPerturbationsNeeded = 2 * self.gradAveNumber
  
  def localLocalStillReady(self, ready, convergence = False):
    return ready
    
  def localEvaluateGradient(self, optVars, gradient):
    return gradient
  
  def localLocalGenerateInput(self,model,oldInput):
    for var in self.optVars:
      self.values[var] = 0.2
#     self.raiseADebug(self.counter['mdlEval'], self.limit)
#     self.raiseADebug(self.values)
      
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