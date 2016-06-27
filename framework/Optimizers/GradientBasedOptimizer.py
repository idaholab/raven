"""
  This module contains the Gradient Based Optimization sampling strategy

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
from .Optimizer import Optimizer
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

class GradientBasedOptimizer(Optimizer):    
  def __init__(self):
    Optimizer.__init__(self)
    
    self.gainParamDict = {}
    
    self.gradDict = {}
    self.gradDict['numIterForAve'] = 1
    self.gradDict['pertNeeded'] = 1
    self.gradDict['pertPoints'] = {}
    self.counter['perturbation'] = 0
    self.readyVarsUpdate = None
    
  
  def localInputAndChecks(self, xmlNode):
    """
      Local method. Place here the additional reading, remember to add initial parameters in the method localGetInitParams
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    
    
    self.gradDict['numIterForAve'] = int(self.paramDict.get('numGradAvgIterations', 1))   
    self.localLocalInputAndChecks(xmlNode)

  @abc.abstractmethod
  def localLocalInputAndChecks(self, xmlNode):
    pass # to be overwritten by subclass
  
  def localInitialize(self,solutionExport=None):   
    self.gradDict['pertPoints'] = {}
        
    #specializing the self.localInitialize() to account for adaptive sampling
    if solutionExport != None : self.localLocalInitialize(solutionExport=solutionExport)
    else                      : self.localLocalInitialize()
  
  @abc.abstractmethod
  def localLocalInitialize(self, solutionExport = None):
    pass

  def localStillReady(self,ready, convergence = False): #,lastOutput=None
    """
      Determines if sampler is prepared to provide another input.  If not, and
      if jobHandler is finished, this will end sampling.
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    
    
    if ready == False:                                                                              
      return ready # Just return if we exceed the max iterations or converges...
    if self.mdlEvalHist == None and self.counter['perturbation'] < self.gradDict['pertNeeded']:          
      return ready # Just return if we just initialize
    elif self.mdlEvalHist.isItEmpty() and self.counter['perturbation'] < self.gradDict['pertNeeded']:    
      return ready # Just return if we just initialize
    
    ready = self.localLocalStillReady(ready, convergence)
    
    return ready
    
    @abc.abstractmethod
    def localLocalStillReady(self, ready, convergence = False):
      pass
      

  def localGenerateInput(self,model,oldInput):
    """
      This class need to be overwritten since it is here that the magic of the sampler happens.
      After this method call the self.inputInfo should be ready to be sent to the model
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, localGenerateInput, list, list containing the new inputs -in reality it is the model that return this the Sampler generate the value to be placed in the input the model
    """      
     
    if self.counter['mdlEval'] > 1:
      if self.counter['perturbation'] < self.gradDict['pertNeeded']:
        self.readyVarsUpdate = False
        self.counter['perturbation'] += 1
      else: # Got enough perturbation
        self.readyVarsUpdate = True  
        self.counter['perturbation'] = 0
        self.counter['varsUpdate'] += 1
    else:
      self.readyVarsUpdate = False
    
      
      
    self.localLocalGenerateInput(model,oldInput)    

  @abc.abstractmethod
  def localLocalGenerateInput(self,model,oldInput):
    pass

  
  def evaluateGradient(self, optVarsValues):
    """
    optVarsValues are the perturbed parameter values for gradient estimation
    optVarsValues should have the form of {pertIndex: {varName: [varValue1 varValue2]}}
    Therefore, each optVarsValues[pertIndex] should return a dict of variable values that is sufficient for gradient 
    evaluation for at least one variable (depending on specific optimization algorithm) 
    """
    gradArray = {}
    for var in self.optVars:
      gradArray[var] = np.ndarray((0,0))
        
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
      
  @abc.abstractmethod
  def localEvaluateGradient(self, optVarsValues, gradient = None):
    return gradient
    






