"""
  This module contains the Ensemble Forward sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
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
from .ForwardSampler import ForwardSampler
import utils
import mathUtils
from BaseClasses import BaseType
from Assembler import Assembler
import Distributions
import DataObjects
import SupervisedLearning
import pyDOE as doe
import Quadratures
import OrthoPolynomials
import IndexSets
import Models
import PostProcessors
import MessageHandler
import GridEntities
distribution1D = utils.find_distribution1D()
#Internal Modules End--------------------------------------------------------------------------------

class EnsembleForwardSampler(ForwardSampler):
  """
    Ensemble Forward sampler. This sampler is aimed to combine Forward Sampling strategies
  """
  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    ForwardSampler.__init__(self)
    self.acceptableSamplers   = ['MonteCarlo','Stratified','Grid','FactorialDesign','ResponseSurfaceDesign','CustomSampler']
    self.printTag             = 'SAMPLER EnsembleForward'
    self.instanciatedSamplers = {}
    self.samplersCombinations = {}

  def localInputAndChecks(self,xmlNode):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ Out, None
    """
    ForwardSampler.readSamplerInit(self,xmlNode)
    from .Factory import returnInstance
    for child in xmlNode:
      if child.tag in self.acceptableSamplers:
        child.attrib['name'] = child.tag
        self.instanciatedSamplers[child.tag] = returnInstance(child.tag,self)
        #FIXME the variableGroups needs to be fixed
        self.instanciatedSamplers[child.tag].readXML(child,self.messageHandler,variableGroups={},globalAttributes=self.globalAttributes)
      elif child.tag in knownTypes():
        self.raiseAnError(IOError,"Sampling strategy "+child.tag+" is not usable in "+self.type+" Sampler. Available are "+",".join(self.acceptableSamplers))
      else:
        self.raiseAnError(IOError,"XML node "+ child.tag + " unknown. Check the Manual!")

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented here because this Sampler requests special objects
      @ In, None
      @ Out, needDict, dict, dictionary of objects needed
    """
    needDict = ForwardSampler._localWhatDoINeed(self)
    for combSampler in self.instanciatedSamplers.values():
      preNeedDict = combSampler.whatDoINeed()
      for key,value in preNeedDict.items():
        if key not in needDict.keys(): needDict[key] = []
        needDict[key] = needDict[key] + value
    return needDict

  def _localGenerateAssembler(self,initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    availableDist = initDict['Distributions']
    availableFunc = initDict['Functions']
    for combSampler in self.instanciatedSamplers.values():
      if combSampler.type != 'CustomSampler': combSampler._generateDistributions(availableDist,availableFunc)
      combSampler._localGenerateAssembler(initDict)
    self.raiseADebug("Distributions initialized!")

  def localInitialize(self):
    """
      Initialize the EnsembleForwardSampler sampler. It calls the localInitialize method of all the Samplers defined in this input
      @ In, None
      @ Out, None
    """
    self.limit = 1
    cnt = 0
    lowerBounds, upperBounds = {}, {}
    for samplingStrategy in self.instanciatedSamplers.keys():
      self.instanciatedSamplers[samplingStrategy].initialize(externalSeeding=self.initSeed,solutionExport=None)
      self.samplersCombinations[samplingStrategy] = []
      self.limit *= self.instanciatedSamplers[samplingStrategy].limit
      lowerBounds[samplingStrategy],upperBounds[samplingStrategy] = 0, self.instanciatedSamplers[samplingStrategy].limit
      while self.instanciatedSamplers[samplingStrategy].amIreadyToProvideAnInput():
        self.instanciatedSamplers[samplingStrategy].counter +=1
        self.instanciatedSamplers[samplingStrategy].localGenerateInput(None,None)
        self.instanciatedSamplers[samplingStrategy].inputInfo['prefix'] = self.instanciatedSamplers[samplingStrategy].counter
        self.samplersCombinations[samplingStrategy].append(copy.deepcopy(self.instanciatedSamplers[samplingStrategy].inputInfo))
      cnt+=1
    self.raiseAMessage('Number of Combined Samples are ' + str(self.limit) + '!')
    # create a grid of combinations (no tensor)
    self.gridEnsemble = GridEntities.GridEntity(self.messageHandler)
    initDict = {'dimensionNames':self.instanciatedSamplers.keys(),
                'stepLength':dict.fromkeys(self.instanciatedSamplers.keys(),[1]),
                'lowerBounds':lowerBounds,
                'upperBounds':upperBounds,
                'computeCells':False,
                'constructTensor':False,
                'excludeBounds':{'lowerBounds':False,'upperBounds':True}}
    self.gridEnsemble.initialize(initDict)

  def localGenerateInput(self,model,myInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, myInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    index = self.gridEnsemble.returnPointAndAdvanceIterator(returnDict = True)
    coordinate = []
    for samplingStrategy in self.instanciatedSamplers.keys(): coordinate.append(self.samplersCombinations[samplingStrategy][int(index[samplingStrategy])])
    for combination in coordinate:
      for key in combination.keys():
        if key not in self.inputInfo.keys(): self.inputInfo[key] = combination[key]
        else:
          if type(self.inputInfo[key]).__name__ == 'dict':
            self.inputInfo[key].update(combination[key])
    self.inputInfo['PointProbability'] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    self.inputInfo['ProbabilityWeight' ] = 1.0
    for key in self.inputInfo.keys():
      if key.startswith('ProbabilityWeight-'):self.inputInfo['ProbabilityWeight' ] *= self.inputInfo[key]
    self.inputInfo['SamplerType'] = 'EnsembleForward'




