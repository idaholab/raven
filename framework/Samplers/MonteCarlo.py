"""
  This module contains the Monte Carlo sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from crisr
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

stochasticEnv = distribution1D.DistributionContainer.instance()

class MonteCarlo(ForwardSampler):
  """
    MONTE CARLO Sampler
  """
  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    ForwardSampler.__init__(self)
    self.printTag = 'SAMPLER MONTECARLO'

  def localInputAndChecks(self,xmlNode):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ Out, None
    """
    ForwardSampler.readSamplerInit(self,xmlNode)
    if xmlNode.find('samplerInit')!= None:
      if xmlNode.find('samplerInit').find('limit')!= None:
        try              : self.limit = int(xmlNode.find('samplerInit').find('limit').text)
        except ValueError: self.raiseAnError(IOError,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
      else: self.raiseAnError(IOError,self,'Monte Carlo sampler '+self.name+' needs the limit block (number of samples) in the samplerInit block')      
      if xmlNode.find('samplerInit').find('samplingType')!= None:
        self.samplingType = xmlNode.find('samplerInit').find('samplingType').text
      else:
          self.samplingType = None
    else: self.raiseAnError(IOError,self,'Monte Carlo sampler '+self.name+' needs the samplerInit block')

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
    # create values dictionary
    for key in self.distDict:
      # check if the key is a comma separated list of strings
      # in this case, the user wants to sample the comma separated variables with the same sampled value => link the value to all comma separated variables

      dim    = self.variables2distributionsMapping[key]['dim']
      totDim = self.variables2distributionsMapping[key]['totDim']
      dist   = self.variables2distributionsMapping[key]['name']
      reducedDim = self.variables2distributionsMapping[key]['reducedDim']
      weight = 1.0
      if totDim == 1:
        for var in self.distributions2variablesMapping[dist]:
          varID  = utils.first(var.keys())          
          if self.samplingType == 'uniform':
            distData = self.distDict[key].getCrowDistDict()
            if ('xMin' not in distData.keys()) or ('xMax' not in distData.keys()):
              self.raiseAnError(IOError,"In the Monte-Carlo sampler a uniform sampling type has been chosen; however, one or more distributions have not specified either the lowerBound or the upperBound")
            lower = distData['xMin']
            upper = distData['xMax'] 
            rvsnum = lower + (upper - lower) * Distributions.random() 
            epsilon = (upper-lower)/self.limit
            midPlusCDF  = self.distDict[key].cdf(rvsnum + epsilon)
            midMinusCDF = self.distDict[key].cdf(rvsnum - epsilon)
            weight *= midPlusCDF - midMinusCDF
          else:
            rvsnum = self.distDict[key].rvs()
          self.inputInfo['SampledVarsPb'][key] = self.distDict[key].pdf(rvsnum)
          for kkey in varID.strip().split(','):
            self.values[kkey] = np.atleast_1d(rvsnum)[0]
      elif totDim > 1:
        if reducedDim == 1:
          if self.samplingType is None:
            rvsnum = self.distDict[key].rvs()
            coordinate = np.atleast_1d(rvsnum).tolist()
          else:
            coordinate = np.zeros(totDim)
            for i in range(totDim):
              lower = self.distDict[key].returnLowerBound(i)
              upper = self.distDict[key].returnUpperBound(i)
              coordinate[i] = lower + (upper - lower) * Distributions.random()        
          if reducedDim > len(coordinate): self.raiseAnError(IOError,"The dimension defined for variables drew from the multivariate normal distribution is exceeded by the dimension used in Distribution (MultivariateNormal) ")
          probabilityValue = self.distDict[key].pdf(coordinate)
          self.inputInfo['SampledVarsPb'][key] = probabilityValue
          for var in self.distributions2variablesMapping[dist]:
            varID  = utils.first(var.keys())
            varDim = var[varID]
            for kkey in varID.strip().split(','):
              self.values[kkey] = np.atleast_1d(rvsnum)[varDim-1]
      else:
        self.raiseAnError(IOError,"Total dimension for given distribution should be >= 1")

    if len(self.inputInfo['SampledVarsPb'].keys()) > 0:
      self.inputInfo['PointProbability'  ]  = reduce(mul, self.inputInfo['SampledVarsPb'].values())
      if self.samplingType == 'uniform':
        self.inputInfo['ProbabilityWeight'  ] = weight
      else:
        self.inputInfo['ProbabilityWeight'  ] = 1.0
    self.inputInfo['SamplerType'] = 'MC'

  def _localHandleFailedRuns(self,failedRuns):
    """
      Specialized method for samplers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns)>0: self.raiseADebug('  Continuing with reduced-size Monte-Carlo sampling.')



