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
distribution1D = utils.find_distribution1D()
#Internal Modules End--------------------------------------------------------------------------------

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
    else: self.raiseAnError(IOError,self,'Monte Carlo sampler '+self.name+' needs the samplerInit block')

  def localInitialize(self):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, None
      @ Out, None
    """
    if self.restartData:
      self.counter+=len(self.restartData)
      self.raiseAMessage('Number of points from restart: %i' %self.counter)
      self.raiseAMessage('Number of points needed:       %i' %(self.limit-self.counter))

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

      if totDim == 1:
        for var in self.distributions2variablesMapping[dist]:
          varID  = utils.first(var.keys())
          rvsnum = self.distDict[key].rvs()
          self.inputInfo['SampledVarsPb'][key] = self.distDict[key].pdf(rvsnum)
          for kkey in varID.strip().split(','):
            self.values[kkey] = np.atleast_1d(rvsnum)[0]
      elif totDim > 1:
        if reducedDim == 1:
          rvsnum = self.distDict[key].rvs()
          coordinate = np.atleast_1d(rvsnum).tolist()
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
      self.inputInfo['PointProbability'  ] = reduce(mul, self.inputInfo['SampledVarsPb'].values())
      #self.inputInfo['ProbabilityWeight' ] = 1.0 #MC weight is 1/N => weight is one
    self.inputInfo['SamplerType'] = 'MC'

  def _localHandleFailedRuns(self,failedRuns):
    """
      Specialized method for samplers to handle failed runs.  Defaults to failing runs.
      @ In, failedRuns, list, list of JobHandler.ExternalRunner objects
      @ Out, None
    """
    if len(failedRuns)>0: self.raiseADebug('  Continuing with reduced-size Monte Carlo sampling.')



