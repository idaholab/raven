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
Created on July 10, 2013

@author: alfoa
"""

#for future compatibility with Python 3-----------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
if not 'xrange' in dir(__builtins__):
  xrange = range
#End compatibility block for Python 3-------------------------------------------

################################################################################
from utils import utils
from .PostProcessor import PostProcessor

from .Metric import Metric
from .DataMining import DataMining
from .ETImporter import ETImporter
from .SafestPoint import SafestPoint
from .LimitSurface import LimitSurface
from .ValueDuration import ValueDuration
from .ImportanceRank import ImportanceRank
from .BasicStatistics import BasicStatistics
from .CrossValidation import CrossValidation
from .LimitSurfaceIntegral import LimitSurfaceIntegral
from .ExternalPostProcessor import ExternalPostProcessor
from .InterfacedPostProcessor import InterfacedPostProcessor
from .TopologicalDecomposition import TopologicalDecomposition
from .ComparisonStatisticsModule import ComparisonStatistics
# from .RavenOutput import RavenOutput # deprecated for now

## These utilize the optional prequisite library PySide, so don't error if they
## do not import appropriately.
try:
  from .TopologicalDecomposition import QTopologicalDecomposition
  from .DataMining import QDataMining
except ImportError:
  pass

## [ Add new class here ]

################################################################################
## Alternatively, to fully automate this file:
# from PostProcessors import *
################################################################################


"""
 Interface Dictionary (factory) (private)
"""

# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'PostProcessor'
__interFaceDict = {}

for classObj in utils.getAllSubclasses(eval(__base)):
  key = classObj.__name__
  __interFaceDict[key] = classObj

## Adding aliases for certain classes that are exposed to the user.
__interFaceDict['External'] = ExternalPostProcessor
try:
  __interFaceDict['TopologicalDecomposition' ] = QTopologicalDecomposition
  __interFaceDict['DataMining'               ] = QDataMining
except NameError:
  ## The correct names should already be used for these classes otherwise
  pass

def knownTypes():
  """
    Returns a list of strings that define the types of instantiable objects for
    this base factory.
    @ In, None
    @ Out, knownTypes, list, list of known types
  """
  return __interFaceDict.keys()

def returnInstance(Type,caller):
  """
    Attempts to create and return an instance of a particular type of object
    available to this factory.
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the instance (used for error/debug messaging).
    @ Out, returnInstance, instance, instance of PostProcessor subclass, a subclass object constructed with no arguments
  """
  try:
    return __interFaceDict[Type](caller.messageHandler)
  except KeyError:
    # print(eval(__base).__subclasses__())
    # print(__interfaceDict.keys())
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)

def returnClass(Type,caller):
  """
    Attempts to return a particular class type available to this factory.
    @ In, Type, string, string should be one of the knownTypes.
    @ In, caller, instance, the object requesting the class (used for error/debug messaging).
    @ Out, returnClass, class, reference to the subclass
  """
  try:
    return __interFaceDict[Type]
  except KeyError:
    caller.raiseAnError(NameError,__name__+': unknown '+__base+' type '+Type)
