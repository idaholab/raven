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
from .Processor import Processor
from .BasicStatistics import BasicStatistics
from .ComparisonStatistics import ComparisonStatistics
from .ExternalProcessor import ExternalProcessor
from .ImportanceRank import ImportanceRank
from .InterfacedProcessor import InterfacedProcessor
from .LimitSurface import LimitSurface
from .LimitSurfaceIntegral import LimitSurfaceIntegral
from .RavenOutput import RavenOutput
from .SafestPoint import SafestPoint

from .TopologicalDecomposition import TopologicalDecomposition
from .DataMining import DataMining

try:
  from .TopologicalDecomposition import QTopologicalDecomposition
  from .DataMining import QDataMining
except ImportError:
  pass
## [ Add new class here ]
################################################################################
## Alternatively, to fully automate this file:
# from Processors import *
################################################################################


"""
 Interface Dictionary (factory) (private)
"""

# This machinery will automatically populate the "knownTypes" given the
# imports defined above.
__base = 'Processor'
__interFaceDict = {}

for classObj in eval(__base).__subclasses__():
  key = classObj.__name__
  __interFaceDict[key] = classObj

## Adding aliases for certain classes that are exposed to the user.
__interFaceDict['InterfacedPostProcessor'] = InterfacedProcessor
__interFaceDict['External'] = ExternalProcessor

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
    @ Out, returnInstance, instance, instance of Processor subclass, a subclass object constructed with no arguments
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