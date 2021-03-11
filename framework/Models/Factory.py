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
Factory for generating the instances of the  Models Module
"""
from utils import utils

from .Model         import Model
from .Dummy         import Dummy
from .ROM           import ROM
from .ExternalModel import ExternalModel
from .Code          import Code
from .EnsembleModel import EnsembleModel
from .HybridModels  import HybridModel
from .HybridModels  import LogicalModel

#### PostProcessors
from .PostProcessors import PostProcessor
from .PostProcessors import FTImporter
from .PostProcessors import BasicStatistics
from .PostProcessors import LimitSurface
from .PostProcessors import Metric
from .PostProcessors import ETImporter
from .PostProcessors.DataMining import DataMining
from .PostProcessors import SafestPoint
from .PostProcessors import ValueDuration
from .PostProcessors import SampleSelector
from .PostProcessors import ImportanceRank
from .PostProcessors import CrossValidation
from .PostProcessors import LimitSurfaceIntegral
from .PostProcessors import FastFourierTransform
from .PostProcessors.ExternalPostProcessor import ExternalPostProcessor
from .PostProcessors import InterfacedPostProcessor
from .PostProcessors.TopologicalDecomposition import TopologicalDecomposition
from .PostProcessors import DataClassifier
from .PostProcessors.ComparisonStatisticsModule import ComparisonStatistics
from .PostProcessors import RealizationAverager
from .PostProcessors.ParetoFrontierPostProcessor import ParetoFrontier
from .PostProcessors.MCSimporter import MCSImporter
from .PostProcessors import EconomicRatio
# from .PostProcessors import RavenOutput # deprecated for now

## These utilize the optional prequisite library PySide, so don't error if they
## do not import appropriately.
try:
  from .PostProcessors.TopologicalDecomposition import QTopologicalDecomposition
  from .PostProcessors.DataMining import QDataMining
  renaming = {'QTopologicalDecomposition': 'TopologicalDecomposition',
              'QDataMining': 'DataMining'}
except ImportError:
  renaming = {}

__base = 'Model'
__interFaceDict = {}


for classObj in utils.getAllSubclasses(eval(__base)):
  key = classObj.__name__
  key = renaming.get(key, key) # get alias if provided, else use key
  __interFaceDict[key] = classObj

# NOTE the following causes QDataMining to be entered into the __interFaceDict a second time,
#      which somehow passed the test machines but seg faulted on my machine. I don't think we need
#      it in there twice, anyway. - talbpaul, 2021-3-11
# try:
#   __interFaceDict['TopologicalDecomposition' ] = QTopologicalDecomposition
#   __interFaceDict['DataMining'               ] = QDataMining
# except NameError:
#   ## The correct names should already be used for these classes otherwise
#   pass

# #here the class methods are called to fill the information about the usage of the classes
for classType in __interFaceDict.values():
  classType.generateValidateDict()
  classType.specializeValidateDict()

## Adding aliases for certain classes that are exposed to the user.
__interFaceDict['External'] = ExternalPostProcessor

def knownTypes():
  """
    Return the known types
    @ In, None
    @ Out, knownTypes, list, list of known types
  """
  return __interFaceDict.keys()

needsRunInfo = True

def returnInstance(Type,runInfoDict,caller):
  """
    function used to generate a Model class
    @ In, Type, string, Model type
    @ Out, returnInstance, instance, Instance of the Specialized Model class
  """
  try:
    return __interFaceDict[Type](runInfoDict)
  except KeyError:
    availableClasses = ','.join(__interFaceDict.keys())
    caller.raiseAnError(NameError,
      'Requested {}, i.e. "{}", is not recognized (Available options: {})'.format(__base, Type, availableClasses))

def validate(className,role,what,caller):
  """
    This is the general interface for the validation of a model usage
    @ In, className, string, the name of the class
    @ In, role, string, the role assumed in the Step
    @ In, what, string, type of object
    @ In, caller, instance, the instance of the caller
    @ Out, None
  """
  if className in __interFaceDict:
    return __interFaceDict[className].localValidateMethod(role,what)
  else:
    caller.raiseAnError(IOError, 'The model "{}" is not registered for class "{}"'.format(className, __base))
