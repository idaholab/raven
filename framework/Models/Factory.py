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
from .PostProcessor import PostProcessor
from .HybridModels  import HybridModel
from .HybridModels  import LogicalModel

#### PostProcessors
from .PostProcessorBase import PostProcessorBase
from .Metric import Metric
from .ETImporter import ETImporter
from .DataMining import DataMining
from .SafestPoint import SafestPoint
from .LimitSurface import LimitSurface
from .ValueDuration import ValueDuration
from .SampleSelector import SampleSelector
from .ImportanceRank import ImportanceRank
from .CrossValidation import CrossValidation
from .BasicStatistics import BasicStatistics
from .LimitSurfaceIntegral import LimitSurfaceIntegral
from .FastFourierTransform import FastFourierTransform
from .ExternalPostProcessor import ExternalPostProcessor
from .InterfacedPostProcessor import InterfacedPostProcessor
from .TopologicalDecomposition import TopologicalDecomposition
from .FTImporter import FTImporter
from .DataClassifier import DataClassifier
from .ComparisonStatisticsModule import ComparisonStatistics
from .RealizationAverager import RealizationAverager
from .ParetoFrontierPostProcessor import ParetoFrontier
from .MCSimporter import MCSImporter
from .EconomicRatio import EconomicRatio
# from .RavenOutput import RavenOutput # deprecated for now

## These utilize the optional prequisite library PySide, so don't error if they
## do not import appropriately.
try:
  from .TopologicalDecomposition import QTopologicalDecomposition
  from .DataMining import QDataMining
except ImportError:
  pass

__base = 'Model'
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

#here the class methods are called to fill the information about the usage of the classes
for classType in __interFaceDict.values():
  classType.generateValidateDict()
  classType.specializeValidateDict()

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
