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

from EntityFactoryBase import EntityFactory
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
from .PostProcessors import HistorySetDelay
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

factory = EntityFactory('Model', needsRunInfo=True)
factory.registerAllSubtypes(Model, alias=renaming)

# #here the class methods are called to fill the information about the usage of the classes
for className in factory.knownTypes():
  classType = factory.returnClass(className)
  classType.generateValidateDict()
  classType.specializeValidateDict()

factory.registerType('External', ExternalPostProcessor)

def validate(className, role, what):
  """
    This is the general interface for the validation of a model usage
    @ In, className, string, the name of the class
    @ In, role, string, the role assumed in the Step
    @ In, what, string, type of object
    @ Out, None
  """
  if className in factory.knownTypes():
    return factory.returnClass(className).localValidateMethod(role, what)
  else:
    caller.raiseAnError(IOError, 'The model "{}" is not registered for class "{}"'.format(className, factory.name))
