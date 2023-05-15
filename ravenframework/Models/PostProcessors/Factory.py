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

from ...EntityFactoryBase import EntityFactory
from .PostProcessorInterface import PostProcessorInterface
from .PostProcessorReadyInterface import PostProcessorReadyInterface
from .BasicStatistics import BasicStatistics
from .LimitSurface import LimitSurface
from .Metric import Metric
from .DataMining import DataMining
from .SafestPoint import SafestPoint
from .ValueDuration import ValueDuration
from .SampleSelector import SampleSelector
from .ImportanceRank import ImportanceRank
from .CrossValidation import CrossValidation
from .LimitSurfaceIntegral import LimitSurfaceIntegral
from .FastFourierTransform import FastFourierTransform
from .ExternalPostProcessor import ExternalPostProcessor
from .TopologicalDecomposition import TopologicalDecomposition
from .ComparisonStatisticsModule import ComparisonStatistics
from .RealizationAverager import RealizationAverager
from .ParetoFrontierPostProcessor import ParetoFrontier
from .EconomicRatio import EconomicRatio
from .ValidationBase import ValidationBase
from .Validations import Probabilistic
from .Validations import PPDSS
from .Validations import PhysicsGuidedCoverageMapping
from .TSACharacterizer import TSACharacterizer

### PostProcessorFunctions (orig: InterfacedPostProcessor)
from .HistorySetDelay import HistorySetDelay
from .HS2PS import HS2PS
from .HStoPSOperator import HStoPSOperator
from .HistorySetSampling import HistorySetSampling
from .HistorySetSnapShot import HistorySetSnapShot
from .HistorySetSync import HistorySetSync
from .TypicalHistoryFromHistorySet import TypicalHistoryFromHistorySet
from .dataObjectLabelFilter import dataObjectLabelFilter
## These utilize the optional prequisite library PySide, so don't error if they
## do not import appropriately.
try:
  from .TopologicalDecomposition import QTopologicalDecomposition
  from .DataMining import QDataMining
  renaming = {'QTopologicalDecomposition': 'TopologicalDecomposition',
              'QDataMining': 'DataMining'}
except ImportError:
  renaming = {}

factory = EntityFactory('PostProcessorInterface', needsRunInfo=True)
factory.registerAllSubtypes(PostProcessorInterface, alias=renaming)

factory.registerType('External', ExternalPostProcessor)
