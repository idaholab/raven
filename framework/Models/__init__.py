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
  The Models module for building, running, and simulating things in RAVEN.

  Created on May 9, 2017
  @author: maljdp
"""

## These lines ensure that we do not have to do something like:
## 'from Models.Model import Model' outside
## of this submodule
from .Model import Model
from .Dummy import Dummy
from .ROM import ROM
from .ExternalModel import ExternalModel
from .Code import Code
from .EnsembleModel import EnsembleModel
from .PostProcessor import PostProcessor
from .HybridModels import HybridModel
from .HybridModels import LogicalModel

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

additionalModules = []
## These utilize the optional prequisite library PySide, so don't error if they
## do not import appropriately.
try:
  from .TopologicalDecomposition import QTopologicalDecomposition
  from .DataMining import QDataMining
  additionalModules.append(QTopologicalDecomposition)
  additionalModules.append(QDataMining)
except ImportError:
  ## User most likely does not have PySide installed and working
  pass


## [ Add new class here ]

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import needsRunInfo
from .Factory import validate

# We should not really need this as we do not use wildcard imports
__all__ = ['Model',
           'Dummy',
           'ROM',
           'ExternalModel',
           'Code',
           'EnsembleModel',
           'PostProcessor',
           'HybridModel',
           'LogicalModel',
           'BasicStatistics',
           'ComparisonStatistics',
           'ExternalPostProcessor',
           'ImportanceRank',
           'InterfacedPostProcessor',
           'LimitSurface',
           'LimitSurfaceIntegral',
           'SafestPoint',
           'TopologicalDecomposition',
           'DataMining',
           'Metric',
           'CrossValidation',
           'ValueDuration',
           'FastFourierTransform',
           'FTImporter',
           'DataClassifier',
           'SampleSelector',
           'ETImporter',
           'RealizationAverager',
           'ParetoFrontier']+ additionalModules
#           'RavenOutput', # deprecated for now
