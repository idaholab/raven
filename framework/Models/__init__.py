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
from .HybridModels import HybridModel
from .HybridModels import LogicalModel

#### PostProcessors
from .PostProcessors import PostProcessor
from .PostProcessors import FTImporter

# from .PostProcessors import Metric
# from .PostProcessors import ETImporter
# from .PostProcessors import DataMining
# from .PostProcessors import SafestPoint
# from .PostProcessors import LimitSurface
# from .PostProcessors import ValueDuration
# from .PostProcessors import SampleSelector
# from .PostProcessors import ImportanceRank
# from .PostProcessors import CrossValidation
# from .PostProcessors import BasicStatistics
# from .PostProcessors import LimitSurfaceIntegral
# from .PostProcessors import FastFourierTransform
# from .PostProcessors import ExternalPostProcessor
# from .PostProcessors import InterfacedPostProcessor
# from .PostProcessors import TopologicalDecomposition
# from .PostProcessors import DataClassifier
# from .PostProcessors.ComparisonStatisticsModule import ComparisonStatistics
# from .PostProcessors import RealizationAverager
# from .PostProcessors.ParetoFrontierPostProcessor import ParetoFrontier
# from .PostProcessors.MCSimporter import MCSImporter
# from .PostProcessors import EconomicRatio
# # from .PostProcessors import RavenOutput # deprecated for now
#
# additionalModules = []
# ## These utilize the optional prequisite library PySide, so don't error if they
# ## do not import appropriately.
# try:
#   from .PostProcessors.TopologicalDecomposition import QTopologicalDecomposition
#   from .PostProcessors.DataMining import QDataMining
#   additionalModules.append(QTopologicalDecomposition)
#   additionalModules.append(QDataMining)
# except ImportError:
#   ## User most likely does not have PySide installed and working
#   pass


## [ Add new class here ]

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import needsRunInfo
from .Factory import validate

# # We should not really need this as we do not use wildcard imports
# __all__ = ['Model',
#            'Dummy',
#            'ROM',
#            'ExternalModel',
#            'Code',
#            'EnsembleModel',
#            'PostProcessor',
#            'HybridModel',
#            'LogicalModel',
#            'BasicStatistics',
#            'ComparisonStatistics',
#            'ExternalPostProcessor',
#            'ImportanceRank',
#            'InterfacedPostProcessor',
#            'LimitSurface',
#            'LimitSurfaceIntegral',
#            'SafestPoint',
#            'TopologicalDecomposition',
#            'DataMining',
#            'Metric',
#            'CrossValidation',
#            'ValueDuration',
#            'FastFourierTransform',
#            'FTImporter',
#            'DataClassifier',
#            'SampleSelector',
#            'ETImporter',
#            'RealizationAverager',
#            'ParetoFrontier']+ additionalModules
# #           'RavenOutput', # deprecated for now
