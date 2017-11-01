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
  The PostProcessors module includes the different ways of manipulating data in
  RAVEN.

  Created on May 9, 2017
  @author: maljdp
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from PostProcessors.PostProcessor import PostProcessor' outside
## of this submodule
from .PostProcessor import PostProcessor
from .BasicStatistics import BasicStatistics
from .ComparisonStatisticsModule import ComparisonStatistics
from .ExternalPostProcessor import ExternalPostProcessor
from .ImportanceRank import ImportanceRank
from .InterfacedPostProcessor import InterfacedPostProcessor
from .LimitSurface import LimitSurface
from .LimitSurfaceIntegral import LimitSurfaceIntegral
from .RavenOutput import RavenOutput
from .SafestPoint import SafestPoint

from .TopologicalDecomposition import TopologicalDecomposition
from .DataMining import DataMining
from .Metric import Metric
from .CrossValidation import CrossValidation

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

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['PostProcessor',
           'BasicStatistics',
           'ComparisonStatistics',
           'ExternalPostProcessor',
           'ImportanceRank',
           'InterfacedPostProcessor',
           'LimitSurface',
           'LimitSurfaceIntegral',
           'RavenOutput',
           'SafestPoint',
           'TopologicalDecomposition',
           'DataMining',
           'Metric',
           'CrossValidation',
           'ETimporter'] + additionalModules
