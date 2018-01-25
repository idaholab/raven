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

from __future__ import absolute_import

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
from .HybridModel   import HybridModel

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
           'HybridModel']
