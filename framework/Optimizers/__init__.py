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
  Optimizers are a class of Samplers that specialize the Adaptive Samplers.

  Reworked 2020-01
  @author: talbpaul
"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from Optimizers.Optimizer import Sampler' outside of this submodule

# TODO
from .Optimizer import Optimizer
from .GradientDescent import GradientDescent
#from .SPSA import SPSA

# TODO
from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass



# # We should not really need this as we do not use wildcard imports
# __all__ = ['Optimizer','GradientBasedOptimizer','SPSA']
