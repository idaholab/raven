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
  parentSelector is a tool for the Genetic Algorithm Optimizer.
"""

from __future__ import absolute_import

# These lines ensure that we do not have to do something like:
# 'from Optimizers.Optimizer import Sampler' outside of this submodule

# TODO
from . import parentSelectors
from .RouletteWheel import RouletteWheel
from .SUS import SUS
from .Tournament import Tournament
from .Rank import Rank
from .Random import Random

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass
