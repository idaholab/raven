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
  GradientApproximators are a tool for the GradientDescent Optimizer.
"""

from __future__ import absolute_import

from .AcceptanceCondition import AcceptanceCondition
from .Strict import Strict

from .Factory import knownTypes
from .Factory import returnInstance
from .Factory import returnClass
