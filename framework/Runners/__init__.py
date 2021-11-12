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
  The Runners module includes the different ways of parallelizing the MultiRuns
  of RAVEN.

  Created on September 12, 2016
  @author: maljdp
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from Runners.Runner import Runner' outside
## of this submodule
from .Runner import Runner
from .InternalRunner import InternalRunner
from .SharedMemoryRunner import SharedMemoryRunner
from .DistributedMemoryRunner import DistributedMemoryRunner
from .PassthroughRunner import PassthroughRunner
from .Error import Error

from .Factory import factory
