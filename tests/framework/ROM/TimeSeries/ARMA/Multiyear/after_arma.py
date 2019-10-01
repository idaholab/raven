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
# All answers are 1.0

import numpy as np

def run(raven, Input):
  """
    RAVEN hook, simple test model
    @ In, raven, object, object with variable data
    @ In, Input, dict, dictionary with info
    @ Out, None
  """
  indexOrder = raven._indexMap[0]['Speed']
  timeDimIndex = indexOrder.index('Time')
  raven.yearly_aggregate = raven.Speed.mean(axis=timeDimIndex)


