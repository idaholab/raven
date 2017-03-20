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
import numpy as np
from itertools import product


def eval(Y):
  l = float(len(Y))
  return np.exp( -sum(Y)/l)

def run(self,Input):
  Yvalue = []
  Yvalue.append(Input['x1'])
  Yvalue.append(Input['x2'])
  Yvalue.append(Input['x3'])
  Yvalue.append(Input['x4'])
  Yvalue.append(Input['x5'])
  self.ans = eval(Yvalue)
