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
  Mixed categorical two-objective test problem. "family" selects which variables are active:
    family 0: continuous x in [0, 1]; obj1 = x, obj2 = 1 - sqrt(x) (a continuous Pareto front)
    family 1: integer n in {0, 1, 2}; three designs that dominate most of the family-0 front
  The inactive variables are carried by the GA but do not affect the design, so copies of a family-1 design differ in
  x. This is the setting in which, without duplicate elimination, copies of a few family-1 designs fill the population.
"""
import numpy as np

def run(self, inputDict):
  """
    Method required by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, inputDict, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  family = int(round(float(np.ravel(self.family)[0])))
  if family == 0:
    x = float(np.ravel(self.x)[0])
    self.obj1 = x
    self.obj2 = 1.0 - np.sqrt(x)
  else:
    n = int(round(float(np.ravel(self.n)[0])))
    self.obj1 = (0.15, 0.40, 0.70)[n]
    self.obj2 = (0.45, 0.20, 0.05)[n]
