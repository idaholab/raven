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
  Two-objective test function (ZDT1 with 2 variables) for the LogicalModel-under-Optimizer test.
  zdtA and zdtB are the same function in two modules so the LogicalModel has two valid sub-models;
  the result must not depend on which one the ControlFunction picks.
"""
import numpy as np

def run(self, inputDict):
  """
    Method required by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, inputDict, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  x1 = float(np.ravel(self.x1)[0])
  x2 = float(np.ravel(self.x2)[0])
  g = 1.0 + 9.0 * x2
  self.obj1 = x1
  self.obj2 = g * (1.0 - np.sqrt(x1 / g))
  self.submodel = 1.0
