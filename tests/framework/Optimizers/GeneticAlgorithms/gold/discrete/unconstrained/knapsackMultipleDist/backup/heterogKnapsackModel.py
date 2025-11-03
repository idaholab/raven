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

def run(self,Input):
  """
    Evaluate a simple function.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  projValue = np.array([1,3,4])
  projCost  = np.array([3,2,5])
  coeff     = np.array([4,14,104])
  knapsackCapacity = 8

  projPlan = np.array([Input['proj1'],Input['proj2'],Input['proj3']])
  self.planValue = 0
  for n in range(0,3):
    if projPlan[n] > 0:
      test = knapsackCapacity - projCost[n]
      if test >= 0:
        knapsackCapacity = knapsackCapacity - projCost[n]
        self.planValue = self.planValue + projValue[n]*projPlan[n]/coeff[n]
      else:
        knapsackCapacity = knapsackCapacity - projCost[n]
        self.planValue = self.planValue - projValue[n]*projPlan[n]/coeff[n]

  if knapsackCapacity >= 0:
    self.validPlan =  0.
  else:
    self.validPlan = 1.


