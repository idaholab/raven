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
    @ In, self, object, container
    @ Out, None
  """
  projValue = np.array([1,3,4,2,1,2,3,4,2,4])
  projCost  = np.array([3,2,5,3,2,4,6,3,5,3])
  knapsackCapacity = 8

  projPlan = np.array([Input['proj1'],Input['proj2'],Input['proj3'],Input['proj4'],Input['proj5'],Input['proj6'],Input['proj7'],Input['proj8'],Input['proj9'],Input['proj10']])
  self.planValue = 0
  for n in range(0,10):
    if projPlan[n]>0:
      test = knapsackCapacity - projCost[n]
      if test>=0:
        knapsackCapacity = knapsackCapacity - projCost[n]
        self.planValue = self.planValue + projValue[n]
      else:
        knapsackCapacity = knapsackCapacity - projCost[n]
        self.planValue = self.planValue - projValue[n]

      #knapsackCapacities[int(projPlan[n])-1] = knapsackCapacities[int(projPlan[n])-1] - projCost[n]
      #self.planValue = self.planValue + projValue[n]

  if knapsackCapacity>=0:
    self.validPlan =  0.
  else:
    self.validPlan = 1.

  #counterNeg = np.sum(knapsackCapacities<0, axis=0)
  #self.planValue = self.planValue - counterNeg * (10.)

