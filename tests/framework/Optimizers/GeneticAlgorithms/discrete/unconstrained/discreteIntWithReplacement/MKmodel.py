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

def run(self,inputs):
  """
    Evaluate a simple function.
    @ In, inputs, dictionary of variables
    @ Out, None
  """
  projValue = np.array([1,3,4,2,1,2,3,4,2,4])
  projCost  = np.array([3,2,5,3,2,4,6,3,5,3])
  knapsackCapacities = np.array([4,5,4,5,5])

  projPlan = np.array([inputs['proj1'],inputs['proj2'],inputs['proj3'],inputs['proj4'],inputs['proj5'],inputs['proj6'],inputs['proj7'],inputs['proj8'],inputs['proj9'],inputs['proj10']])
  self.planValue = 0
  for n in range(0,10):
    if projPlan[n]>0:
      test = knapsackCapacities[int(projPlan[n])-1] - projCost[n]
      if test>=0:
        knapsackCapacities[int(projPlan[n])-1] = knapsackCapacities[int(projPlan[n])-1] - projCost[n]
        self.planValue = self.planValue + projValue[n]
      else:
        knapsackCapacities[int(projPlan[n])-1] = knapsackCapacities[int(projPlan[n])-1] - projCost[n]
        self.planValue = self.planValue - projValue[n]

  if (knapsackCapacities>=0).all():
    self.validPlan =  0.
  else:
    self.validPlan = 1.

