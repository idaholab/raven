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

# @author: Mohammad Abdo (@Jimmy-INL)

def evaluate(Inputs):
  """
  This function evaluates the sum of weighted inputs and calculates two local sums based on the indices.

  @ In, Inputs, dict, dictionary of variables
  @ Out, Sum, float, the sum of weighted inputs.
  @ Out, LocalSum1, float, the sum of weighted inputs for indices 0 and 1.
  @ Out, LocalSum2, float, the sum of weighted inputs for indices 2 and 3.
  """
  Sum = 0
  LocalSum1 = 0
  LocalSum2 = 0
  for ind, var in enumerate(Inputs.keys()):
    # write the objective function here
    Sum += (ind + 1) * Inputs[var]
    if (ind == 0) or (ind == 1):
      LocalSum1 += (ind + 1) * Inputs[var]
    if (ind == 2) or (ind == 3):
      LocalSum2 += (ind + 1) * Inputs[var]
  return Sum[:], LocalSum1[:], LocalSum2[:]

def run(self,Inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, Inputs, dict, additional inputs
    @ Out, None
  """
  self.obj1,self.obj2,self.obj3 = evaluate(Inputs)
