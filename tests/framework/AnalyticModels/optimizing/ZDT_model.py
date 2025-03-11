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

# @author: Junyung Kim (@JunyungKim-INL) and Mohammad Abdo (@Jimmy-INL)

import math

def evaluate(Inputs):
  """
    @In, Inputs, dictionary with the input
    @Out, obj1, obj2, floating point arrays with the objective function.
  """
  Sum = 0
  for _,var in enumerate(Inputs.keys()):
    # write the objective function here
    if (var == 'x1') :
      obj1 = Inputs[var]
    else:
      Sum += Inputs[var]
  g = 1 + (9/(len(Inputs.keys())-1)*Sum )
  h = 1 - math.sqrt(obj1/g)
  obj2 = g * h
  return obj1[:], obj2[:]

def run(self,Inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, Inputs, dict, additional inputs
    @ Out, None
  """
  self.obj1,self.obj2 = evaluate(Inputs)
