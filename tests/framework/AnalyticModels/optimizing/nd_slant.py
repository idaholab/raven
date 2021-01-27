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

# takes parameters
# returns sum of parameters, resulting in tapering function to a corner
# bounded to [0, 1] in all dimensions results in an opt at {1}_N with value 0
import numpy as np

def run(raven, Inputs):
  """
    External model to evaluate
    @ In, raven, object, container from RAVEN to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  #vs = list(Inputs.values())
  for n in ['x', 'y']:
    v = getattr(raven, n)
    if v < 0 or v > 1:
      raise RuntimeError(f'Value out of domain! {n} = {v}')
  raven.ans = main(raven.x, raven.y)

def main(x, y):
  """
    Evaluation.
    @ In, x, float, value
    @ In, y, float, value
    @ Out, ans, float, value
  """
  val = 1.0 - (x+y)/2.0
  return val
