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

def run(raven, inputs):
  """
    Evaluate.
    @ In, raven, object, raven self
    @ In, inputs, dict, key-value pairs
    @ Out, None
  """
  if not constrain(raven):
    raise RuntimeError(f'Out of bounds: ({raven.x}, {raven.y})!')
  raven.ans = main(raven.x, raven.y)

def main(x, y):
  """
    Evaluate.
    @ In, x, float, value
    @ In, y, float, value
    @ Out, main, float, value
  """
  return (x + 0.1)**2 + (y + 0.1) **2


def constrain(raven):
  """
    Constrain.
    @ In, raven, object, raven self
    @ Out, explicitConstrain, point ok or not?
  """
  x = raven.x
  y = raven.y
  # circle
  res = np.sqrt(x**2 + y**2)
  if res <= 0.2:
    return False
  # rectangle
  if 0.25 < x < 0.75 and 0 < y < 1:
    return False
  return True

def implicitConstrain(raven):
  """
    Implicit constrain.
    @ In, raven, object, raven self
    @ Out, implicitConstrain, point ok or not?
  """
  x = raven.x
  ans = raven.ans
  if x+ans <= -0.2:
    return False
  else:
    return True
