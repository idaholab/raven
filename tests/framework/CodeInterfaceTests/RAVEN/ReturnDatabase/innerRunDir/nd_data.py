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
    Run method.
    @ In, raven, object, RAVEN object
    @ In, inputs, dict, input dictionary
    @ Out, None
  """
  # inputs: a, b, c
  # outputs: d, e, f
  # indices: d(), e(x), f(x, y)
  a = raven.a
  b = raven.b
  c = raven.c

  nx = 5
  ny = 3
  x = np.arange(nx) * 0.1
  y = np.arange(ny) * 10

  d = a*a
  e = x * b
  f = np.arange(nx*ny).reshape(nx, ny) * c

  # save
  raven.x = x
  raven.y = y
  raven.d = d
  raven.e = e
  raven.f = f
  raven._indexMap = {'e': ['x'],
                     'f': ['x', 'y']
                    }

