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
# takes input parameters x,y
# returns value in "ans"
# elliptic paraboloid function with a single global minimum
# optimal minimum at f(-0.5, 0.5) = 0
# parameter range is -2 <= x <= 2, -2 <= y <= 2
import numpy as np

def run(raven, Inputs):
  """
    External model to evaluates 10*(x+0.5)**2 + 10*(y-0.5)**2.
    @ In, raven, object, container from RAVEN to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  x = raven.x
  y = raven.y
  raven.ans = main(x, y)

def main(x, y):
  z = 10*(x+0.5)**2 + 10*(y-0.5)**2
  return z
