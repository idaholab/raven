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
# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(1,1) = 0
# parameter range is -10 <= x,y <= 10
import numpy as np

def evaluate(x,y):
  """
    Evaluates Levi function.
    @ In, x, float, value
    @ In, y, float, value
    @ Out, evaluate, float, value at x, y
  """
  return (np.sin(3*np.pi*x)**2) + (((x-1)**2)*(1 + (np.sin(3*np.pi*y)**2))) + (((y-1)**2)*(1 + (np.sin(2*np.pi*y)**2)))

def run(self,Inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, Inputs, dict, additional inputs
    @ Out, None
  """
  self.ans = evaluate(self.x,self.y)
