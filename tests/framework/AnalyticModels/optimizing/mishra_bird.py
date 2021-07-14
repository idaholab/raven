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
# constrained function
# optimal minimum at f(-3.1302468, -1.5821422) = -106.7645367
# parameter range is -10 <= x <= 0, -6.5 <= y <= 0
import numpy as np

def evaluate(x,y):
  """
    Evaluates Mishra bird function.
    @ In, x, float, value
    @ In, y, float, value
    @ Out, evaluate, value at x, y
  """
  return np.sin(y)*np.exp(1.-np.cos(x))**2 + np.cos(x)*np.exp(1.-np.sin(y))**2 + (x-y)**2

###
# RAVEN hooks
###

def run(self,Inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, Inputs, dict, additional inputs
    @ Out, None
  """
  self.ans = evaluate(self.x,self.y)