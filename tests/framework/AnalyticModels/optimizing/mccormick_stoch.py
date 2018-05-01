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
# documented in analytic functions
import numpy as np

def evaluate(x,y):
  """
    Evaluates McCormick function.
    @ In, x, float, first param
    @ In, y, float, second param
    @ Out, evaluate, float, evaluation
  """
  return np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1.

def stochastic(scale=1.0):
  """
    Generates random noise
    @ In, scale, float, optional, multiplicative scale for noise range
    @ Out, stochastic, float, random value
  """
  return np.random.rand()*scale

def run(self,Inputs):
  """
    Evaluation hook for RAVEN; combines base eval and noise
    @ In, self, object, contains variables as members
    @ In, Inputs, dict, same information in dictionary form
    @ Out, None
  """
  scale = Inputs.get('stochScale',1.0)
  self.base = evaluate(self.x,self.y)
  self.stoch = stochastic(scale)
  self.ans = self.base + self.stoch

