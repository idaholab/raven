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
# optimal minimum at f(1.0, 1.0) = 0
# parameter range is -1.5 <= x <= 1.5, -1.5 <= y <= 1.5
import numpy as np

def evaluate(x,y):
  """
    Evaluates Rosenbrock function.
    @ In, x, float, value
    @ In, y, float, value
    @ Out, evaluate, value at x, y
  """
  return (1- x)**2 + 100*(y-x**2)**2

def constraint(x,y):
  """
    Evaluates the constraint function @ a given point $(x,y)$
    @ In, self, object, RAVEN container
    @ Out, g(x, y), float, constraint function after modifications
                          $g(x, y) = 2 - (x**2+y**2)$
    because the original constraint was (x**2+y**2) <= 2
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 0.001 - (f(x,y) - c)
  """
  return 2 - (x**2+y**2)

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

def constrain(self):
  """
    Constrain calls the constraint function.
    @ In, self, object, RAVEN container
    @ Out, explicitConstrain, float, positive if the constraint is satisfied
          and negative if violated.
  """
  explicitConstrain = constraint(self.x,self.y)
  return explicitConstrain
