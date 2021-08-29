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
# optimal minimum at f(2.0052938,1.1944509)=-2.0239884
# parameter range is -2.25 <= x <= 2.5, -2.5 <= y <= 1.75
import numpy as np
import math

def evaluate(x,y):
  """
    Evaluates Simionescu function.
    @ In, x, float, value
    @ In, y, float, value
    @ Out, evaluate, value at x, y
  """
  return 0.1*x*y

def constraint(x,y):
  """
    Evaluates the constraint function @ a given point $(x,y)$
    @ In, self, object, RAVEN container
    @ Out, g(x, y), float, constraint function after modifications
                          $g(x, y) = [rT + rS * np.cos(n * math.atan(x/y))]**2 - (x**2+y**2)$
    because the original constraint was $ x^{2}+y^{2}\leq \left[r_{T}+r_{S}\cos \left(n\arctan {\frac {x}{y}}\right)\right]^{2}$
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 0.001 - (f(x,y) - c)
  """
  rT = 1.0
  rS = 0.2
  n = 8
  g = np.power([rT + rS * np.cos(n * math.atan(x/y))],2) - (x**2+y**2)
  return g

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
