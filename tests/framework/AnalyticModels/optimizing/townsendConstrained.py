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
from math import sin, cos, atan2

def evaluate(x,y):
  """
    Evaluates Townsend function.
    @ In, x, float, value
    @ In, y, float, value
    @ Out, evaluate, value at x, y
  """
  return -np.power([cos((x-0.1)*y)],2)-x*sin(3*x+y)

def constraint(x,y):
  """
    Evaluates the constraint function @ a given point $(x,y)$
    @ In, self, object, RAVEN container
    @ Out, g(x, y), float, constraint function after modifications
                          $g(x, y) = [2*cos(t) - cos(2*t)/2 - cos(3*t)/4 - cos(4*t)/8]**2
                          + (2*sin(t))**2 - (x**2+y**2)$
    because the original constraint was $x^{2}+y^{2}<\left[2\cos t-{\frac {1}{2}}\cos 2t-{\frac {1}{4}}\cos 3t-{\frac {1}{8}}\cos 4t\right]^{2}+[2\sin t]^{2}} where: t = Atan2(x,y)$
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 0.001 - (f(x,y) - c)
  """
  t = atan2(x, y)
  return np.power([2*cos(t) - cos(2*t)/2 - cos(3*t)/4 - cos(4*t)/8],2) + np.power((2*sin(t)),2) - (x**2+y**2)

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
