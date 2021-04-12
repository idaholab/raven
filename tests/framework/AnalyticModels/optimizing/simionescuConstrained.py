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
  return 0.1*x*y

def constraint(x,y):
  rT = 1.0
  rS = 0.2
  n = 8
  return np.power([rT + rS * np.cos(n * math.atan(x/y))],2) - (x**2+y**2)

###
# RAVEN hooks
###

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

def constrain(self):
  return constraint(self.x,self.y)
