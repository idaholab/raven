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
# parameter range is -1.5 <= x <= 1.5, -0.5 <= y <= 2.5
import numpy as np

def evaluate(x,y):
  return (1- x)**2 + 100*(y-x**2)**2

def constraint(x,y):
  return 2 - (x+y)

###
# RAVEN hooks
###

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

def constrain(self):
  return constraint(self.x,self.y)
