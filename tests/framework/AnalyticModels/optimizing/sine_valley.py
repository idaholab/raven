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
# Function designed by D. Maljovec to have sinusoidally-shaped minimum region
#
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(0,0) = 0
# parameter range is -4.5 <= x,y <= 4.5
import numpy as np

def evaluate(x,y):
  return (np.cos(0.7*(x+y))-(y-x))**2 + 0.1*(x+y)**2

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

def precond(y):
  self.x = y

