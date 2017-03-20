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
'''
Created on 6/24/15

@author: maljdan

Test function with undulations of varying size used to test the new persistence
algorithms

'''
import numpy as np
import math

def eval(inp):
  retVal = 0
  for xi in inp:
    xi *= 500
    xi -= 250
    retVal += -xi * (math.sin(math.sqrt(math.fabs(xi))))
  return float('%.8f' % retVal)

def run(self,Input):
  self.Z = eval((self.X,self.Y))
