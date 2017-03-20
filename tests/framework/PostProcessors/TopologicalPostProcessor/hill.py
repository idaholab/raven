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
Created on 6/16/15

@author: maljdan

A single 2D Gaussian bump for testing topological methods on.

'''
import numpy as np
import math

def eval(inp):
  retVal = (math.exp(- ((inp[0]/10. - .55)**2 + (inp[1]/10.-.75)**2)/.125) \
            + 0.01*(inp[0]+inp[1])/10.)
  return float('%.8f' % retVal)

def run(self,Input):
  self.Z = eval((self.X,self.Y))
