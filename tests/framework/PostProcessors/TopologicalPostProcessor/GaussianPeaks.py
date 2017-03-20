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
Created on 8/18/15

@author: maljdan

Test function with 4 Gaussian peaks derived from Sam Gerber's original
implementation. This test function has four peaks of varying size and span in
the input space.

'''
import numpy as np
import math

def eval(inp):
  retVal = (0.7) * math.exp(-((inp[0]-.25)**2)/0.09) \
         + (0.8) * math.exp(-((inp[1]-.25)**2)/0.09) \
         + (0.9) * math.exp(-((inp[0]-.75)**2)/0.01) \
         +        math.exp(-((inp[1]-.75)**2)/0.01)
  return float('%.8f' % retVal)

def run(self,Input):
  self.Z = eval((self.X,self.Y))
