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
Created on 4/28/16

@author: maljdan
'''
import numpy as np
import math

def eval(inp):
  retVal = (inp[0] - .5)**2 + (inp[1] - .5)**2
  if retVal > 0.25:
    return 1
  else:
    return 0

def run(self,Input):
  # Run a test problem with X on the scale [2,3] and Y on the scale
  #  [-1000,1000], to make sure that the scaling is handled approriately.
  #  The function above assumes both variables are in the range [0,1], so we
  #  will scale them here. One dimension is scaled up and centered at zero, the
  #  other is translated and has the correct scale, this should be good enough
  #  for testing purposes.
  self.Z = eval(((self.X-2),(self.Y+1000)/(2000)))
