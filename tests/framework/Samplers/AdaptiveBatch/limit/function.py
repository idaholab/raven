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
Created on 8/13/15

@author: maljdan
'''

import numpy as np
import math
import time
import random

def eval(inp):
  retVal = inp[1] - inp[0]
  # time.sleep(random.unif(10,60))
  # time.sleep(1)
  return float('%.8f' % retVal)

def run(self,Input):
  self.y = eval((self.x1,self.x2))
