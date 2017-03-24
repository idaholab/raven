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
Created on 2/10/15

@author: maljdan
'''

import numpy as np
import math

def seg(obj,i):
  unique_keys = set()
  for idx in xrange(len(obj.X)):
    unique_keys.add((obj.minLabel[idx],obj.maxLabel[idx]))
  unique_keys = list(unique_keys)

  indices = []
  for idx in xrange(len(obj.X)):
    if obj.minLabel[idx] == unique_keys[i][0] \
    and obj.maxLabel[idx] == unique_keys[i][1]:
      indices.append(idx)

  obj.X = np.array(obj.X)[indices].tolist()
  obj.Y = np.array(obj.Y)[indices].tolist()
  obj.Z = np.array(obj.Z)[indices].tolist()

  return indices

def seg1(self):
  return seg(self,0)
def seg2(self):
  return seg(self,1)
def seg3(self):
  return seg(self,2)
def seg4(self):
  return seg(self,3)
