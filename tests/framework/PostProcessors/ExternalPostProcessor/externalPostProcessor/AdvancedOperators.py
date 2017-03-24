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
Created on 12/23/14

@author: maljdan
'''
import numpy as np
import math

def Norm(self):
  return np.sqrt(np.power(self.A,2) + np.power(self.B,2))

def Mean(self):
  return (self.A + self.B) / 2.

def Sum(self):
  return self.A + self.B

def Delta(self):
  return self.A - self.B

def Max(self):
  return max(max(self.A),max(self.B))

def Min(self):
  return min(min(self.A),min(self.B))
