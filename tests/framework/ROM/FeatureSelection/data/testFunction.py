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
"""
Created on 11/15/2022

@author: Andrea Alfonsi
"""
import numpy as np
import math

def eval(inp):
  """
    Evaluation method fpr the quadratic model
    @ In, inp, tuple, tuple of input values
    @ Out, eval, float, the result of the evaluation
  """
  retVal = (inp[0] - .5)**2 + (inp[1] - .5)**2
  return float('%.8f' % retVal)

def run(self,Input):
  """
    Run method of the model that is aimed to test feature
    selections algorithms. The variables /= X and Y are
    simply used to add a noise and they should be filtered
    out by the algorithm applied.
    @ In, Input, dict, dictionary of input (sampled) vars
    @ Out, None
  """
  # The variables /= X,Y are sampled from a U(-1,1)
  self.Z = eval(((self.X-2),(self.Y+1000)/(2000)))
  for var in Input:
    if var not in ['X','Y']:
      self.Z += Input[var]/1000.
