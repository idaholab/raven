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
####
import numpy as np

def evaluate(y,val):
  ans = np.dot(y,val)
  return ans

def constraint(y,wt):
  condition = 400.
  if np.dot(y,wt) < condition:
    return True
  return False

###
# RAVEN hooks
###

def run(self,Inputs):
  self.items = np.linspace(0,22,22)
  self.val = np.asarray([150,25,200,160,60,45,60,40,30,10,70,30,15,10,40,70,75,80,20,12,50,10])
  self.wt = np.asarray([9,13,153,50,15,68,27,39,23,52,11,32,24,48,73,42,43,22,7,18,4,30])
  self.ans = evaluate(self.y, self.val)

def constrain(self):
  self.wt = np.asarray([9,13,153,50,15,68,27,39,23,52,11,32,24,48,73,42,43,22,7,18,4,30])
  return constraint(self.y,self.wt)
