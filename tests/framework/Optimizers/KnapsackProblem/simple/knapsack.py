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
  condition = 67.
  if np.dot(y,wt) < condition:
    return True
  return False

###
# RAVEN hooks
###

def run(self,Inputs):
  self.items = np.linspace(0,5,6)
  self.val = np.asarray([505,352,458,220,354,545])
  self.wt = np.asarray([23,26,20,18,32,26])
  self.ans = evaluate(self.y, self.val)

def constrain(self):
  self.wt = np.asarray([23,26,20,18,32,26])
  return constraint(self.y,self.wt)
