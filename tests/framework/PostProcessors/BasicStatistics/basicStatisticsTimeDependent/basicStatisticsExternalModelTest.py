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

''' from wikipedia: dx/dt = sigma*(y-x)  ; dy/dt = x*(rho-z)-y  dz/dt = x*y-beta*z  ; '''

import numpy as np
import copy
#import pylab as pyl
#import random
#import mpl_toolkits.mplot3d.axes3d as p3

def initialize(self,runInfoDict,inputFiles):
  print('Life is beautiful my friends. Do not waste it!')
  self.max_time        = 0.03
  self.t_step          = 0.01
  self.numberTimeSteps = int(self.max_time/self.t_step)
  self.x               = np.zeros(self.numberTimeSteps)
  self.y               = np.zeros(self.numberTimeSteps)
  self.z               = np.zeros(self.numberTimeSteps)
  self.time            = np.zeros(self.numberTimeSteps)-self.t_step
  self.cnt             = 0.0
  return

def run(self,Input):
  self.cnt = 1.0
  self.x0 = 1.0
  self.y0 = 1.0
  self.z0 = 1.0
  self.x01  = copy.deepcopy(self.cnt+Input['x0'])
  self.x02 = copy.deepcopy(self.cnt+Input['x0'])
  self.z01  = copy.deepcopy(self.cnt+Input['x0'])
  self.z02 = 101.0 - copy.deepcopy(self.cnt+Input['x0'])
  self.y01  = copy.deepcopy(Input['x0'])
  self.y02 = copy.deepcopy(Input['y0'])
  self.time[0]= -self.t_step*5.0
  self.x[0] =  copy.deepcopy(self.cnt+Input['x0'])
  self.y[0] =  copy.deepcopy(self.cnt+Input['y0'])
  self.z[0] =  copy.deepcopy(self.cnt+Input['z0'])
  for t in range ( self.numberTimeSteps-1):
    self.time[t+1] = self.time[t] + self.t_step*5.0
    self.x[t+1]    = self.x[t] + (self.y[t]-self.x[t])*self.t_step
    self.y[t+1]    = self.y[t] + (self.x[t]*self.z[t]-self.y[t])*self.t_step
    self.z[t+1]    = self.z[t] + (self.x[t]*self.y[t]-self.z[t])*self.t_step
