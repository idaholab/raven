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

# from wikipedia: dx/dt = sigma*(y-x)  ; dy/dt = x*(rho-z)-y  dz/dt = x*y-beta*z

import numpy as np

def initialize(self,runInfoDict,inputFiles):
  self.sigma = 10.0
  self.rho   = 28.0
  self.beta  = 8.0/3.0
  return

def run(self,Input):
  max_time = 0.7
  t_step = 0.005
  disc = 1.0
  self.sigma = 10.0
  self.rho   = -28.0
  self.beta  = 8.0/3.0

  numberTimeSteps = int(max_time/t_step)

  self.x2    = np.zeros(numberTimeSteps)
  self.y2    = np.zeros(numberTimeSteps)
  self.z2    = np.zeros(numberTimeSteps)
  self.time2 = np.zeros(numberTimeSteps)

  self.x0 = Input['x0']
  self.y0 = Input['y0']
  self.z0 = Input['z0']

  self.x2[0] = Input['x0']
  self.y2[0] = Input['y0']
  self.z2[0] = Input['z0']
  self.time2[0]= 0.0

  for t in range (numberTimeSteps-1):
    self.time2[t+1] = self.time2[t] + t_step
    self.x2[t+1]    = self.x2[t] + disc*self.sigma*(self.y2[t]-self.x2[t]) * t_step
    self.y2[t+1]    = self.y2[t] + disc*(self.x2[t]*(self.rho-self.z2[t])-self.y2[t]) * t_step
    self.z2[t+1]    = self.z2[t] + disc*(self.x2[t]*self.y2[t]-self.beta*self.z2[t]) * t_step
