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
#import pylab as pyl
#import random
#import mpl_toolkits.mplot3d.axes3d as p3

def initialize(self,runInfoDict,inputFiles):
  self.sigma = 10.0
  self.rho   = 28.0
  self.beta  = 8.0/3.0
  return

def run(self,Input):
  max_time = 0.03
  t_step = 0.01

  numberTimeSteps = int(max_time/t_step)

  self.firstOut = np.zeros(numberTimeSteps)
  self.secondOut = np.zeros(numberTimeSteps)
  self.z = np.zeros(numberTimeSteps)
  self.time = np.zeros(numberTimeSteps)

  firstVar  = Input['awful.variable+name']
  secondVar = Input['@another|awful name']
  self.z0  = Input['z0']

  self.firstOut[0] = firstVar
  self.secondOut[0] = secondVar
  self.z[0] = Input['z0']
  self.time[0]= 0

  for t in range (numberTimeSteps-1):
    self.time[t+1] = self.time[t] + t_step
    self.firstOut[t+1]    = self.firstOut[t] + self.sigma*(self.secondOut[t]-self.firstOut[t]) * t_step
    self.secondOut[t+1]    = self.secondOut[t] + (self.firstOut[t]*(self.rho-self.z[t])-self.secondOut[t]) * t_step
    self.z[t+1]    = self.z[t] + (self.firstOut[t]*self.secondOut[t]-self.beta*self.z[t]) * t_step
