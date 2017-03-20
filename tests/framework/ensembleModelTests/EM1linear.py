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
import numpy as np
from scipy.integrate import odeint
import copy
import time
# CAUTION HERE #
# IT IS IMPORTANT THAT THE USER IS AWARE THAT THE EXTERNAL MODEL (RUN METHOD)
# NEEDS TO BE THREAD-SAFE!!!!!
# IN CASE IT IS NOT, THE USER NEEDS TO SET A LOCK SYSTEM!!!!!
import threading
localLock = threading.RLock()

def initialize(self, runInfo, inputs):
  self.N                       = 10 # number of points to discretize
  self.L                       = 1.0 # lenght of the rod
  self.X                       = np.linspace(0, self.L, self.N) # position along the rod
  self.h                       = self.L / (self.N - 1) # discretization step
  self.initialTemperatures     = 150.0 * np.ones(self.X.shape) # initial temperature
  self.shapeToUse              = copy.deepcopy(self.X.shape)
  self.timeDiscretization      = np.linspace(0.0, 100.0, 10)

def odeFuncion(u, t, xshape, N, h, k):
  dudt     = np.zeros(xshape)
  dudt[0] = 0 # constant at boundary condition
  dudt[-1] = 0
  # now for the internal nodes
  for i in range(1, N-1): dudt[i] = k * (u[i + 1] - 2*u[i] + u[i - 1]) / h**2
  return dudt

def run(self, Input):
  global localLock
  self.initialTemperatures[0]  = self.leftTemperature  # one boundary condition
  self.initialTemperatures[-1] = self.rightTemperature # the other boundary condition
  with localLock:
    solution = odeint(odeFuncion, self.initialTemperatures, self.timeDiscretization, args=copy.deepcopy((self.shapeToUse,self.N,self.h,self.k)),full_output = 1)
    self.solution = solution[0][-1,5]

