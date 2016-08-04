import numpy as np
from scipy.integrate import odeint
import copy
from numpy import interp
import time
# CAUTION HERE #
# IT IS IMPORTANT THAT THE USER IS AWARE THAT THE EXTERNAL MODEL (RUN METHOD)
# NEEDS TO BE THREAD-SAFE!!!!!
# IN CASE IT IS NOT, THE USER NEEDS TO SET A LOCK SYSTEM!!!!!
import threading
localLock = threading.RLock()

def initialize(self, runInfo, inputs):
  self.N                       = 12 # number of points to discretize
  self.L                       = 1.0 # lenght of the rod
  self.X                       = np.linspace(0, self.L, self.N) # position along the rod
  self.h                       = self.L / (self.N - 1) # discretization step
  self.initialTemperatures     = 150.0 * np.ones(self.X.shape) # initial temperature
  self.shapeToUse              = copy.deepcopy(self.X.shape)
  self.timeDiscretization      = np.linspace(0.0, 100.0, 10)

def odeFuncion(u, t, xshape, N, h, k, timeFromFirstModel):
  dudt     = np.zeros(xshape)
  dudt[0] = 0 # constant at boundary condition
  dudt[-1] = 0
  
  # k and timeFromFirstModel are arrays that need to be interpolated! (interp)
  
  # now for the internal nodes
  for i in range(1, N-1): dudt[i] = interp(t, timeFromFirstModel, k) * (u[i + 1] - 2*u[i] + u[i - 1]) / h**2
  return dudt

def run(self, Input):
  global localLock
  self.initialTemperatures[0]  = self.leftTemperature  # one boundary condition
  self.initialTemperatures[-1] = self.rightTemperature # the other boundary condition
  with localLock:
    solution = odeint(odeFuncion, self.initialTemperatures, self.timeDiscretization, args=copy.deepcopy((self.shapeToUse,self.N,self.h,self.k,self.timeFromFirstModel)),full_output = 1)
    self.solution = solution[0][:,5]
    self.solutionK = (self.solution/10000.0)*self.k

