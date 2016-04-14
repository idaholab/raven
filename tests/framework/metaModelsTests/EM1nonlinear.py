import numpy as np
from scipy.integrate import odeint
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
  self.initialTemperatures     = 600.0 * np.ones(self.X.shape) # initial temperature
  self.timeDiscretization      = np.linspace(0.0, 100.0, 11)

def odeFuncion(T, t, xshape, N, h, k):
    dTdt     = np.zeros(xshape)
    dTdt[0]  = 0 # constant at boundary condition
    dTdt[-1] = 0
    # now for the internal nodes
    for i in range(1, N-1): dTdt[i] = k * (T[i + 1] - 2*T[i] + T[i - 1]) / h**2
    return dTdt

def run(self, Input):
  global localLock
  self.initialTemperatures[0]  = self.leftTemperature  # one boundary condition
  self.initialTemperatures[-1] = self.rightTemperature
  with localLock:
    solution = odeint(odeFuncion, self.initialTemperatures, self.timeDiscretization, args=(self.X.shape,self.N,self.h,self.k),full_output = 1)
    self.solution = solution[0][-1,5]
  self.averageTemperature = (self.leftTemperature + self.rightTemperature)/2.0
  self.linearHeat = self.k*(max(self.leftTemperature,self.rightTemperature)-min(self.leftTemperature,self.rightTemperature))
  print("mid value is "+str(self.solution))

