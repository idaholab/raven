
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
  self.time            = np.zeros(self.numberTimeSteps)
  self.cnt             = 0.0
  return

def createNewInput(self,myInput,samplerType,**Kwargs): return Kwargs['SampledVars']

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
  self.time[0]= 0

  for t in range ( self.numberTimeSteps-1):
    self.time[t+1] = self.time[t] + self.t_step
    self.x[t+1]    = self.x[t] + (self.y[t]-self.x[t])*self.t_step
    self.y[t+1]    = self.y[t] + (self.x[t]*self.z[t]-self.y[t])*self.t_step
    self.z[t+1]    = self.z[t] + (self.x[t]*self.y[t]-self.z[t])*self.t_step


