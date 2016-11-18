
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
  self.y = np.zeros(numberTimeSteps)
  self.z = np.zeros(numberTimeSteps)
  self.time = np.zeros(numberTimeSteps)

  self.firstVar  = Input['firstVar']
  self.secondVar = Input['secondVar']
  self.thirdVar  = Input['thirdVar']

  self.firstOut[0] = Input['firstVar']
  self.secondOut[0] = Input['secondVar']
  self.z[0] = Input['thirdVar']
  self.time[0]= 0

  for t in range (numberTimeSteps-1):
    self.time[t+1] = self.time[t] + t_step
    self.firstOut[t+1]    = self.firstOut[t] + self.sigma*(self.secondOut[t]-self.firstOut[t]) * t_step
    self.y[t+1]    = self.y[t] + (self.x[t]*(self.rho-self.z[t])-self.y[t]) * t_step
    self.z[t+1]    = self.z[t] + (self.x[t]*self.y[t]-self.beta*self.z[t]) * t_step


