import numpy as np
import math

def run(self,Input):
  self.tSBO   = Input['tSBO'][0]
  self.tRec   = Input['tRec'][0]
  
  max_time = 10.0*60.0
  numberTimeSteps = 1000
  
  dt = max_time/numberTimeSteps
  
  self.time = np.arange(0.0,max_time,dt)
  
  self.T = np.zeros(numberTimeSteps)
  self.P = np.zeros(numberTimeSteps)
  self.heatRem = np.zeros(numberTimeSteps)
  
  tau = 130.0
  mCP = 100.0
  
  for t in range (numberTimeSteps-1):
    if t==0:
      self.T[t] = 800.0
    
    self.P[t] = 1000.0 * math.exp(-self.time[t]/tau)

    if self.time[t]<self.tSBO:
      self.heatRem[t] = self.P[t]*1.05
    elif self.time[t]>self.tSBO and self.time[t]<self.tRec:
      self.heatRem[t] = self.P[t] *1.05 * math.exp(-(self.time[t]-self.tSBO)/10.0)
    elif self.time[t]>self.tRec:
      self.heatRem[t] = 150.0 * (1.0-math.exp(-(self.time[t]-self.tRec)/50.0))

    self.T[t+1] = self.T[t] + 1.0/mCP * dt * (self.P[t] - self.heatRem[t])

