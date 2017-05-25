import numpy as np
import math

def initialize(self,runInfoDict,inputFiles):
  return

def run(self,Input):
  max_time = 24.0*60.0
  numberTimeSteps = 1000

  dt = max_time/numberTimeSteps

  self.T       = np.zeros(0)
  self.time    = np.zeros(0)
  self.heatRem = np.zeros(0)
  P            = np.zeros(numberTimeSteps)
  self.Tbase   = np.zeros(numberTimeSteps)

  powerTau = 300.0

  self.timeBase = np.arange(0.0,max_time,dt)

  for t in range (numberTimeSteps-1):
    if t==0:
      self.Tbase[t] = 800.0
      self.T        = np.append(self.T,self.Tbase[t])
      self.time     = np.append(self.time,self.timeBase[t])
      self.heatRem  = np.append(self.heatRem,0)

    P[t] = 1500.0 * math.exp(-self.time[t]/powerTau)

    if self.timeBase[t]<self.tSBO:
      heatRem = 1.1
      mCP = 100.0
    elif self.timeBase[t]>=self.tSBO and self.timeBase[t]<=(self.tREC+self.tSBO):
      heatRem = math.exp(-(self.timeBase[t]-self.tSBO)/50.0)
      mCP = 200.0
    elif self.timeBase[t]>(self.tREC+self.tSBO):
      heatRem = 1.2 #* (1.0-math.exp(-(self.timeBase[t]-self.tREC-self.tSBO)/50.0))
      mCP = 100.0

    self.Tbase[t+1] = self.Tbase[t] + 1.0/mCP * dt * P[t] * (1.0 - heatRem)

    self.heatRem = np.append(self.heatRem,heatRem)
    self.T       = np.append(self.T,self.Tbase[t+1])
    self.time    = np.append(self.time,self.timeBase[t+1])

    if self.T[t+1]>1400.0:
      break

  self.Tmax = np.amax(self.T)
  self.Tfin = self.T[-1]

  if self.Tmax>1400.0:
    self.outcome = 1.0
  else:
    self.outcome = 0.0

