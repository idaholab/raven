import random
import numpy as np

def run(self,Input):
  # intput: None
  # output: time

  T  = 20. * 365.
  dt = 1.

  powerDuration = 18.*30. # 18 months, 30 days per months
  SDduration    = 30.
  cycle = powerDuration + SDduration

  self.time    = np.arange(dt,T,dt)
  self.opPower = 100 * np.ones(len(self.time))

  counter = 1.0
  for ts in range(len(self.time)):
    if (self.time[ts]%cycle)<(powerDuration%cycle):
      self.opPower[ts] = 100.
    else:
      self.opPower[ts] = 0.
