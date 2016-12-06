import numpy as np
import math

def initialize(self,runInfoDict,inputFiles):
  return

def run(self,Input):
  pump1Time       = Input['pump1Time'][0]
  pump2Time       = Input['pump2Time'][0]
  valveTime       = Input['valveTime'][0]

  self.failureTime = min(valveTime,(pump1Time+pump2Time))

  # failure time 100%:  9.528*60.0
  # failure time 120%: 10.128*60.0

  if self.failureTime<9.528*60.0:
    self.outcome = 1
    self.Tmax = 1600.0
  else:
    self.outcome = 0
    self.Tmax = 800.0

  if pump1Time<1440.0:
    self.pump1State = 1
  else:
    self.pump1State = 0

  if pump2Time<1440.0:
    self.pump2State = 1
  else:
    self.pump2State = 0

  if valveTime<1440.0:
    self.valveState = 1
  else:
    self.valveState = 0
