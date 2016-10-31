import numpy as np
import math

def initialize(self,runInfoDict,inputFiles):
  return

def run(self,Input):
  tSBO            = 0.0 #Input['tSBO'][0]
  tRec            = 0.0 #Input['tRec'][0]
  powerMultiplier = 1.0 # Input['powerMultuplier'][0]
  pump1Time       = Input['pump1Time'][0]
  pump2Time       = Input['pump2Time'][0]
  valveTime       = Input['valveTime'][0]
    
  max_time = 24.0*60.0
  numberTimeSteps = 1000
  
  dt = max_time/numberTimeSteps
  
  time     = np.arange(0.0,max_time,dt)
  timePlot = np.arange(0.0,max_time/60.0,dt/60.0)
  
  T            = np.zeros(numberTimeSteps)
  P            = np.zeros(numberTimeSteps)
  heatRem      = np.zeros(numberTimeSteps)
  heatRemCoeff = np.zeros(numberTimeSteps)
  
  self.failureTime = min(valveTime,(pump1Time+pump2Time))
  
  powerTau = 300.0
  
  for t in range (numberTimeSteps-1):
    if t==0:
      T[t] = 800.0
    
    P[t] = 1500.0 * powerMultiplier * math.exp(-time[t]/powerTau)
    
    heatRemCoeff[t] = 0.0
    '''
    if time[t]<pump1Time:
      heatRemCoeff[t] = heatRemCoeff[t] + .80
    if time[t]<pump2Time:
      heatRemCoeff[t] = heatRemCoeff[t] + .35
    '''
    if time[t]<(pump1Time + pump2Time):
      heatRemCoeff[t] = 1.15
    else:
      heatRemCoeff[t] = 0.0
    if time[t]>valveTime:
      heatRemCoeff[t] = heatRemCoeff[t] * 0.0
  
    if time[t]<tSBO:
      heatRem[t] = P[t] * heatRemCoeff[t]
    elif time[t]>tSBO and time[t]<tRec:
      heatRem[t] = P[t] * heatRemCoeff[t] * math.exp(-(time[t]-tSBO)/1.0)
    elif time[t]>tRec:
      heatRem[t] = P[t]* heatRemCoeff[t] * (1.0-math.exp(-(time[t]-tRec)/10.0))
    
    if time[t]<valveTime and time[t]<(pump1Time + pump2Time):
      mCP = 300.0
    else:
      mCP = 80.0
    
    T[t+1] = T[t] + 1.0/mCP * dt * (P[t] - heatRem[t])
    
    if T[t+1]>1400.0:
      break
      
  self.Tmax = np.amax(T)
    
  if self.Tmax>1400.0:
    self.outcome = 1
  else:
    self.outcome = 0
    
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