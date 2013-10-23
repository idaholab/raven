from scipy.stats import poisson,binom
import numpy as np
import copy
import random

def initialize(self,runInfoDict,inputFiles):
  self.SampledVars = []
  self.p_detector     = 0.05
  self.p_signalCondit = 0.01
  self.p_starter      = 0.08
  self.p_pump         = 0.05
  self.counter = 0
  return
def createNewInput(self,myInput,samplerType,**Kwargs):
  self.SampledVars.append(Kwargs['sampledVars'])
  return None

def readMoreXML(self,xmlNode):
  child = xmlNode.find('initialize')
  for son in child:
    exec('self.'+son.tag + ' =  float(son.text)')

def run(self,Input,jobHandler):
  self.NumberOfFires = self.SampledVars.pop()['numberOfFires']
  #print('NumberOfFires ' + str(self.NumberOfFires))
  
  self.testHistOut = np.zeros(10)
  for x in range(10) : self.testHistOut[x] = x
  
  self.outcome = 1
     
  if self.NumberOfFires==0:
    self.outcome *= 1
  else:   
    detectorStatus = random.random()
    if detectorStatus < self.p_detector: self.outcome *= 0
    
    signalCondStatus = random.random()
    if signalCondStatus < self.p_signalCondit: self.outcome *= 0
    
    for i in range(self.NumberOfFires): 
      pumpStatus     = random.random()
      if pumpStatus < self.p_pump: self.outcome *= 0
      
      starterStatus = random.random()
      if starterStatus < self.p_starter: self.outcome *= 0
  self.counter = self.counter + 1
  print('counter : ' + str(self.counter))
