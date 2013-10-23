from scipy.stats import poisson,binom
import numpy as np
import copy
#import random

def initialize(self,runInfoDict,inputFiles):
  self.SampledVars = []
  #self.p_detector     = 0.05
  #self.p_signalCondit = 0.01
  #self.p_starter      = 0.08
  #self.p_pump         = 0.05
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
  self.x = self.SampledVars.pop()['x']
  self.y = self.SampledVars.pop()['y']
  self.outcome = self.x*self.x + self.y
