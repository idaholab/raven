from scipy.stats import poisson,binom
import numpy as np
import copy
#import random

def initialize(self,runInfoDict,inputFiles):
  self.SampledVars = None
  #self.p_detector     = 0.05
  #self.p_signalCondit = 0.01
  #self.p_starter      = 0.08
  #self.p_pump         = 0.05
  self.counter = 0
  return
def createNewInput(self,myInput,samplerType,**Kwargs):
  print(Kwargs)
  self.SampledVars = Kwargs['sampledVars']
  return None

def readMoreXML(self,xmlNode):
  child = xmlNode.find('initialize')
  for son in child:
    exec('self.'+son.tag + ' =  float(son.text)')

def run(self,Input,jobHandler):
  self.x = self.SampledVars['x']
  self.y = self.SampledVars['y']
  self.outcome = self.x*self.x + self.y
