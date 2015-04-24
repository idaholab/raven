import math

def initialize(self,runInfoDict,inputFiles):
  self.SampledVars = None
  return

def createNewInput(self,myInput,samplerType,**Kwargs):
  return Kwargs['SampledVars']

def run(self,Input):
  
  self.x1 = Input['x1']
  self.x2 = Input['x2']
  self.x3 = Input['x3']
  
  a =  1.0
  b =  2.0
  c =  3.0
  l = -1.0
  
  self.y1 = self.x1
  self.y2 = self.x1
  self.y3 = a*self.x1 + b*self.x2 - c*self.x3
  self.y4 = self.x1*self.x1 + self.x1*self.x2*self.x3
  self.y5 = math.exp(l*self.x1)