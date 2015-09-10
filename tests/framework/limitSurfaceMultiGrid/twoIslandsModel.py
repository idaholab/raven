import math

def run(self,Input):
  self.x1 = Input['x1']
  self.x2 = Input['x2']
  self.y4 = (self.x1**2.0) + (self.x2**2.0)
  
  if self.y4 <= 1.0: self.failure = 0.0
  else             : self.failure = 1.0
