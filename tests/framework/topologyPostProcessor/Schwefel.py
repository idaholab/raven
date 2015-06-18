import numpy as np
import math

def eval(inp):
  retVal = 0
  for xi in inp:
    xi *= 1000
    xi -= 500
    retVal += -xi * (math.sin(math.sqrt(math.abs(xi))))
  return float('%.8f' % retVal)

def run(self,Input):
  self.Z = eval((self.X,self.Y))
