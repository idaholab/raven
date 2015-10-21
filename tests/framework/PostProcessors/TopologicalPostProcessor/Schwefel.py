'''
Created on 6/24/15

@author: maljdan

Test function with undulations of varying size used to test the new persistence
algorithms

'''
import numpy as np
import math

def eval(inp):
  retVal = 0
  for xi in inp:
    xi *= 500
    xi -= 250
    retVal += -xi * (math.sin(math.sqrt(math.fabs(xi))))
  return float('%.8f' % retVal)

def run(self,Input):
  self.Z = eval((self.X,self.Y))
