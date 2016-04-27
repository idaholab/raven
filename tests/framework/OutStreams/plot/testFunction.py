'''
Created on 4/26/16

@author: maljdan
'''
import numpy as np
import math

def eval(inp):
  retVal = math.sqrt((inp[0] - .5)**2 + (inp[1] - .5)**2)
  return float('%.1f' % retVal)

def run(self,Input):
  self.Z = eval(((self.X),(self.Y-1)))
