'''
Created on 8/18/15

@author: maljdan

Test function with 4 Gaussian peaks derived from Sam Gerber's original
implementation. This test function has four peaks of varying size and span in
the input space.

'''
import numpy as np
import math

def eval(inp):
  retVal = (0.7) * math.exp(-((inp[0]-.25)**2)/0.09) \
         + (0.8) * math.exp(-((inp[1]-.25)**2)/0.09) \
         + (0.9) * math.exp(-((inp[0]-.75)**2)/0.01) \
         +        math.exp(-((inp[1]-.75)**2)/0.01)
  return float('%.8f' % retVal)

def run(self,Input):
  self.Z = eval((self.X,self.Y))
