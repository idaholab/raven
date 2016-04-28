'''
Created on 4/28/16

@author: maljdan
'''
import numpy as np
import math

def eval(inp):
  retVal = (inp[0] - .5)**2 + (inp[1] - .5)**2
  return float('%.8f' % retVal)

def eval2(inp):
  retVal = inp[0]+inp[1]
  return float('%.8f' % retVal)

def run(self,Input):
  # Run a test problem with X on the scale [2,3] and Y on the scale
  #  [-1000,1000], to make sure that the scaling is handled approriately.
  #  The function above assumes both variables are in the range [0,1], so we
  #  will scale them here. One dimension is scaled up and centered at zero, the
  #  other is translated and has the correct scale, this should be good enough
  #  for testing purposes.
  self.Z = eval(((self.X-2),(self.Y+1000)/(2000)))
  self.Z2 = eval2(((self.X-2),(self.Y+1000)/(2000)))