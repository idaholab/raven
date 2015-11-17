'''
Created on 6/16/15

@author: maljdan
'''
import numpy as np
import math

def eval(inp):
  retVal = (math.exp(- ((inp[0] - .55)**2 + (inp[1]-.75)**2)/.125) \
            + 0.01*(inp[0]+inp[1]))
  return float('%.8f' % retVal)

def gerber(x,y):
  return   (1./2.) * math.exp(-((x-.25)**2)/0.09) \
         + (1./4.) * math.exp(-((y-.25)**2)/0.09) \
         + (3./4.) * math.exp(-((x-.75)**2)/0.01) \
         +           math.exp(-((y-.75)**2)/0.01)

def run(self,Input):
  # Run a test problem with X on the scale [2,3] and Y on the scale
  #  [-1000,1000], to make sure that the scaling is handled approriately.
  #  The function above assumes both variables are in the range [0,1], so we
  #  will scale them here. One dimension is scaled up and centered at zero, the
  #  other is translated and has the correct scale, this should be good enough
  #  for testing purposes.
  self.Z = eval(((self.X-2),(self.Y+1000)/(2000)))
  # self.Z = gerber((self.X-2),(self.Y+1000)/(2000))
