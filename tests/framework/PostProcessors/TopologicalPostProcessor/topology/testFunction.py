'''
Created on 6/16/15

@author: maljdan
'''
import numpy as np
import math

def eval(inp):
  retVal = (math.exp(- ((inp[0]/10. - .55)**2 + (inp[1]/10.-.75)**2)/.125) \
            + 0.01*(inp[0]+inp[1])/10.)
  return float('%.8f' % retVal)

def run(self,Input):
  self.Z = eval((self.X,self.Y))
