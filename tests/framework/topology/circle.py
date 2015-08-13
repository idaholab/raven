'''
Created on 8/13/15

@author: maljdan
'''
#test comment being added...

import numpy as np
import math

def eval(inp):
  retVal = math.sqrt(inp[0]*inp[0] + inp[1]*inp[1])
  return float('%.8f' % retVal)

def run(self,Input):
  self.y = eval((self.x1,self.x2))
