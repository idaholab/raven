'''
Created on NO NEED TO KNOW

@author: alfoa
'''
import numpy as np
import math

def failureProbability(self):
  failure = 0
  success = 0
  for element in range(len(self.x01)):
    if self.x01[element] > 2: failure += 1
    else                    : success += 1
  if failure > 0 and success > 0: return float(failure)/float(failure+success)
  else: return 0.0
