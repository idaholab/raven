'''
Created on 2/10/15

@author: maljdan
'''

import numpy as np
import math

def seg(obj,i):
  unique_keys = set()
  for idx in xrange(len(obj.X)):
    unique_keys.add((obj.minLabel[idx],obj.maxLabel[idx]))
  unique_keys = list(unique_keys)

  indices = []
  for idx in xrange(len(obj.X)):
    if obj.minLabel[idx] == unique_keys[i][0] \
    and obj.maxLabel[idx] == unique_keys[i][1]:
      indices.append(idx)

  obj.X = np.array(obj.X)[indices].tolist()
  obj.Y = np.array(obj.Y)[indices].tolist()
  obj.Z = np.array(obj.Z)[indices].tolist()

  return indices

def seg1(self):
  return seg(self,0)
def seg2(self):
  return seg(self,1)
def seg3(self):
  return seg(self,2)
def seg4(self):
  return seg(self,3)
