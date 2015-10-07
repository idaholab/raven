'''
Created on Jan 20, 2015

@author: alfoa
'''
import numpy as np
import math

def __supportBoundingTest(self): pass
def __residuum(self): pass
def __gradient(self): pass

def __residuumSign(self):
  print('SYSTEM FAILURE IS ' + str(self.systemFailed))
  if self.systemFailed == 1: return -1.0
  else                     : return 1.0
