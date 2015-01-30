'''
Created on Oct 20, 2013

@author: alfoa
'''
import numpy as np
import math

def __residuum__(self):
  print('variables in function '+str(self.__varType__))
  return

def __gradient__(self):
  print('variables in function '+str(self.__varType__))
  return

def __supportBoundingTest__(self):
  print('variables in function '+str(self.__varType__))
  return

def __residuumSign(self):
#  return np.copysign(1, 5-self.auxTime[-1:] -self.tempTH[-1:])
  print('CLAD DAMAGED IS ' + str(self.CladDamaged))
  if self.CladDamaged == 1: return -1.0
  else: return 1.0
