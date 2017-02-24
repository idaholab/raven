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
  print('CLAD DAMAGED IS ' + str(self.CladDamaged))
  if self.CladDamaged == 1: return -1.0
  else: return 1.0
