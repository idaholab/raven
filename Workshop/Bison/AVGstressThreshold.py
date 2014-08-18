'''
Created on Oct 20, 2013

@author: crisr
'''
import numpy as np

def __residuum(self):
  return 4-self.temp

def __gradient(self):
  print('variables in function '+str(self.__varType__))
  return 

def __supportBoundingTest(self):
  print('variables in function '+str(self.__varType__))
  return 


def __residuumSign(self):
  if self.average_creep_strain_hoop>-0.008: return  1
  else          : return -1