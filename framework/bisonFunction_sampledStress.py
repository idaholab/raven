'''
Created on Oct 20, 2013

@author: crisr
'''
import numpy as np
np.random.seed(10)
def __residuum__(self):
  print('variables in function '+str(self.__varType__))
  return 

def __gradient__(self):
  print('variables in function '+str(self.__varType__))
  return 

def __supportBoundingTest__(self):
  print('variables in function '+str(self.__varType__))
  return 

def __residualSign__(self):
  print(str(self.max_stress_spl-self.max_stress[-1:]))
  return np.copysign(1.0,self.max_stress_spl[-1:]-self.max_stress[-1:])


