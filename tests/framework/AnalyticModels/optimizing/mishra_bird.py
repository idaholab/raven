# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# constrained function
# optimal minimum at f(-3.1302468, -1.5821422) = -106.7645367
# parameter range is -10 <= x <= 0, -6.5 <= y <= 0
import numpy as np

def evaluate(x,y):
  return np.sin(y)*np.exp(1.-np.cos(x))**2 + np.cos(x)*np.exp(1.-np.sin(y))**2 + (x-y)**2

def constraint(x,y):
  condition = 25.
  if (x+5.)**2 + (y+5.)**2 < condition:
    return True
  return False

###
# RAVEN hooks
###

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

def constrain(self):
  return constraint(self.x,self.y)
