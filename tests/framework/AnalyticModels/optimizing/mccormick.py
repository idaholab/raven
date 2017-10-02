# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# documented in analytic functions
import numpy as np

def evaluate(x,y):
  return np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1.

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

