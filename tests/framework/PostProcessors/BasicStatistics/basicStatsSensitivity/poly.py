import numpy as np
from itertools import product

def eval(Y):
  A = np.array([1.0,2.2,-3.3,-0.2,11])
  B = np.array([-3,4,8,-0.1,66])
  return np.dot(A,(Y+B))

def run(self,Input):
  self.ans = eval(Input.values())
