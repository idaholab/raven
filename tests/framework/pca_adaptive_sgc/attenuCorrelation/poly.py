import numpy as np
from itertools import product


def eval(Y):
  l = float(len(Y))
  return np.exp( -sum(Y)/l)

def run(self,Input):
  Yvalue = []
  Yvalue.append(Input['x1'])
  Yvalue.append(Input['x2'])
  Yvalue.append(Input['x3'])
  Yvalue.append(Input['x4'])
  Yvalue.append(Input['x5'])
  self.ans = eval(Yvalue)
