# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes any number of input parameters x[i]
# returns value in "ans"
# optimal minimum at f(1,1,...,1,1) = 0, only minimum for up to 3 inputs
# note for 4 to 7 inputs, a second local minima exists near (-1,1,...,1), and it gets complicated after that
# parameter range is -inf <= x[i] <= inf

import numpy as np

def evaluate2d(X,Y):
  return 100*(Y-X*X)**2 + (X-1)**2

def evaluate(*args):#xs):
  xs = np.array(args)
  return np.sum( 100.*(xs[1:] - xs[:-1]**2)**2 + (xs[:-1] - 1.)**2 )

def run(self,Inputs):
  self.ans = evaluate(Inputs.values())

