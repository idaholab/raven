# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes input parameters x,y
# returns value in "ans"
# documented in analytic functions
import numpy as np

def evaluate(x,y):
  """
    Evaluates McCormick function.
    @ In, x, float, first param
    @ In, y, float, second param
    @ Out, evaluate, float, evaluation
  """
  return np.sin(x+y) + (x-y)**2 - 1.5*x + 2.5*y + 1.

def stochastic(scale=1.0):
  """
    Generates random noise
    @ In, scale, float, optional, multiplicative scale for noise range
    @ Out, stochastic, float, random value
  """
  return np.random.rand()*scale

def run(self,Inputs):
  """
    Evaluation hook for RAVEN; combines base eval and noise
    @ In, self, object, contains variables as members
    @ In, Inputs, dict, same information in dictionary form
    @ Out, None
  """
  scale = Inputs.get('stochScale',1.0)
  self.base = evaluate(self.x,self.y)
  self.stoch = stochastic(scale)
  self.ans = self.base + self.stoch

