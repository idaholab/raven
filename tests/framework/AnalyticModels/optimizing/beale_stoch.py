# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
# but with added stochastic component that is 10% of the function value
#
# takes input parameters x,y
# returns value in "ans"
# optimal AVERAGE minimum at f(3,0.5) = 0
# parameter range is -4.5 <= x,y <= 4.5
import numpy as np

def stochastic(scale=1.0):
  """
    Returns a random value scaled by an input.
    @ In, scale, float, optional, extends the range from 0 to scale
    @ Out, stochastic, float, scaled random value
  """
  return scale*np.random.rand()*np.random.choice([-1,1])

def evaluate(x,y):
  """
    Evaluates Beale's function
    @ In, x, float, first value
    @ In, y, float, second value
    @ Out, evaluate, float, Beale's function
  """
  return (1.5 - x + x*y)**2 + (2.25 - x + x*y*y)**2 + (2.625 - x + x*y*y*y)**2

def run(self,Inputs):
  """
    Method hook for RAVEN; combines function evaluation and stochasticity
    @ In, self, object, arbitrary object with variables as members
    @ In, Inputs, dict, same information in a dictionary
    @ Out, None
  """
  val = evaluate(self.x,self.y)
  stoch = Inputs.get('stochScale',1.0)
  self.ans = val + stoch

