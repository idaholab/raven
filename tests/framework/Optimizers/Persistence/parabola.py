#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Computes an inverse parabola with maximum value at x=0 of 1.0.
#
import numpy as np

#static seed
np.random.seed(42)

def base(values):
  return -sum(x**2 for x in values)+1.

def evaluate(values):
  return base(values)

def run(self,Input):
  """
    Method require by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  self.ans = evaluate(Input.values())
