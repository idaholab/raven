#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
#  This is a simple polynomial evaluation of all single-order combination polynomials
#    For instance, xyz + xy + xz + yz + x + y + z + 1
#
import numpy as np

def evaluate(inp):
  return np.prod(list(1.+n for n in inp))

def run(self,Input):
  self.ans  = evaluate(Input.values())

#
#  This model has analytic mean and variance and is documented in raven/docs/tests
#
