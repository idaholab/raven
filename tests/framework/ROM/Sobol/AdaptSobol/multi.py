#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
import numpy as np

def evaluate(inp):
  return np.prod(list(1.+n for n in inp))

def evaluate2(inp):
  if len(inp)>0: return np.exp(-sum(inp)/len(inp))
  else: return 1.0

def run(self,Input):
  self.ans  = evaluate (Input.values())
  self.ans2 = evaluate2(Input.values())

#
# documentation for these tests can be found in raven/docs/tests/tensor_poly.tex
#                                           and raven/docs/tests/attenuate.tex
#
