#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Simulates the attenuation of a beam through a purely-scattering medium with N distinct materials and unit length.
#     The uncertain inputs are the opacities.
#
import numpy as np

def evaluate(inp):
  if len(inp)>0: return np.exp(-sum(inp)/len(inp))
  else: return 1.0

def run(self,Input):
  self.ans  = evaluate(Input.values())

#
#  This model has analytic mean and variance documented in raven/docs/tests
#
