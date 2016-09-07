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
  y1 = Input['y1']
  y2 = Input['y2']
  self.ans  = evaluate([y1,y2])

#
#  This model has analytic mean and variance documented in raven/docs/tests
#
