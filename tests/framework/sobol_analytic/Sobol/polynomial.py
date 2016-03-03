#***************************************
#* Simple analytic test ExternalModule *
#***************************************
# This test is used to compute the global sensitivities:
# The analytic solutions is:
# mean = 1.0, variance = 0.72*
# S1=S2=S3 = 0.2747
# S12=S23=S31 = 0.0549
# S123 = 0.0110
#
import numpy as np

def evaluate(inp):
  return 1.0/(2.0**len(inp))*np.prod(list(1.+3.0*n**2 for n in inp))

def run(self,Input):
  self.ans  = evaluate(Input.values())

#
#  This model has analytic mean and variance and is documented in raven/docs/tests
#
