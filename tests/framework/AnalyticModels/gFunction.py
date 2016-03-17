#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# From Satelli and Sobol 1995, About the Use of Rank Transformation in Sensitivity Analysis of Model Output,
#    Reliability Engineering and System Safety, 50, 225-239
#
#
import numpy as np
def g(x,a):
  return (abs(4.*x-2.)+a)/(1.+a)

def run(self,Input):
  #a_n stored in "tuners"
  tuners = {'x1':1,
            'x2':2,
            'x3':5,
            'x4':10,
            'x5':20,
            'x6':50,
            'x7':100,
            'x8':500}
  tot = 1.
  for key,val in Input.items():
    tot*=g(val,tuners[key])
  self.ans = tot

#
#  This model has analytic mean and variance documented in raven/docs/tests
#
