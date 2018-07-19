import numpy as np
import math
import random
from scipy.integrate import quad

def timeDepLambda(t,a,b):
  exponent = a+t*b
  return 10**exponent

def pdfFailure(t,a,b):
  first  = timeDepLambda(t,a,b)
  second = math.exp(-quad(timeDepLambda, 0, t, args=(a,b))[0])
  return first*second

def run(self,Input):
  # lambda(t) = a + t*b
  # intput: a_V2, b_V2, T (max time)
  # output: t_V2, p_V2

  status = random.random()
  if status < 0.5:
    # Sample t from [0,T] (unformly distributed)
    self.t_V2 = random.random()*Input['T']
    # Calculate p of occurence in [0,t]
    self.p_V2 = quad(pdfFailure, 0, self.t_V2, args=(Input['a_V2'],Input['b_V2']))[0]
  else:
    self.t_V2 = Input['T'] + 1.
    self.p_V2 = quad(pdfFailure, self.t_V2, 1.E6, args=(Input['a_V2'],Input['b_V2']))[0]
