import numpy as np
import math
import random
from scipy.integrate import quad

def timeDepLambda(t,a,b):
  return a+t*b

def pdfFailure(t,a,b):
  first  = timeDepLambda(t,a,b)
  second = math.exp(-quad(timeDepLambda, 0, t, args=(a,b))[0])
  return first*second

def run(self,Input):
  # lambda(t) = a + t*b
  # intput: a_V1, b_V1, T (max time)
  # output: t_V1, p_V1

  status = random.random()
  if status < 0.5:
    # Sample t from [0,T] (unformly distributed)
    self.t_V1 = random.random()*Input['T']
    # Calculate p of occurence in [0,t]
    self.p_V1 = quad(pdfFailure, 0, self.t_V1, args=(Input['a_V1'],Input['b_V1']))[0]
  else:
    self.t_V1 = Input['T'] + 1.
    self.p_V1 = quad(pdfFailure, self.t_V1, np.inf, args=(Input['a_V1'],Input['b_V1']))[0]

