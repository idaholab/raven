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

  self.p_V2 = np.zeros(Input['time'].size)

  for index,value in np.ndenumerate(Input['time']):
    #self.p_V2[index[0]] = quad(pdfFailure, 0, value, args=(Input['a_V2'],Input['b_V2']))[0]
    self.p_V2[index[0]] = 1. - quad(timeDepLambda, 0, value, args=(Input['a_V2'],Input['b_V2']))[0]

