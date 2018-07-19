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

  self.p_V1 = np.zeros(Input['time'].size)

  for index,value in np.ndenumerate(Input['time']):
    #self.p_V1[index[0]] = quad(pdfFailure, 0, value, args=(Input['a_V1'],Input['b_V1']))[0]
    self.p_V1[index[0]] = 1. - math.exp(-quad(timeDepLambda, 0, value, args=(Input['a_V1'],Input['b_V1']))[0])
