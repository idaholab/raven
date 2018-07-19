import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
import random

def RULmodel(a,b):
  # a = 2.31
  # b = 0.267
  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

  rvs = beta.rvs(a, b)
  time = 7300 * rvs
  return time


def run(self,Input):
  # lambda(t) = a + t*b
  # intput: alpha, beta, T (max time)
  # output: t, p

  self.p_P1 = np.zeros(Input['time'].size)

  for index,value in np.ndenumerate(Input['time']):
     self.p_P1[index[0]] = beta.cdf(value/7300,Input['alpha_P1'],Input['beta_P1'])


