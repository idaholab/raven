from scipy.stats import norm
from scipy.integrate import quad
import random
import numpy as np

def RULmodel(mu,sigma,time):
  # a = 2.31
  # b = 0.267
  prob = norm.cdf(time, mu, sigma)
  return prob


def run(self,Input):
  # intput: alpha, beta
  # output: t, p
  self.time = Input['opPower']
  self.p_SG = np.zeros(len(self.time))

  for index,value in np.ndenumerate(self.time):
     self.p_SG[index[0]] = float(RULmodel(Input['alpha_SG'], Input['beta_SG'], value))
