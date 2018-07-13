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

  self.p_SG = np.zeros(len(self.time))

  for ts in range(len(self.time)):
    self.p_SG[ts] = float(RULmodel(Input['alpha_SG'], Input['beta_SG'], self.time[ts]))