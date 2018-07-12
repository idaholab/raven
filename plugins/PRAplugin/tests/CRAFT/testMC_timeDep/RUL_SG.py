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

  self.p_SG = np.zeros(Input['time'].size)

  for index,value in np.ndenumerate(Input['time']):
     self.p_SG[index[0]] = float(RULmodel(Input['alpha_SG'], Input['beta_SG'], value))
       
