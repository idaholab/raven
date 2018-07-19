from scipy.stats import norm
from scipy.integrate import quad
import random

def RULmodel(mu,sigma,time):
  # a = 2.31
  # b = 0.267
  prob = norm.cdf(time, mu, sigma)
  return prob


def run(self,Input):
  # intput: alpha, beta
  # output: t, p

  status = random.random()
  if status < 0.5:
    # Sample t from [0,T] (unformly distributed)
    self.t_SG = random.random()*Input['T']
    self.p_SG = float(RULmodel(Input['alpha_SG'], Input['beta_SG'], self.t_SG))
  else:
    self.t_SG = Input['T']+ 1.0
    self.p_SG = 1.0 - float(RULmodel(Input['alpha_SG'], Input['beta_SG'], self.t_SG))

