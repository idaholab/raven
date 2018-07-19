import random
import numpy as np

def unavailability(rho, compLambda, delta, T):
  # see J.K. Vaurio RESS vol.49, pp. 23-36 (1995)
  unavail = rho + delta/T + 0.5*compLambda*T
  return unavail

def run(self,Input):
  # intput: rho,compLambda, delta, T
  # output: unavailability
  rho = 1.E-3
  compLambda = 1.E-5
  T = 1000.
  #self.delta = 0.2

  self.p_P2 = float(unavailability(rho,compLambda,Input['delta_P2'],T)) * np.ones(Input['time'].size)
