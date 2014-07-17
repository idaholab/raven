
''' from wikipedia: dx/dt = sigma*(y-x)  ; dy/dt = x*(rho-z)-y  dz/dt = x*y-beta*z  ; '''

import numpy as np


def run(self,Input):
  self.prod   = 10*self.ThExp*self.GrainRad
  self.sum    = 5*self.ThExp -0.6*self.GrainRad
  self.sin    = np.sin(self.ThExp/5.e-7)*np.sin((self.GrainRad-0.5)*10)
#  if self.sin == np.NZERO:self.sin=np.zeros(1) 

