import numpy as np
import math
import random
from scipy.integrate import quad


def run(self,Input):
  # intput: t_P1, t_P2, T (max time)
  # output: outcome

  if self.t_P1 > self.T:
    self.outcome_P1P2 = 0 # OK
    self.p_P1P2_ET = self.p_P1
    self.t_P1P2_ET = self.T + 1.0
  elif self.t_P1 < self.T and self.t_P2 > self.T:
    self.outcome_P1P2 = 2 # Power_red
    self.p_P1P2_ET = self.p_P1
    self.t_P1P2_ET = self.t_P1
  elif self.t_P1 < self.T and self.t_P2 < self.T:
    self.outcome_P1P2 = 3 # SD + reg
    self.p_P1P2_ET = self.p_P1 * self.p_P2
    self.t_P1P2_ET = self.t_P1
  else:
    print('error')


