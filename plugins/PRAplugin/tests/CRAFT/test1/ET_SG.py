import numpy as np
import math
import random
from scipy.integrate import quad


def run(self,Input):
  # intput: t, T (max time)
  # output: outcome

  if self.t_SG > self.T:
    self.outcome_SG = 0 # OK
  else:
    self.outcome_SG = 1 # SD

  self.p_SG_ET = self.p_SG
  self.t_SG_ET = self.t_SG
