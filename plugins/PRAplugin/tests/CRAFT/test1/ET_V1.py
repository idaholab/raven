import numpy as np
import math
import random
from scipy.integrate import quad


def run(self,Input):
  # intput: t, T (max time)
  # output: outcome

  if self.t_V1 > self.T:
    self.outcome_V1 = 0 # OK
  else:
    self.outcome_V1 = 1 # SD

  self.p_V1_ET = self.p_V1

