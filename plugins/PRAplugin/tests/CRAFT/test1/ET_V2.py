import numpy as np
import math
import random
from scipy.integrate import quad


def run(self,Input):
  # intput: t, T (max time)
  # output: outcome

  if self.t_V2 > self.T:
    self.outcome_V2 = 0 # OK
  else:
    self.outcome_V2 = 1 # SD

  self.p_V2_ET = self.p_V2
  self.t_V2_ET = self.t_V2