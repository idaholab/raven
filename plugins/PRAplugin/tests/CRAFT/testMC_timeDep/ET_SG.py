import numpy as np
import math
import random
from scipy.integrate import quad


def run(self,Input):
  # intput: t, T (max time)
  # output: outcome

  self.outcome_SG = self.p_SG * np.ones(Input['time'].size)
