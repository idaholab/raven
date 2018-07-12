import numpy as np
import math
import random
from scipy.integrate import quad


def run(self,Input):
  # intput: t, T (max time)
  # output: outcome

  self.outcome_V1 = self.p_V1 * np.ones(Input['time'].size)


