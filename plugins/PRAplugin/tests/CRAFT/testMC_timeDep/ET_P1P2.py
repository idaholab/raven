import numpy as np
import math
import random
from scipy.integrate import quad


def run(self,Input):
  # intput: t_P1, t_P2, T (max time)
  # output: outcome

  self.outcome_P1P2_0 = np.ones(Input['time'].size)
  self.outcome_P1P2_1 = np.ones(Input['time'].size)
  self.outcome_P1P2_2 = np.ones(Input['time'].size)

  for index,value in np.ndenumerate(Input['time']):
    self.outcome_P1P2_0[index[0]] = (1.-Input['p_P1'][index[0]]) * (1.-Input['p_P2'][index[0]])
    self.outcome_P1P2_1[index[0]] = Input['p_P1'][index[0]]*(1.-Input['p_P2'][index[0]])
    self.outcome_P1P2_2[index[0]] = Input['p_P1'][index[0]] * Input['p_P2'][index[0]]



