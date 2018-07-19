import random
import numpy as np

def run(self,Input):
  # intput:
  # output:

  self.cost_P1P2_0 = np.zeros(Input['time'].size)

  numberDaysPred   = float(random.randint(2,7))
  costPerDayPred   = 0.3 * (0.8 + 0.4 * random.random())
  cost_P1P2_1   = numberDaysPred * costPerDayPred
  self.cost_P1P2_1 = cost_P1P2_1 * np.ones(Input['time'].size)

  numberDaysSD = float(random.randint(7,21))
  costPerDaySD = 0.8 + 0.4 * random.random()
  costSD       = numberDaysSD * costPerDaySD

  costPerDayReg = 0.2 + 0.1 * random.random()
  costReg       = numberDaysSD * costPerDayReg

  self.cost_P1P2_2 = (costSD + costReg) * np.ones(Input['time'].size)
