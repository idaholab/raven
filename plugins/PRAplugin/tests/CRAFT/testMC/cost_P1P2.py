import random

def run(self,Input):
  # intput:
  # output:

  if self.outcome_P1P2 == 0:
    self.cost_P1P2 = 0.
  elif self.outcome_P1P2 == 2:
    numberDaysPred   = float(random.randint(2,7))
    costPerDayPred   = 0.3 * (0.8 + 0.4 * random.random())
    self.cost_P1P2   = numberDaysPred * costPerDayPred
  elif self.outcome_P1P2 == 3:
    numberDaysSD = float(random.randint(7,21))
    costPerDaySD = 0.8 + 0.4 * random.random()
    costSD       = numberDaysSD * costPerDaySD

    costPerDayReg = 0.2 + 0.1 * random.random()
    costReg       = numberDaysSD * costPerDayReg

    self.cost_P1P2 = costSD + costReg
  else:
    print('error costP1P2')

  self.p_P1P2_cost = self.p_P1P2_ET
  self.t_P1P2_cost = self.t_P1P2_ET
