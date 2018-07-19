import random

def run(self,Input):
  # intput:
  # output:

  if self.outcome_V1 == 0:
    self.cost_V1 = 0.
  else:
    numberDaysSD = float(random.randint(10,30))
    costPerDay   = 0.8 + 0.4 * random.random()
    self.cost_V1 = numberDaysSD * costPerDay

  self.p_V1_cost = self.p_V1_ET
  self.t_V1_cost = self.t_V1_ET
