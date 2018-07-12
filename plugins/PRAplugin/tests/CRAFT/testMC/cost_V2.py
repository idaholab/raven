import random

def run(self,Input):
  # intput: 
  # output: 

  if self.outcome_V2 == 0:
    self.cost_V2 = 0.
  else:
    numberDaysSD = float(random.randint(10,30))
    costPerDay   = 0.8 + 0.4 * random.random()
    self.cost_V2 = numberDaysSD * costPerDay 

  self.p_V2_cost = self.p_V2_ET
  self.t_V2_cost = self.t_V2_ET