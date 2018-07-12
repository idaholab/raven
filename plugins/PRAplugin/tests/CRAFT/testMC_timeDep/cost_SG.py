import random
import numpy as np

def run(self,Input):
  # intput: 
  # output: 

  numberDaysSD = float(random.randint(30,60))
  costPerDay   = 0.8 + 0.4 * random.random()
  cost_SG = numberDaysSD * costPerDay

  self.cost_SG = cost_SG * np.ones(Input['time'].size)