import math
import numpy as np
def run(self, Input):
  self.averageTemperature = (self.leftTemperature + self.rightTemperature)/2.0
  self.timeFromFirstModel = np.linspace(0.0, 100.0, 10)
  self.k = np.zeros(len(self.timeFromFirstModel))
  for ts in range(len(self.timeFromFirstModel)):
    self.k[ts] = 38.23/(129.2 + self.averageTemperature) + 0.6077E-12*self.averageTemperature
    self.k[ts] = self.solutionK[ts]+ self.k[ts] + self.k[ts]*self.timeFromFirstModel[ts]/100.0
