import math
def run(self, Input):
  # self.leftTemperature (boundary condition - left) self.rightTemperature (boundary condition - right)
  self.averageTemperature = (self.leftTemperature + self.rightTemperature)/2.0
  self.k = 38.23/(129.2 + self.averageTemperature) + 0.6077E-12*self.averageTemperature