import math
import numpy

def run(self, Input):
  number_of_steps = 16
  self.time = numpy.zeros(number_of_steps)
  uniform = Input["uniform"]
  self.out = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*i
    time = self.time[i]
    self.out[i] = math.sin(time+uniform)
