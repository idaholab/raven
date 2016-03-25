import math
import numpy

def run(self, Input):
  number_of_steps = 16
  self.time = numpy.zeros(number_of_steps)
  gauss = Input["gauss"]
  uniform = Input["uniform"]
  self.out1 = numpy.zeros(number_of_steps)
  self.out2 = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*i
    time = self.time[i]
    self.out1[i] = math.sin(time+gauss+uniform)
    self.out2[i] = time*gauss*uniform
