import math
import numpy

def run(self, Input):
  number_of_steps = 2
  self.time = numpy.zeros(number_of_steps)
  #Get inputs
  x = Input["x"]
  self.out1 = numpy.zeros(number_of_steps)
  self.out2 = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = float(i)
    #calculate outputs
    self.out1[i] = x
    self.out2[i] = self.time[i]
