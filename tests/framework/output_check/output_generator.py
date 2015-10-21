import math
import numpy

def run(self, Input):
  number_of_steps = 16
  self.time = numpy.zeros(number_of_steps)
  var1 = Input["var1"]
  var2 = Input["var2"]
  self.sine = numpy.zeros(number_of_steps)
  self.cosine = numpy.zeros(number_of_steps)
  self.square = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*i
    time = self.time[i]
    self.sine[i] = math.sin(time+var1+var2)
    self.cosine[i] = math.cos(time+var1+var2)
    self.square[i] = time**2.0+var1+var2
