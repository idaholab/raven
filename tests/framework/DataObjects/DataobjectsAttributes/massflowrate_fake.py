import math
import numpy

def run(self, Input):
  number_of_steps = 20
  self.time = numpy.zeros(number_of_steps)
  dt = 0.0001
  Tw = Input["Tw"]
  dummy1 = Input["Dummy1"]
  self.pipe_Area = numpy.zeros(number_of_steps)
  self.pipe_Tw = numpy.zeros(number_of_steps)
  self.pipe_Hw = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = dt*i
    time = self.time[i]
    self.pipe_Area[i] = 0.25 + time
    self.pipe_Tw[i] = Tw + time
    self.pipe_Hw[i] = dummy1 + time
