import math
import numpy

def run(self, Input):
  number_of_steps = 16
  self.time = numpy.zeros(number_of_steps)
  zeroToOne = Input["zeroToOne"]
  self.pipe1_Hw = numpy.zeros(number_of_steps)
  self.pipe1_Dh = numpy.zeros(number_of_steps)
  self.pipe1_Area = numpy.zeros(number_of_steps)
  self.pump_mass_flow_rate = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*i
    time = self.time[i]
    self.pipe1_Hw[i] = time+20.0
    self.pipe1_Dh[i] = time*3.0 + 40.0
    self.pipe1_Area[i] = time*2.0 + 10.0 + zeroToOne
    self.pump_mass_flow_rate = time*3.0 + zeroToOne
