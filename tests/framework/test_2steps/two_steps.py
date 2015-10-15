import math
import numpy

def run(self, Input):
  number_of_steps = 16
  self.time = numpy.zeros(number_of_steps)
  #Get inputs
  Gauss1 = Input["Gauss1"]
  auxBackupTimeDist = Input["auxBackupTimeDist"]
  Gauss2 = Input["Gauss2"]
  CladFailureDist = Input["CladFailureDist"]
  self.out1 = numpy.zeros(number_of_steps)
  self.out2 = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*i
    time = self.time[i]
    #calculate outputs
    self.out1 = time + Gauss1+auxBackupTimeDist + Gauss2 + CladFailureDist
    self.out2 = time*Gauss1*auxBackupTimeDist*Gauss2*CladFailureDist
