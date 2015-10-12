import math
import numpy

def run(self, Input):
  number_of_steps = 16
  self.time = numpy.zeros(number_of_steps)
  DeltaTimeScramToAux = Input["DeltaTimeScramToAux"]
  DG1recoveryTime = Input["DG1recoveryTime"]
  self.CladTempThreshold = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*i
    self.CladTempThreshold[i] = self.time[i]*50.0 + DeltaTimeScramToAux*200.0 + DG1recoveryTime*500.0
