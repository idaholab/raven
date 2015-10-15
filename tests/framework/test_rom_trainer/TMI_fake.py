import math
import numpy

def run(self, Input):
  number_of_steps = 16
  self.time = numpy.zeros(number_of_steps)
  DeltaTimeScramToAux = Input["DeltaTimeScramToAux"]
  DG1recoveryTime = Input["DG1recoveryTime"]
  self.CladTempThreshold = numpy.zeros(number_of_steps)
  self.UpperPlenumEnergy= numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*i
    self.CladTempThreshold[i] = self.time[i]*50.0 + DeltaTimeScramToAux*200.0 + DG1recoveryTime*500.0
    self.UpperPlenumEnergy[i] = self.time[i]*5.0 + DeltaTimeScramToAux*30.0 + DG1recoveryTime*40.0 + DeltaTimeScramToAux*DG1recoveryTime*5.0
