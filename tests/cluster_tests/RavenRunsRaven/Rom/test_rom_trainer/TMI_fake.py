# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
