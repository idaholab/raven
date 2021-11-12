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
    self.pump_mass_flow_rate[i] = time*3.0 + zeroToOne + 1.0
