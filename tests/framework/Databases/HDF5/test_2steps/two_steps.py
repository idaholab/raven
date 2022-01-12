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
  #Get inputs
  Gauss1 = Input["Gauss1"]
  auxBackupTimeDist = Input["auxBackupTimeDist"]
  Gauss2 = Input["Gauss2"]
  CladFailureDist = Input["CladFailureDist"]
  self.out1 = numpy.zeros(number_of_steps)
  self.out2 = numpy.zeros(number_of_steps)
  for i in range(len(self.time)):
    self.time[i] = 0.25*float(i)
    time = self.time[i]
    #calculate outputs
    self.out1[i] = time + Gauss1+auxBackupTimeDist + Gauss2 + CladFailureDist
    self.out2[i] = time*Gauss1*auxBackupTimeDist*Gauss2*CladFailureDist
