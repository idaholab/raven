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
