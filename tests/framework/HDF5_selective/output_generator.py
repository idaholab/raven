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
