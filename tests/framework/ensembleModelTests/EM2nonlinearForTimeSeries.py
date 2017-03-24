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
import numpy as np
def run(self, Input):
  self.averageTemperature = (self.leftTemperature + self.rightTemperature)/2.0
  self.timeFromFirstModel = np.linspace(0.0, 100.0, 10)
  self.k = np.zeros(len(self.timeFromFirstModel))
  for ts in range(len(self.timeFromFirstModel)):
    self.k[ts] = 38.23/(129.2 + self.averageTemperature) + 0.6077E-12*self.averageTemperature
    self.k[ts] = self.solutionK[ts]+ self.k[ts] + self.k[ts]*self.timeFromFirstModel[ts]/100.0
