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
def run(self, Input):
  # self.leftTemperature (boundary condition - left) self.rightTemperature (boundary condition - right)
  self.averageTemperature = (self.leftTemperature + self.rightTemperature)/2.0
  self.k = 38.23/(129.2 + self.averageTemperature) + 0.6077E-12*self.averageTemperature
