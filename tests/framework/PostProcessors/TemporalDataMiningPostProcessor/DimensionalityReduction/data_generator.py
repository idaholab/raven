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
from sklearn import datasets
import numpy as np

def run(self, Input):
  number_of_timesteps = 21
  end_time = 40;
  self.Time = np.linspace(0, end_time, number_of_timesteps)

  id = int(Input['n'])

  iris = datasets.load_iris()
  x1 = iris.data[id,0]
  x2 = iris.data[id,1]
  x3 = iris.data[id,2]
  x4 = iris.data[id,3]

  self.x1 = np.zeros(number_of_timesteps)
  self.x2 = np.zeros(number_of_timesteps)
  self.x3 = np.zeros(number_of_timesteps)
  self.x4 = np.zeros(number_of_timesteps)

  for j in range(number_of_timesteps):
    t = self.Time[j]
    self.x1[j] = x1*np.exp(-t/15)*1.0 + np.sin(2*np.pi*t/end_time)*1.0
    self.x2[j] = x2*np.exp(-t/15)*1.0 + np.cos(2*np.pi*t/end_time)*1.0+1.0
    self.x3[j] = x3*np.exp(-t/15)*1.0 - np.cos(2*np.pi*t/end_time)*1.0 + np.sin(2*np.pi*t/end_time)*1.0
    self.x4[j] = x4*np.exp(-t/15)*1.0 - np.cos(2*np.pi*t/end_time)*1.0 - np.sin(2*np.pi*t/end_time)*1.0
