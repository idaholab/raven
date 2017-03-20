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
import numpy as np
import random
random.seed(64616)


def run(self, Input):
  cluster_starts = 3

  cluster_x = [2.0, 4.0, 4.5]
  cluster_y = [2.0, 1.0, 4.0]

  cluster_dx = [0.1, 0.05, -0.05]
  cluster_dy = [0.1, 0.2, -0.2]

  number_of_timesteps = 100

  cluster = int(Input['cluster'])
  point_x = cluster_x[cluster] + Input['dx']
  point_y = cluster_y[cluster] + Input['dy']
  point_dx = cluster_dx[cluster] + Input['ddx']
  point_dy = cluster_dy[cluster] + Input['ddy']


  self.Time = np.zeros(number_of_timesteps)
  self.x = np.zeros(number_of_timesteps)
  self.y = np.zeros(number_of_timesteps)

  for j in range(number_of_timesteps):
    Time = j*0.1
    self.Time[j] = Time
    cx,cy = point_x+Time*point_dx,point_y+Time*point_dy
    self.x[j] = cx
    self.y[j] = cy
