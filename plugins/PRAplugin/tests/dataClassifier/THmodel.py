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

def run(self,Input):

  self.ACC_status = Input['ACC_status'][0]
  self.time_LPI   = Input['time_LPI'][0]
  self.time_LPR   = Input['time_LPR'][0]

  timeToCD_LPI = 6.
  timeToCD_LPR = 11.

  if self.ACC_status == 1.:
    self.out = 1.
    self.LPI_status = 1.
    self.LPR_status = 1.
  else:
    self.LPI_act = self.time_LPI + 1.
    if self.LPI_act > timeToCD_LPI:
      self.out = 1.
      self.LPI_status = 1.
      self.LPR_status = 1.
    else:
      self.LPI_status = 0.
      self.LPR_act = self.LPI_act + self.time_LPR
      if self.LPR_act > timeToCD_LPR:
        self.out = 1.
        self.LPR_status = 1.
      else:
        self.out = 0.
        self.LPR_status = 0.



