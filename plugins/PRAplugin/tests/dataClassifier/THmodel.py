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
  """
    This method computes the final status of this simplified LB LOCA event
    @ In, Input, dict, dictionary of inputs from RAVEN
    @ Out, None
  """
  self.ACCstatus = Input['ACC_status'][0]
  self.timeLPI   = Input['time_LPI'][0]
  self.timeLPR   = Input['time_LPR'][0]

  timeToCDLPI = 6.
  timeToCDLPR = 11.

  if self.ACCstatus == 1.:
    self.out = 1.
    self.LPI_status = 1.
    self.LPR_status = 1.
  else:
    self.LPIact = self.timeLPI + 1.
    if self.LPIact > timeToCDLPI:
      self.out = 1.
      self.LPI_status = 1.
      self.LPR_status = 1.
    else:
      self.LPI_status = 0.
      self.LPRact = self.LPIact + self.timeLPR
      if self.LPRact > timeToCDLPR:
        self.out = 1.
        self.LPR_status = 1.
      else:
        self.out = 0.
        self.LPR_status = 0.



