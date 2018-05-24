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
    This method computes the final status of this simplified LB LOCA event in time dependent mode
    @ In, Input, dict, dictionary of inputs from RAVEN
    @ Out, None
  """
  self.ACC      = Input['ACC_sim']
  self.time_LPI = Input['time_LPI']
  self.time_LPR = Input['time_LPR']

  timeToCD_LPI = 6.
  timeToCD_LPR = 11.

  if self.ACC == 1.:
    self.time       = np.array([ 0, 1, 2])
    self.temp       = np.array([10,15,20])
    self.out        = np.array([0.,0.,1.])
    self.ACC_status = np.array([0.,1.,1.])
    self.LPI_status = np.array([0.,0.,0.])
    self.LPR_status = np.array([0.,0.,0.])
  else:
    self.LPI_act = self.time_LPI + 1.
    if self.LPI_act > timeToCD_LPI:
      self.time       = np.array([ 0, 1, 2, 3, 4, 5])
      self.temp       = np.array([10,10,10,10,15,20])
      self.out        = np.array([0.,0.,0.,0.,0.,1.])
      self.ACC_status = np.array([0.,0.,0.,0.,0.,0.])
      self.LPI_status = np.array([0.,0.,0.,0.,0.,1.])
      self.LPR_status = np.array([0.,0.,0.,0.,0.,0.])
    else:
      self.LPR_act = self.LPI_act + self.time_LPR
      if self.LPR_act > timeToCD_LPR:
        self.time       = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.temp       = np.array([10,10,10,10,10,10,10,15,20])
        self.out        = np.array([0.,0.,0.,0.,0.,0.,0.,0.,1.])
        self.ACC_status = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.LPI_status = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.LPR_status = np.array([0.,0.,0.,0.,0.,0.,0.,0.,1.])
      else:
        self.time       = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10])
        self.temp       = np.array([10,10,10,10,10,10,10,10,10,10,10])
        self.out        = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.ACC_status = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.LPI_status = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.LPR_status = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])



