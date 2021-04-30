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
def run(self, inputDict):
  """
    Method required by RAVEN to run this as an external model.
    log likelihood function
    @ In, self, object, object to store members on
    @ In, inputDict, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  mu = 0.
  sigma =  1.
  self.xin = inputDict['xin']
  if self.xin < 10. and self.xin > -10.:
    self.zout = -0.5*(self.xin-mu)**2/sigma**2
  else:
    self.zout = np.asarray([-1.0E6])
