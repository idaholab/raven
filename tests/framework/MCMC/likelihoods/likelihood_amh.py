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

import scipy.stats as st
import numpy as np

def run(self, inputDict):
  """
    Method required by RAVEN to run this as an external model.
    log likelihood function
    @ In, self, object, object to store members on
    @ In, inputDict, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  self.xin = inputDict['xin']
  self.yin = inputDict['yin']
  mus = np.array([5, 5])
  sigmas = np.array([[1, .9], [.9, 1]])
  self.zout = st.multivariate_normal.pdf(np.concatenate((self.xin, self.yin)), mean=mus, cov=sigmas)
