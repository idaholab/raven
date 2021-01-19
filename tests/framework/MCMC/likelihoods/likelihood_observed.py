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

def initialize(self, runInfoDict, inputFiles):
  """
    Method to generate the observed data
    @ In, runInfoDict, dict, the dictionary containing the runInfo
    @ In, inputFiles, list, the list of input files
    @ Out, None
  """
  np.random.seed(1086)
  self.mu = np.array([0, 0])
  self.cov = np.array([[1., 0.42], [0.42, 1.]])
  # 1000 observed data, ie. data with shape (1000, 2)
  self.samples = 1000
  self.data = np.random.multivariate_normal(self.mu, self.cov, size=self.samples)

def run(self, inputDict):
  """
    Method required by RAVEN to run this as an external model.
    log likelihood function
    @ In, self, object, object to store members on
    @ In, inputDict, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  self.rho = inputDict['rho']

  det = 1.0 - self.rho**2
  sum1 = - self.samples/2.0 * np.log(det)

  for i in range(self.samples):
    xi = self.data[i, 0]
    yi = self.data[i, 1]
    sum1 -= 0.5/det * (xi**2 - 2.0*self.rho*xi*yi + yi**2)

  self.pout = sum1
