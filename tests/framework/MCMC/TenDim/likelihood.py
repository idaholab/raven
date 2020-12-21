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
  self.dim = 10
  seed = 1086
  np.random.seed(seed)
  self.cov = 10**(np.random.randn(self.dim)*1.5)
  self.mu = st.norm(loc=0, scale=10).rvs(self.dim)

def run(self, inputDict):
  """
    Method required by RAVEN to run this as an external model.
    log likelihood function
    @ In, self, object, object to store members on
    @ In, inputDict, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  vars = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
  xin = []
  for var in vars:
    xin.extend(inputDict[var])
  xin = np.asarray(xin)
  if np.all(xin < 500) and np.all(xin > -500):
    zout = st.multivariate_normal(mean=self.mu, cov=self.cov).logpdf(xin)
  else:
    zout = -1.0E6
  self.zout = np.atleast_1d(zout)
