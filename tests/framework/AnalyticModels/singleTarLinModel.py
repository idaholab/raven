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
#***************************************
#* Simple analytic test ExternalModule *
#***************************************
#
# Simulates a steady state linear model that maps $J-$parameters (i.e., $\mathbb{R}^J$) to k Responses
#
# External Modules
import numpy as np
##################
# Author: Mohammad Abdo (@Jimmy-INL)
def run(self,Input):
  """
    Method require by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  self.FOM1 = main(Input)

def main(Input):
  """
    Target Model evaluation method
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, y[:], floats, list of response values from the linear model $ y = Ax+b $
  """
  m = len([key for key in Input.keys() if 'o' in key]) # number of experiments
  n = len([par for par in Input.keys() if 'p' in par]) # number of parameters
  A = np.array([Input['o1']]).reshape(-1,n)
  b = Input['bT'].reshape(-1,1)
  x = np.atleast_2d(np.array([Input['p1'],Input['p2']])).reshape(-1,1)
  assert(np.shape(A)[1],np.shape(b)[0],n)
  assert(np.shape(A)[0],np.shape(b)[0],m)
  y = A @ x + b
  return y[:]