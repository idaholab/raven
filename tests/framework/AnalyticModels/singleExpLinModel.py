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

# A = np.array([[2, -3],[1,8],[-5, -5]])
# b = np.array([[0],[0],[0]])

def run(self,Input):
  """
    Method require by RAVEN to run this as an external model.
    @ In, self, object, object to store members on
    @ In, Input, dict, dictionary containing inputs from RAVEN
    @ Out, None
  """
  self.F1 = main(Input)

def main(Input):
  # y = A @ np.array(list(Input.values())).reshape(-1,1) + b
  m = len([key for key in Input.keys() if 'e' in key]) # number of experiments
  n = len([par for par in Input.keys() if 'p' in par]) # number of parameters
  A = np.array([Input['e1']]).reshape(-1,n)
  b = Input['bE'].reshape(-1,1)
  x = np.atleast_2d(np.array([Input['p1'],Input['p2']])).reshape(-1,1)
  assert(np.shape(A)[1],np.shape(b)[0])
  assert(np.shape(A)[0],np.shape(b)[0],m)
  y = A @ x + b
  return y[:]


if __name__ == '__main__':
  Input = {}
  Input['e1'] = [2,-3]
  Input['bE'] = np.array([[0],[0],[0]])
  Input['x1'] = 5.5
  Input['x2'] = 8
  a = main(Input)
  print(a)
