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
# from https://en.wikipedia.org/wiki/Test_functions_for_optimization
#
# takes any number of input parameters x[i]
# returns value in "ans"
# optimal minimum at f(1,1,...,1,1) = 0, only minimum for up to 3 inputs
# note for 4 to 7 inputs, a second local minima exists near (-1,1,...,1), and it gets complicated after that
# parameter range is -inf <= x[i] <= inf

import numpy as np

def evaluate2d(X,Y):
  return 100*(Y-X*X)**2 + (X-1)**2

def evaluate(*args):#xs):
  xs = np.array(args)
  return np.sum( 100.*(xs[1:] - xs[:-1]**2)**2 + (xs[:-1] - 1.)**2 )

def run(self,Inputs):
  self.ans = evaluate(Inputs.values())

