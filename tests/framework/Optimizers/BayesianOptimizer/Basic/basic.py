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
# from Tutorials on GpyOpt library
#
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(9.49587, 6.37986) = -129.60286
# parameter range is 0 <= x,y <= 10
import math
import numpy as np
def evaluate(x,y):
  """
    evaluates the function at each given x,y.
    @ In, x, float, first input dimention.
    @ In, y, float, second input dimention.
    @ Out, evaluate, float, f(x,y)
  """
  return (x**2 + y**2)*(np.sin(x)**2 - np.cos(y))

def run(self,Inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, Inputs, dict, additional inputs
    @ Out, None
  """
  self.ans = evaluate(self.x,self.y)

if __name__ == '__main__':
  import sys
  x = float(sys.argv[1])
  y = float(sys.argv[2])
  print(evaluate(x, y))
