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
# takes input parameters x,y
# returns value in "ans"
# optimal minimum at f(512,404.2319) = -959.6407
# parameter range is -512 <= x,y <= 512
import math
import numpy as np
def evaluate(x,y):
  return -(y+47)*math.sin(np.sqrt(abs(x/2+y+47))-x*math.sin(np.sqrt(abs(x-y-47))))

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

if __name__ == '__main__':
  import sys
  x = float(sys.argv[1])
  y = float(sys.argv[2])
  print(evaluate(x, y))
