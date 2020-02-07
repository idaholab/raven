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
# optimal minimum at f(3,0.5) = 0
# parameter range is -4.5 <= x,y <= 4.5

def evaluate(x,y):
  return (1.5 - x + x*y)**2 + (2.25 - x + x*y*y)**2 + (2.625 - x + x*y*y*y)**2

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

def grad(x, y):
  return [gradX(x, y), gradY(x, y)]

def gradX(x, y):
  tot = 0
  consts = (1.5, 2.25, 2.625)
  for i in range(1, 4):
    tot += 2 * (y**i - 1) * (x * (y**i - 1) + consts[i-1])
  return tot

def gradY(x, y):
  tot = 0
  consts = (1.5, 2.25, 2.625)
  for i in range(1, 4):
    tot += 2 * i * x * (x * (y**i - 1) + consts[i-1])
  return tot

if __name__ == '__main__':
  import sys
  x = float(sys.argv[1])
  y = float(sys.argv[2])
  print(evaluate(x, y), grad(x, y))
