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
# documented in analytic functions

def evaluate(x,y):
  first = 1. + (x + y + 1.)**2 * (19. - 14.*x + 3.*x*x - 14.*y + 6.*x*y + 3.*y*y)
  second = 30. + (2.*x - 3.*y)**2 * (18. - 32.*x + 12.*x*x + 48.*y - 36.*x*y + 27.*y*y)
  return first*second

def run(self,Inputs):
  self.ans = evaluate(self.x,self.y)

