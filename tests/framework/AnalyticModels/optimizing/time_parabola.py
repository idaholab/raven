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


"""
Optimizing function.  Has minimum at x=0, (t-y)=0 for each value of t.
"""

import numpy as np

def evaluate(x,y,t):
  return x*x + np.sum((t-y)**2*np.exp(-t))

def run(self,Input):
  # "x" is scalar, "ans" and "y" depend on vector "t"
  self.t = np.linspace(-5,5,11)
  self.ans = evaluate(self.x,self.y,self.t)

