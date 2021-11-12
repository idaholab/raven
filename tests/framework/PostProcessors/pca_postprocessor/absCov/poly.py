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
import numpy as np
from itertools import product

def eval(Y):
  A = np.array([1.0,2.2,-3.3,-0.2,11])
  B = np.array([-3,4,8,-0.1,66])
  return np.dot(A,(Y+B))

def run(self,Input):
  self.ans = eval([Input['x1'],Input['x2'],Input['x3'],Input['x4'],Input['x5']])
