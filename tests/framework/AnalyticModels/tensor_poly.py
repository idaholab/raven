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
#  This is a simple polynomial evaluation of all single-order combination polynomials
#    For instance, xyz + xy + xz + yz + x + y + z + 1
#
import numpy as np

def evaluate(inp):
  return np.prod(list(1.+n for n in inp))

def run(self,Input):
  self.ans  = evaluate(Input.values())

#
#  This model has analytic mean and variance and is documented in raven/docs/tests
#
