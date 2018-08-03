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
# This model tests the stochastic collocation with gamma distribution

import numpy as np

def eval(inp,exp):
  return sum(n**exp for n in inp)

def run(self,Input):
  self.ans = eval((self.x1,self.x2),1.0)
  self.ans2 = eval((self.x1,self.x2),2.0)

#
#  This model has analytic mean and variance documented in raven/docs/tests
#
