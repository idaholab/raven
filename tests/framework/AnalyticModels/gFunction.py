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
# From Satelli and Sobol 1995, About the Use of Rank Transformation in Sensitivity Analysis of Model Output,
#    Reliability Engineering and System Safety, 50, 225-239
#
#
import numpy as np
def g(x,a):
  return (abs(4.*x-2.)+a)/(1.+a)

def run(self,Input):
  #a_n stored in "tuners"
  tuners = {'x1':1,
            'x2':2,
            'x3':5,
            'x4':10,
            'x5':20,
            'x6':50,
            'x7':100,
            'x8':500}
  tot = 1.
  for key,val in Input.items():
    tot*=g(val,tuners[key])
  self.ans = tot

#
#  This model has analytic mean and variance documented in raven/docs/tests
#
