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
# This test is used to compute the global sensitivities:
# The analytic solution for 5 input variables is:
# mean = 1.0, variance = 0.728
# S1 = S2 = S3 = 0.2747
# S12 = S23 = S31 = 0.0549
# S123 = 0.0110
#
# input variables are distributed as y ~ [0,1]
# from: Bruno Sudret, Global Sensitivity Analysis using Polynomial Chaos Expansion, Reliability Engineering and System Safety, 2008

import numpy as np

def evaluate(inp):
  return 1.0/(2.0**len(inp))*np.prod(list(1.+3.0*n**2 for n in inp))

def run(self,Input):
  self.ans  = evaluate(Input.values())

#
#  This model has analytic mean and variance and is documented in raven/docs/tests
#
