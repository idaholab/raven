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
# From Ishigami and Homma, 1990
#     Analytic UQ problem with mean, variance, sensitivities
#
import numpy as np

def run(self,Input):
  self.ans  = np.sin(self.x1) + 7.*np.sin(self.x2)**2 + 0.1*self.x3**4*np.sin(self.x1)

# included in Sudret, Global Sensitivity Analysis (2008) and many other places
# x1, x2, x3 are all distributed uniformly on [-pi, pi], a=7, b=0.1
#
# The variance is given by
#   a^2/8 + b*pi^4/5 + b^2*pi^8/18 + 1/2 = 13.8446
# Partial variance are given as
# D_1 = b*pi^4/5 + b^2*pi^8/50 + 1/2     = 4.34589
# D_2 = a^2/8 = 49/8                     = 6.125
# D_3 = 0                                = 0
# D_12 = D_23 = 0                        = 0
# D_13 = 8*b^2*pi^8/225                  = 3.3737
# D_123 = 0                              = 0
#
# Sobol sensitivities are the above divided by the total, which is
# S_1   = 0.3138
# S_2   = 0.4424
# S_3   = 0
# S_12  = 0
# S_13  = 0.2436
# S_23  = 0
# S_123 = 0
#
#  This model has analytic mean and variance documented in raven/docs/tests
#
