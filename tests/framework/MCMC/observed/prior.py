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
A non-informative prior for covariance matrices is the Jeffreys prior
(see Gelman et al., 1995), which is of the form 1/det(Cov)^(3/2).
For example, given covariance Cov = [[1, rho], [rho,1]]
"""
def pdf(self):
  """
    Method required for "probabilityFunction" used by MCMC sampler
    that is used to define the prior probability function
    @ In, None
    @ Out, priorPDF, float, the prior pdf value
  """
  priorPDF = 1/(1-self.rho**2)**(3/2)
  return priorPDF
