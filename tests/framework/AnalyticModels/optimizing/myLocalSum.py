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

def evaluate(inputs):
  """
    Evaluates a weighted sum function.
    $sum = \Sigma_{i=0} (i+1)*x_{i}$

    min with replacement = n*(n-1)/2*lb occurs at x_{i} = lb (i.e., lower bound of the discrete variables)
    max with replacement = n*(n-1)/2*ub occurs at x_{i} = ub (i.e., upper bound of the discrete variables)
    min w/o replacement  = $\Sigma_{i=0}^{n-1} (lb+i)(i+1)$ occurs at x_{i} = lb+i
    max w/o replacement  = $\Sigma_{i=0}^{n-1} (ub-n+1+i)(i+1)$ occurs at x_{i} = ub-n+1+i

    @ In, inputs, dictionary of variables
    @ Out, sum, value at inputs
  """
  summ = 0
  for ind,var in enumerate(inputs.keys()):
    summ += (ind+1) * inputs[var]
  return summ[:]

def run(self,Inputs):
  """
    Function to calculate the average of the sampled six variables. This is used to check distribution for large number of samples.
    @ In, Input, ParameterInput, RAVEN sampled params.
    @ Out, None
  """
  self.ans = evaluate(Inputs)
