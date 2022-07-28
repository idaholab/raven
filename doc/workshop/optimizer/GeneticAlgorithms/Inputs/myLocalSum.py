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

# @author: Mohammad Abdo (@Jimmy-INL)

def evaluate(Inputs):
  """
    Evaluates a weighted sum function.
    $summ = \Sigma_{i=0} (i+1)*x_{i}$

    min with replacement = n*(n-1)/2*lb occurs at x_{i} = lb (i.e., lower bound of the discrete variables)
    max with replacement = n*(n-1)/2*ub occurs at x_{i} = ub (i.e., upper bound of the discrete variables)
    min w/o replacement  = $\Sigma_{i=0}^{n-1} (lb+i)(i+1)$ occurs at x_{i} = lb+i
    max w/o replacement  = $\Sigma_{i=0}^{n-1} (ub-n+1+i)(i+1)$ occurs at x_{i} = ub-n+1+i

    @ In, Inputs, dictionary, dictionary of inputs passed to the external model
    @ Out, Sum, float, objective function
  """
  Sum = 0
  for ind,var in enumerate(Inputs.keys()):
    # write the objective function here
    Sum += (ind + 1) * Inputs[var]
  return Sum[:]

def run(self,Inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, Inputs, dict, additional inputs
    @ Out, None
  """
  self.obj = evaluate(Inputs) # make sure the name of the objective is consistent obj
