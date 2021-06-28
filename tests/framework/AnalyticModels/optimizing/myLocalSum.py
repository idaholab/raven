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
    $summ = \Sigma_{i=0} (i+1)*x_{i}$

    min with replacement = n*(n-1)/2*lb occurs at x_{i} = lb (i.e., lower bound of the discrete variables)
    max with replacement = n*(n-1)/2*ub occurs at x_{i} = ub (i.e., upper bound of the discrete variables)
    min w/o replacement  = $\Sigma_{i=0}^{n-1} (lb+i)(i+1)$ occurs at x_{i} = lb+i
    max w/o replacement  = $\Sigma_{i=0}^{n-1} (ub-n+1+i)(i+1)$ occurs at x_{i} = ub-n+1+i

    @ In, inputs, dictionary of variables
    @ Out, summ, value at inputs
  """
  summ = 0
  for ind,var in enumerate(inputs.keys()):
    summ += (ind+1) * inputs[var]
  return summ[:]

def constraint(self):
  """
    Evaluates the constraint function @ a given point ($\vec(x)$)
    @ In, self, object, RAVEN container
    @ Out, g(x1,x2), float, $g(\vec(x)) = x1 + x2 - 6$
    because the original constraint was x1 + x2 > 6
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 0.001 - (f(x,y) - c)
  """
  g = self.x1 + self.x2 - 6
  return g

def impConstraint(self):
  """
    Evaluates the constraint function @ a given point ($\vec(x)$)
    @ In, self, object, RAVEN container
    @ Out, g(x1,x2), float, $g(\vec(x)) = x1 + x2 - 6$
    because the original constraint was x1 + x2 > 6
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 0.001 - (f(x,y) - c)
  """
  g = self.x1 + self.x2 + self.ans- 9
  return g

def run(self,Inputs):
  """
    RAVEN API
    @ In, self, object, RAVEN container
    @ In, Inputs, dict, additional inputs
    @ Out, None
  """
  self.ans = evaluate(Inputs)

def constrain(self):
  """
    Constrain calls the constraint function.
    @ In, self, object, RAVEN container
    @ Out, explicitConstrain, float, positive if the constraint is satisfied
           and negative if violated.
  """
  explicitConstrain = constraint(self)
  return explicitConstrain

def impConstrain(self):
  """
    Constrain calls the constraint function.
    @ In, self, object, RAVEN container
    @ Out, explicitConstrain, float, positive if the constraint is satisfied
           and negative if violated.
  """
  implicitConstrain = impConstraint(self)
  return implicitConstrain