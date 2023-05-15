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
# @ author: Mohammad Abdo (@Jimmy-INL)

def constrain(Input):
  """
    This function calls the explicit constraint whose name is passed through Input.name
    the evaluation function g is negative if the explicit constraint is violated and positive otherwise.
    This suits the constraint handling in the Genetic Algorithms,
    but not the Gradient Descent as the latter expects True if the solution passes the constraint and False if it violates it.
    @ In, Input, object, RAVEN container
    @ Out, g, float, explicit constraint evaluation (negative if violated and positive otherwise)
  """
  g = eval(Input.name)(Input)
  return g

def implicitConstraint(Input):
  """
    Evaluates the implicit constraint function at a given point/solution ($\vec(x)$)
    @ In, Input, object, RAVEN container
    @ Out, g(inputs x1,x2,..,output or dependent variable), float, implicit constraint evaluation function
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 1e-6 - abs((f(x,y) - c)) (equality constraint)
  """
  g = eval(Input.name)(Input)
  return g

def expConstr1(Input):
  """
    first explicit constraint called by the constrain function.
    @ Out, g, float, explicit constraint 1 evaluation function (positive if the constraint is satisfied
           and negative if violated).
  """
  # explicit constraint: solution is in a standard circle (centered at the origin of the x1-x3 plan) whose raduis is 5 units
  g = 25 - (Input.x1**2 + Input.x3)**2
  return g

def impConstr1(Input):
  """
    first implicit constraint called by the implicitConstraint function:
    Evaluates the implicit constraint function @ a given point ($\vec(x)$)
    @ In, self, object, RAVEN container
    @ Out, g(inputs,outputs), float, $g(\vec(x,y)) = f(\vec(x,y)) - a$
    because the original constraint was f(\vec(x,y)) > a
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 1e-12 - abs(f(x,y) - c)
  """
  g = 250 - Input.ymax
  return g

