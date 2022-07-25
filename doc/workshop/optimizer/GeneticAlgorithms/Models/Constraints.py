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

import numpy as np

def XX(Input):#Complete this: give the function the correct name#
  g = eval(Input.name)(Input)
  return g

def XX(Input):#Complete this: give the function the correct name#
  """
    Evaluates the constraint function @ a given point ($\vec(x)$)
    @ In, self, object, RAVEN container
    @ Out, g(x1,x2,obj), float, constraint evaluation function
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


def XX(Input):#You are free to pick this name but it has to be similar to the one in the xml#
  """
    Let's assume that the constraint is:
    $ x3+x4 < 8 $
    then g the constraint evaluation function (which has to be > 0) is taken to be:
    g = 8 - (x3+x4)
    in this case if g(\vec(x)) < 0 then this x violates the constraint and vise versa
  """
  return # Write the constraint gere

def expConstr2(Input):
  """
    What about equality constraints?
    let's consider the constraint x1**2 + x2**2 = 25
    The way to write g is to use a very small number for imnstance, epsilon = 1e-12
    and then g = epsilon - abs(constraint)
    See below
  """
  return 1e-12 - abs(Input.x1**2 + Input.x2**2 - 25)

def impConstr1(Input):
  """
    The implicit constraint involves variables from the output space, for example the objective variable or
    a dependent variable that is not in the optimization search space
  """
  return 10 - Input.x1**2 - Input.obj

def impConstr2(Input):
  return Input.x1**2 + Input.obj - 10