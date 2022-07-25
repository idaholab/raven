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

def implicitConstraint(Input):
  """
    Evaluates the constraint function @ a given point ($\vec(x)$)
    @ In, self, object, RAVEN container
    @ Out, g(x1,x2,..), float, $g(\vec(x)) = f(\vec(x)) - a$
    because the original constraint was f(\vec(x)) > a
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 0.001 - (f(x,y) - c)
  """
  g = 250 - Input.ymax
  return g

def constrain(self,Input):
  """
    Constrain calls the constraint function.
    @ In, self, object, RAVEN container
    @ Out, explicitConstrain, float, positive if the constraint is satisfied
           and negative if violated.
  """
  implicitConstraint = implicitConstraint(Input)
  return implicitConstraint