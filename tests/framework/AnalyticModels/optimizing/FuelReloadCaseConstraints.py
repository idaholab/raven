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
def implicitConstraint(Input):
  """
    Evaluates the constraint function @ a given point ($\vec(x)$)
    @ In, self, object, RAVEN container
    @ Out, g(x1,x2), float, $g(\vec(x)) = x1 + x2 - 9$
    because the original constraint was x1 + x2 > 9
            the way the constraint is designed is that
            the constraint function has to be >= 0,
            so if:
            1) f(x,y) >= 0 then g = f
            2) f(x,y) >= a then g = f - a
            3) f(x,y) <= b then g = b - f
            4) f(x,y)  = c then g = 0.001 - (f(x,y) - c)

    Now for this constrinat let's consider 6 Genes (x1..x6), and 5 Fuel IDs (0..4)
    But let's assume that fuel 1 is needed to be used twice, where as fuels 1+4 need to be used 6 times
  """
  g = eval(Input.name)(Input)
  return g



def impConstr1(Input):
  # fuelID 1 should be repeated twice
  fuelID = 1
  required = 2
  chrom = list([Input.x1,Input.x2,Input.x3,Input.x4,Input.x5,Input.x6])
  g1 = chrom.count(fuelID) - required
  g2 = required - chrom.count(fuelID)
  return min(g1,g2)

def impConstr2(Input):
  # sum of times fuelID 1 and 4 are used should be 6
  fuelID = np.atleast_1d([1,4])
  required = 6
  chrom = list([Input.x1,Input.x2,Input.x3,Input.x4,Input.x5,Input.x6])
  # sum = 6 (this is broken to two constraints)
  # g1: not > 6 (if < 6 g1 is negative and hence violated)
  # g2: not < 6 (if >6 g2 is negative and hence violated)
  # if = 6 then g1, and g2 both are zeros and the function returns min(0,0)
  # which is zero which means no penalty (no violation)
  g1 = sum([chrom.count(fuel) for fuel in fuelID]) - required
  g2 = required - sum([chrom.count(fuel) for fuel in fuelID])
  return min(g1,g2)