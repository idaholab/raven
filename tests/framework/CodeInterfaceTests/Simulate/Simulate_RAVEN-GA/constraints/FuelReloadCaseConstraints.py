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
  # fuelID type 0 repeated 11 times
  fuelID = 0
  required = 11
  chrom = list([Input.loc1,Input.loc2,Input.loc3,Input.loc4,Input.loc5,Input.loc6,Input.loc7
               ,Input.loc8,Input.loc9,Input.loc10,Input.loc11,Input.loc12,Input.loc13,Input.loc14
               ,Input.loc15,Input.loc16,Input.loc17,Input.loc18,Input.loc19,Input.loc20,Input.loc22
			   ,Input.loc23,Input.loc24,Input.loc25,Input.loc28,Input.loc29])
  g1 = chrom.count(fuelID) - required
  g2 = required - chrom.count(fuelID)
  return min(g1,g2)

def impConstr2(Input):
  # fuelID type 0 repeated 11 times
  fuelID = 1
  required = 4
  chrom = list([Input.loc1,Input.loc2,Input.loc3,Input.loc4,Input.loc5,Input.loc6,Input.loc7
               ,Input.loc8,Input.loc9,Input.loc10,Input.loc11,Input.loc12,Input.loc13,Input.loc14
               ,Input.loc15,Input.loc16,Input.loc17,Input.loc18,Input.loc19,Input.loc20,Input.loc22
			   ,Input.loc23,Input.loc24,Input.loc25,Input.loc28,Input.loc29])
  g1 = chrom.count(fuelID) - required
  g2 = required - chrom.count(fuelID)
  return min(g1,g2)

def impConstr3(Input):
  # fuelID type 0 repeated 11 times
  fuelID = 2
  required = 3
  chrom = list([Input.loc1,Input.loc2,Input.loc3,Input.loc4,Input.loc5,Input.loc6,Input.loc7
               ,Input.loc8,Input.loc9,Input.loc10,Input.loc11,Input.loc12,Input.loc13,Input.loc14
               ,Input.loc15,Input.loc16,Input.loc17,Input.loc18,Input.loc19,Input.loc20,Input.loc22
			   ,Input.loc23,Input.loc24,Input.loc25,Input.loc28,Input.loc29])
  g1 = chrom.count(fuelID) - required
  g2 = required - chrom.count(fuelID)
  return min(g1,g2)
  
def impConstr4(Input):
  # fuelID type 0 repeated 11 times
  fuelID = 3
  required = 5
  chrom = list([Input.loc1,Input.loc2,Input.loc3,Input.loc4,Input.loc5,Input.loc6,Input.loc7
               ,Input.loc8,Input.loc9,Input.loc10,Input.loc11,Input.loc12,Input.loc13,Input.loc14
               ,Input.loc15,Input.loc16,Input.loc17,Input.loc18,Input.loc19,Input.loc20,Input.loc22
			   ,Input.loc23,Input.loc24,Input.loc25,Input.loc28,Input.loc29])
  g1 = chrom.count(fuelID) - required
  g2 = required - chrom.count(fuelID)
  return min(g1,g2)
  
def impConstr5(Input):
  # fuelID type 0 repeated 11 times
  fuelID = 4
  required = 3
  chrom = list([Input.loc1,Input.loc2,Input.loc3,Input.loc4,Input.loc5,Input.loc6,Input.loc7
               ,Input.loc8,Input.loc9,Input.loc10,Input.loc11,Input.loc12,Input.loc13,Input.loc14
               ,Input.loc15,Input.loc16,Input.loc17,Input.loc18,Input.loc19,Input.loc20,Input.loc22
			   ,Input.loc23,Input.loc24,Input.loc25,Input.loc28,Input.loc29])
  g1 = chrom.count(fuelID) - required
  g2 = required - chrom.count(fuelID)
  return min(g1,g2)
  
def impConstr2old(Input):
  # sum of times fuelID 1 and 3 are used should be 7
  fuelID = np.atleast_1d([1,3])
  required = 7
  chrom = list([Input.loc1,Input.loc2,Input.loc3,Input.loc4,Input.loc5,Input.loc6,Input.loc7
               ,Input.loc8,Input.loc9,Input.loc10,Input.loc11,Input.loc12,Input.loc13,Input.loc14
               ,Input.loc15,Input.loc16,Input.loc17,Input.loc18,Input.loc19,Input.loc20,Input.loc22
			   ,Input.loc23,Input.loc24,Input.loc25,Input.loc28,Input.loc29])
  # sum = required (this is broken to two constraints)
  # g1: not > required (if < required g1 is negative and hence violated)
  # g2: not < required (if > required g2 is negative and hence violated)
  # if = required then g1, and g2 both are zeros and the function returns min(0,0)
  # which is zero which means no penalty (no violation)
  g1 = sum([chrom.count(fuel) for fuel in fuelID]) - required
  g2 = required - sum([chrom.count(fuel) for fuel in fuelID])
  return min(g1,g2)
  
def impConstr3old(Input):
  # sum of times fuelID 2 and 4 are used should be 8
  fuelID = np.atleast_1d([2,4])
  required = 8
  chrom = list([Input.loc1,Input.loc2,Input.loc3,Input.loc4,Input.loc5,Input.loc6,Input.loc7
               ,Input.loc8,Input.loc9,Input.loc10,Input.loc11,Input.loc12,Input.loc13,Input.loc14
               ,Input.loc15,Input.loc16,Input.loc17,Input.loc18,Input.loc19,Input.loc20,Input.loc22
			   ,Input.loc23,Input.loc24,Input.loc25,Input.loc28,Input.loc29])
  # sum = required (this is broken to two constraints)
  # g1: not > required (if < required g1 is negative and hence violated)
  # g2: not < required (if > required g2 is negative and hence violated)
  # if = required then g1, and g2 both are zeros and the function returns min(0,0)
  # which is zero which means no penalty (no violation)
  g1 = sum([chrom.count(fuel) for fuel in fuelID]) - required
  g2 = required - sum([chrom.count(fuel) for fuel in fuelID])
  return min(g1,g2)