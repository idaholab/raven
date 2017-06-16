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
  This Module performs Unit Tests for the randomUtils methods
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os,sys
import numpy as np
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import utils
utils.find_crow(frameworkDir)
from utils import randomUtils

print (randomUtils)

results = {"pass":0,"fail":0}

def checkTrue(comment,value,expected):
  """
    Takes a boolean and checks it against True or False.
  """
  if value == expected:
    results["pass"] += 1
    return True
  else:
    print("checking answer",comment,value,"!=",expected)
    results["fail"] += 1
    return False

def checkAnswer(comment,value,expected,tol=1e-10,updateResults=True):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, updateResults, bool, optional, if True updates global results
    @ Out, None
  """
  if abs(value - expected) > tol:
    print("checking answer",comment,value,"!=",expected)
    if updateResults:
      results["fail"] += 1
    return False
  else:
    if updateResults:
      results["pass"] += 1
    return True

def checkArray(comment,check,expected,tol=1e-10):
  """
    This method is aimed to compare two arrays of floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, check, list, the value to compare
    @ In, expected, list, the expected value
    @ In, tol, float, optional, the tolerance
    @ Out, None
  """
  same=True
  if len(check) != len(expected):
    same=False
  else:
    for i in range(len(check)):
      same = same*checkAnswer(comment+'[%i]'%i,check[i],expected[i],tol,False)
  if not same:
    print("checking array",comment,"did not match!")
    results['fail']+=1
    return False
  else:
    results['pass']+=1
    return True

def checkType(comment,value,expected,updateResults=True):
  """
    This method compares the data type of two values
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, updateResults, bool, optional, if True updates global results
    @ Out, None
  """
  if type(value) != type(expected):
    print("checking type",comment,value,'|',type(value),"!=",expected,'|',type(expected))
    if updateResults:
      results["fail"] += 1
    return False
  else:
    if updateResults:
      results["pass"] += 1
    return True

### BEGIN TESTS
# NOTE that due to seeding, this test relies HEAVILY on not changing the orders of calls to randomUtils!
# Reseed at the beginning of sections and add new tests to the end of sections.

# randomSeed(), setting the random seed
randomUtils.randomSeed(42)
# check that seed is set
checkAnswer('First float from set seed',randomUtils.random(),0.374540114397)
# check resetting seed
randomUtils.randomSeed(12345) #next float would be 0.796542984386 if seed didn't change
checkAnswer('First float from set seed',randomUtils.random(),0.929616086867)

### random(), sampling on [0,1]
## single sampling
randomUtils.randomSeed(42)
vals = np.array([randomUtils.random() for _ in range(100)])
mean = np.average(vals)
stdv = np.std(vals)
checkAnswer('mean of 100 single samples',mean,0.44717121008)
checkAnswer('stdv of 100 single samples',stdv,0.294751019373)
## 1d batch sampling
randomUtils.randomSeed(42)
vals = randomUtils.random(100)
mean = np.average(vals)
stdv = np.std(vals)
checkAnswer('mean of 100 batch samples',mean,0.44717121008)
checkAnswer('stdv of 100 batch samples',stdv,0.294751019373)
## 2d batch sampling
randomUtils.randomSeed(42)
vals = randomUtils.random(10,100)
# check statistics
mean = np.average(vals)
stdv = np.std(vals)
checkAnswer('mean of 10x100 batch samples',mean,0.499200436821)
checkAnswer('stdv of 10x100 batch samples',stdv,0.291958707014)
# check single entry
right = [0.374540114397, 0.796542984386, 0.950714311784, 0.183434787715, 0.731993938501,
         0.779690997624, 0.598658486409, 0.59685016158 , 0.156018638554, 0.445832757616]
checkArray('10x100 batch first entry',vals[0],right)

### randomIntegers(), sampling integers in a range
randomUtils.randomSeed(42)
right = [14,18,20,12,17]
for i in range(5):
  n = randomUtils.randomIntegers(10,20,None) #no message handler, error handling will error out
  checkAnswer('random integer, {} sample'.format(i),n,right[i])

### randomPermutation(), rearranging lists
randomUtils.randomSeed(42)
l = [1,2,3,4,5]
l2 = randomUtils.randomPermutation(l,None)
checkArray('random permutation',l2,[2,4,5,1,3])

### randPointsOnHypersphere(), unit hypersphere surface sampling (aka random direction)

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.randomUtils</name>
    <author>talbpaul</author>
    <created>2017-06-16</created>
    <classesTested>utils.randomUtils</classesTested>
    <description>
       This test performs Unit Tests for the randomUtils methods
    </description>
  </TestInfo>
"""
