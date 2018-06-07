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
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import utils
utils.find_crow(frameworkDir)
from utils import randomUtils
randomENG = utils.findCrowModule("randomENG")

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

def checkAnswer(comment,value,expected,tol=1e-7,updateResults=True):
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

def checkArray(comment,check,expected,tol=1e-7):
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

# set the stochastic environment TODO check both someday?
randomUtils.stochasticEnv = 'crow'

# randomSeed(), setting the random seed
randomUtils.randomSeed(42)
# check that seed is set
checkAnswer('First float from first seed',randomUtils.random(),0.374540118847)
# check resetting seed
randomUtils.randomSeed(12345) #next float would be 0.95071430641 if seed didn't change
checkAnswer('First float from second seed',randomUtils.random(),0.929616092817)

### random(), sampling on [0,1]
## single sampling
randomUtils.randomSeed(42)
vals = np.array([randomUtils.random() for _ in range(int(1e5))])
mean = np.average(vals)
stdv = np.std(vals)
checkAnswer('mean of 1e5 single samples',mean,0.5, tol=1e-3)
checkAnswer('stdv of 1e5 single samples',stdv,np.sqrt(1./12.), tol=1e-3)
## 1d batch sampling
randomUtils.randomSeed(42)
vals = randomUtils.random(1e5)
mean = np.average(vals)
stdv = np.std(vals)
checkAnswer('mean of 1e5 batch samples',mean,0.5, tol=1e-3)
checkAnswer('stdv of 1e5 batch samples',stdv,np.sqrt(1./12.), tol=1e-3)
## 2d batch sampling
randomUtils.randomSeed(42)
vals = randomUtils.random(10,1000)
# check statistics
mean = np.average(vals)
stdv = np.std(vals)
checkAnswer('mean of 10x100 batch samples',mean,0.5,tol=1e-3)
checkAnswer('stdv of 10x100 batch samples',stdv,np.sqrt(1./12.),tol=1e-3)

### randomNormal
## first test box muller
mean,stdev = randomUtils.BoxMullerGenerator().testSampling(1e5)
checkAnswer('Box Muller generator mean',mean,0.0,tol=5e-3)
checkAnswer('Box Muller generator stdv',stdev,1.0,tol=1e-3)
## test single value
vals = randomUtils.randomNormal()
checkAnswer('random normal single value',vals,1.90167449657)
## test single point
right = [1.11130480322, 0.698326166056, 2.82788725018]
vals = randomUtils.randomNormal(3)
checkArray('random normal single point',vals,right)
## test many points
right = [0,0,0]
vals = randomUtils.randomNormal(3,5)
checkAnswer('randomNormal number of samples',len(vals),5)
checkAnswer('randomNormal size of sample',len(vals[0]),3)

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
randomUtils.randomSeed(42)
## check the radius is always 1 (if not specified)
for i in range(1,6):
  pt = randomUtils.randPointsOnHypersphere(i)
  checkAnswer('Random {}D hypersphere surface'.format(i),np.sum(pt*pt),1.0)
## check the sum of the squares is always the square of the radius
randomUtils.randomSeed(42)
for i in [0.2,0.7,1.5,10.0, 100.0]:
  pt = randomUtils.randPointsOnHypersphere(4,r=i)
  checkAnswer('Random 4D hypersphere surface with {} radius'.format(i),np.sum(pt*pt),i*i)
## check multiple sampling simultaneously
randomUtils.randomSeed(42)
samps = randomUtils.randPointsOnHypersphere(5,samples=100)
checkAnswer('Simultaneous random 5D on hypersphere, 0 axis',samps.shape[0],100)
checkAnswer('Simultaneous random 5D on hypersphere, 1 axis',samps.shape[1],5)
for i,s in enumerate(samps):
  checkAnswer('Entry {}, simultaneous random 5D hypersphere'.format(i),np.sum(s*s),1.0)
## visual check; skipped generally but left for debugging
if False:
  import matplotlib.pyplot as plt
  samps = randomUtils.randPointsOnHypersphere(2,samples=1e4)
  x = samps[:,0]
  y = samps[:,1]
  plt.plot(x,y,'.')
  plt.show()

### randPointsInHypersphere(), random point in hypersphere
randomUtils.randomSeed(42)
## check the radius is always 1 or less (if not specified)
for i in range(1,6):
  pt = randomUtils.randPointsInHypersphere(i)
  checkTrue('Random {}D hypersphere interior'.format(i),np.sum(pt*pt)<=1.0,True)
## check the sum of the squares is always the square of the radius
for i in [0.2,0.7,1.5,10.0, 100.0]:
  pt = randomUtils.randPointsInHypersphere(4,r=i)
  checkTrue('Random 4D hypersphere surface with {} radius'.format(i),np.sum(pt*pt)<=i*i,True)
## check multiple sampling simultaneously
samps = randomUtils.randPointsInHypersphere(5,samples=100)
checkAnswer('Simultaneous random 5D in hypersphere, 0 axis',samps.shape[0],100)
checkAnswer('Simultaneous random 5D in hypersphere, 1 axis',samps.shape[1],5)
for i in range(samps.shape[1]):
  s = samps[i]
  checkTrue('Entry {}, simultaneous random 5D hypersphere'.format(i),np.sum(s*s)<=1.0,True)
## check if it is possible to instanciate multiple random number generators (isolated)
## this is more a test for the crow_modules.randomENGpy[2,3]
firstRNG = randomENG.RandomClass()
secondRNG = randomENG.RandomClass()
# seed with different seeds
firstRNG.seed(200286)
secondRNG.seed(20021986)
checkTrue('Check if two instances of RNG (different seed) are different',
          firstRNG.random() != secondRNG.random(),True)
# seed with same seeds
firstRNG.seed(200286)
secondRNG.seed(200286)
checkTrue('Check if two instances of RNG (same seed) are identical',
          firstRNG.random() == secondRNG.random(),True)
## visual check; skipped generally but left for debugging
if False:
  import matplotlib.pyplot as plt
  samps = randomUtils.randPointsInHypersphere(2,samples=1e4)
  x = samps[:,0]
  y = samps[:,1]
  plt.plot(x,y,'.')
  plt.show()


# RNG factory
## unseeded (default seeding)
engine = randomUtils.newRNG()
sampled = [engine.random() for _ in range(5)]
correct = [0.814723692093,
           0.135477004139,
           0.905791934325,
           0.835008589978,
           0.126986811898]
checkArray('Independent RNG, unseeded',sampled,correct)

## reseeded (42)
engine.seed(42)
correct = [0.374540114397,
           0.796542984386,
           0.950714311784,
           0.183434787715,
           0.731993938501]
sampled = [engine.random() for _ in range(5)]
checkArray('Independent RNG, reseeded',sampled,correct)

## seeded (42) -> should be same as "reseeded"
engine = randomUtils.newRNG()
engine.seed(42)
sampled = [engine.random() for _ in range(5)]
checkArray('Independent RNG, seeded',sampled,correct)

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
