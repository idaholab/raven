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
  This Module performs Unit Tests for the mathUtils methods
  It cannot be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)

import os,sys
import numpy as np
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import mathUtils
import numpy as np

print (mathUtils)

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
    print("checking answer",comment,':',value,"!=",expected)
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


### check "normal" pdf, cdf
# tests is (x,mu,sigma,normal,cdf)
tests = [ (1.0, 0.0, 1.0, 0.241971 , 0.841345),
          (0.0, 0.0, 1.0, 0.398942 , 0.5     ),
          (11., 10., 1.0, 0.241971 , 0.841345),
          (1.0, 0.0, 4.0, 0.096667 , 0.598706),
          (5.0,-2.0, 0.1, 0.0      , 1.0     )]
# test default
x,mu,sigma,normal,cdf = tests[0]
testNormal = mathUtils.normal(x)
testCdf = mathUtils.normalCdf(x)
checkAnswer('default normal (%f)' %(x),testNormal,normal,1e-5)
checkAnswer('default cdf    (%f)' %(x),testCdf   ,cdf   ,1e-5)
# test full values
for test in tests:
  x,mu,sigma,normal,cdf = test
  testNormal = mathUtils.normal(x,mu,sigma)
  testCdf = mathUtils.normalCdf(x,mu,sigma)
  checkAnswer('normal (%f,%f,%f)' %(x,mu,sigma),testNormal,normal,1e-5)
  checkAnswer('cdf    (%f,%f,%f)' %(x,mu,sigma),testCdf   ,cdf   ,1e-5)

### check "skewNormal"
# tests is (position, shape, location, scale, value)
tests = [(0, 4, 0, 1, 0.398942),
         (0, 0, 0, 1, 0.398942),
         (2, 4, 1, 2, 0.344056),
         (1, 2, 3, 4, 0.027929)]
for test in tests:
  x,alpha,xi,omega,val = test
  testSkewNormal = mathUtils.skewNormal(x,alpha,xi,omega)
  checkAnswer('skewNormal (%f,%f,%f,%f)' %(x,alpha,xi,omega),testSkewNormal,val,1e-5)

### check "createInterp"
# TODO

### check "simpson"
def f(x):
  """
    Simple squaring function.
    @ In, x, float, value
    @ Out, f, float, square value
  """
  return x*x

simp = mathUtils.simpson(f,-1,1,5)
checkAnswer('simpson',simp,0.677333333333,1e-5)

### check "getGraphs"
# TODO I don't know what this does.  Documentation is poor.

### check "countBins"
data = [0.1,0.2,
        1.1,1.2,1.3,
        2.1,2.2,2.3,2.4,
        3.1,3.2,3.3]
boundaries = [1,2,3]
counted = mathUtils.countBins(data,boundaries)
checkArray('countBins',counted,[2,3,4,3],1e-5)

### check "log2"
data = [(1e-15,-49.82892),
          (0.5,-1.0),
          (1.0,0.0),
            (4,2.0),
           (10,3.32193),
         (1e34,112.945556)]
for d in data:
  dat,soln = d
  val = mathUtils.log2(dat)
  checkAnswer('log2',val,soln,1e-5)

### check "calculateStats"
data = [0.6752,0.0610,0.1172,0.5233,0.0056]
moms = mathUtils.calculateStats(data)
checkAnswer('calculateStats.mean'          ,moms['mean'          ], 0.27646 ,1e-5)
checkAnswer('calculateStats.stdev'         ,moms['stdev'         ], 0.30211 ,1e-5)
checkAnswer('calculateStats.variance'      ,moms['variance'      ], 0.073015,1e-5)
checkAnswer('calculateStats.skewness'      ,moms['skewness'      ], 0.45134 ,1e-5)
checkAnswer('calculateStats.kurtosis'      ,moms['kurtosis'      ],-1.60548 ,1e-5)
checkAnswer('calculateStats.sampleVariance',moms['sampleVariance'], 0.09127 ,1e-5)

### check "historySetWindows"
# TODO I think this takes a historySet?  Documentation is poor.

### check "convertNumpyToLists"
datDict = {'a':np.array([1,2,3,4,5]),
           'b':np.array([2,3,4,5]),
           'c':np.array([3,4,5])}
datMat = np.array([
           np.array([1,2,3]),
           np.array([4,5,6]),
           np.array([7,8,9]),
           ])
datAry = np.array([1,2,3,4])
convDict = mathUtils.convertNumpyToLists(datDict)
convMat  = mathUtils.convertNumpyToLists(datMat)
convAry  = mathUtils.convertNumpyToLists(datAry)
for v,(key,value) in enumerate(convDict.items()):
  checkType('convertNumpyToList.Dict[%i].type)' %v,value,[])
  checkArray('convertNumpyToList.Dict[%i].values)' %v,value,datDict[key])
checkType('convertNumpyToList.Matrix.type)',convMat,[])
for e,entry in enumerate(convMat):
  checkType('convertNumpyToList.Matrix[%i].type)' %e,entry,[])
  checkArray('convertNumpyToList.Dict[%i].values)' %e,entry,datMat[e])
checkType('convertNumpyToList.Array.type)',convAry,[])
checkArray('convertNumpyToList.Array.values)',convAry,datAry)


### check "interpolateFunction"
# TODO some documentation missing

### check "distance"
points = np.array([(1,1,1),(2,2,2),(3,3,3)])
find = [(0,0,0),(1,2,1),(1.1,2.1,2.1),(2,2,2),(10,10,10)]
dists = [(1.73205 ,3.46410 ,5.19615),
         (1.0     ,1.41421 ,3.0     ),
         (1.55885 ,0.91104 ,2.28692 ),
         (1.73205 ,0.0     ,1.73205 ),
         (15.58846,13.85641,12.12436)]

for i,f in enumerate(find):
  dist = mathUtils.distance(points,f)
  checkArray('distance %s' %str(f),dist,dists[i],1e-5)


### check "numpyNearestMatch"
findIn = np.array([(1,1,1),(2,2,2),(3,3,3)])
find =    [(0,0,0),(1,2,1),(1,2,2),(2,2,2),(10,10,10)]
idcs    = [   0   ,   0   ,   1   ,   1   ,    2     ]
correct = [(1,1,1),(1,1,1),(2,2,2),(2,2,2),( 3, 3, 3)]
for i,f in enumerate(find):
  idx,ary = mathUtils.numpyNearestMatch(findIn,f)
  checkAnswer('numpyNearersMatch %s' %str(f),idx,idcs[i],1e-5)
  checkArray('numpyNearersMatch %s' %str(f),ary,correct[i],1e-5)


### check relative differences
# similar order magnitude
checkAnswer('relativeDiff O(1)',mathUtils.relativeDiff(1.234,1.233),0.00081103000811)
# large order magnitude
checkAnswer('relativeDiff O(1e10)',mathUtils.relativeDiff(1.234e10,1.233e10),0.00081103000811)
# small order magnitude
checkAnswer('relativeDiff O(1e-10)',mathUtils.relativeDiff(1.234e-10,1.233e-10),0.00081103000811)
# different magnitudes
checkAnswer('relativeDiff different magnitudes',mathUtils.relativeDiff(1.234e10,1.233e-10),1.00081103000811e20,tol=1e6)
# measured is 0
checkAnswer('relativeDiff first is zero',mathUtils.relativeDiff(0,1.234),1.0)
# expected is 0
checkAnswer('relativeDiff second is zero',mathUtils.relativeDiff(1.234,0),1.0)
# both are 0
checkAnswer('relativeDiff both are zero',mathUtils.relativeDiff(0,0),0.0)
# first is inf
checkAnswer('relativeDiff first is inf',mathUtils.relativeDiff(np.inf,0),np.inf)
# second is inf
checkAnswer('relativeDiff second is inf',mathUtils.relativeDiff(0,np.inf),np.inf)
# both are inf
checkAnswer('relativeDiff both are inf',mathUtils.relativeDiff(np.inf,np.inf),0)

### check float comparison
#moderate order of magnitude
checkTrue('compareFloats moderate OoM match',mathUtils.compareFloats(3.141592,3.141593,tol=1e-6),True)
checkTrue('compareFloats moderate OoM mismatch',mathUtils.compareFloats(3.141592,3.141593,tol=1e-8),False)
#small order of magnitude
checkTrue('compareFloats small OoM match',mathUtils.compareFloats(3.141592e-15,3.141593e-15,tol=1e-6),True)
checkTrue('compareFloats small OoM mismatch',mathUtils.compareFloats(3.141592e-15,3.141593e-15,tol=1e-8),False)
#small order of magnitude
checkTrue('compareFloats large OoM match',mathUtils.compareFloats(3.141592e15,3.141593e15,tol=1e-6),True)
checkTrue('compareFloats large OoM mismatch',mathUtils.compareFloats(3.141592e15,3.141593e15,tol=1e-8),False)


### check "NDinArray"
points = np.array([(0.61259532,0.27325707,0.81182424),
                   (0.54608679,0.82470626,0.39170769)])
findSmall = (0.55,0.82,0.39)
findLarge = (0.61259532123,0.27325707123,0.81182423999)
found,idx,entry = mathUtils.NDInArray(points,findSmall,tol=1e-2)
checkAnswer('NDInArray %s found' %str(findSmall),int(found),1)
checkAnswer('NDInArray %s idx' %str(findSmall),idx,1)
checkArray('NDInArray %s entry' %str(findSmall),entry,points[1])
found,idx,entry = mathUtils.NDInArray(points,findSmall,tol=1e-3)
checkAnswer('NDInArray %s not found' %str(findSmall),int(found),0)
checkType('NDInArray %s no idx' %str(findSmall),idx,None)
checkType('NDInArray %s no entry' %str(findSmall),entry,None)
found,idx,entry = mathUtils.NDInArray(points,findLarge,tol=1e-8)
checkAnswer('NDInArray %s found' %str(findLarge),int(found),1)
checkAnswer('NDInArray %s idx' %str(findLarge),idx,0)
checkArray('NDInArray %s entry' %str(findLarge),entry,points[0])

### check "normalizationFactors"
zeroList       = [0,0,0,0,0]
fourList       = [4,4,4,4,4]
sequentialList = [0,1,2,3,4]

factors = mathUtils.normalizationFactors(zeroList, mode='z')
checkArray('Z-score normalization zeroList: ', factors, (0,1))
factors = mathUtils.normalizationFactors(zeroList, mode='scale')
checkArray('0-1 scaling zeroList: ', factors, (0,1))
factors = mathUtils.normalizationFactors(zeroList, mode='none')
checkArray('No scaling zeroList: ', factors, (0,1))

factors = mathUtils.normalizationFactors(fourList, mode='z')
checkArray('Z-score normalization fourList: ', factors, (4,4))
factors = mathUtils.normalizationFactors(fourList, mode='scale')
checkArray('0-1 scaling fourList: ', factors, (4,4))
factors = mathUtils.normalizationFactors(fourList, mode='none')
checkArray('No scaling fourList: ', factors, (0,1))

factors = mathUtils.normalizationFactors(sequentialList, mode='z')
checkArray('Z-score normalization sequentialList: ', factors, (2,1.41421356237))
factors = mathUtils.normalizationFactors(sequentialList, mode='scale')
checkArray('0-1 scaling sequentialList: ', factors, (0,4))
factors = mathUtils.normalizationFactors(sequentialList, mode='none')
checkArray('No scaling sequentialList: ', factors,(0,1))

#check hyperrectangle diagonal on several dimensions
## 2d
sideLengths = [3,4]
checkAnswer('2D hyperdiagonal',mathUtils.hyperdiagonal(sideLengths),5)
## 3d
sideLengths.append(12)
checkAnswer('3D hyperdiagonal',mathUtils.hyperdiagonal(sideLengths),13)
## 3d
sideLengths.append(84)
checkAnswer('4D hyperdiagonal',mathUtils.hyperdiagonal(sideLengths),85)

# check diffWithInfinites
i = float('inf')
n = np.inf
checkAnswer('InfDiff inf    - inf'   ,mathUtils.diffWithInfinites( n, i), 0)
checkAnswer('InfDiff inf    - finite',mathUtils.diffWithInfinites( n, 0), i)
checkAnswer('InfDiff inf    - (-inf)',mathUtils.diffWithInfinites( n,-n), i)
checkAnswer('InfDiff finite - inf'   ,mathUtils.diffWithInfinites( 0, n),-i)
checkAnswer('InfDiff finite - finite',mathUtils.diffWithInfinites( 3, 2), 1)
checkAnswer('InfDiff finite - (-inf)',mathUtils.diffWithInfinites( 0,-n), i)
checkAnswer('InfDiff -inf   - inf'   ,mathUtils.diffWithInfinites(-n, n),-i)
checkAnswer('InfDiff -inf   - finite',mathUtils.diffWithInfinites(-n, 0),-i)
checkAnswer('InfDiff -inf   - (-inf)',mathUtils.diffWithInfinites(-n,-n), 0)

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.mathUtils</name>
    <author>talbpaul</author>
    <created>2016-11-01</created>
    <classesTested>utils.mathUtils</classesTested>
    <description>
       This test performs Unit Tests for the mathUtils methods
       It cannot be considered part of the active code but of the regression test system
    </description>
    <revisions>
      <revision author="talbpaul" date="2016-11-08">Relocated utils tests</revision>
      <revision author="alfoa" date="2017-01-21">Adding this test description.</revision>
      <revision author="alfoa" date="2019-03-04">Moved methods isAString, isAFloat, isAInteger, isABoolean from mathUtils to utils</revision>
    </revisions>
  </TestInfo>
"""
