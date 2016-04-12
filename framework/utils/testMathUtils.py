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
import mathUtils
frameworkDir = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(frameworkDir)

print (mathUtils)

results = {"pass":0,"fail":0}

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
    if updateResults: results["fail"] += 1
    return False
  else:
    if updateResults: results["pass"] += 1
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
    if updateResults: results["fail"] += 1
    return False
  else:
    if updateResults: results["pass"] += 1
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
checkAnswer('simpson',simp,0.490666667,1e-5)

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

print(results)

sys.exit(results["fail"])
