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
  This Module performs Unit Tests for the frontUtils methods
"""


import os,sys
import numpy as np
ravenDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir))
sys.path.append(ravenDir)
from ravenframework.utils import utils
utils.find_crow(os.path.join(ravenDir,"ravenframework"))
from ravenframework.utils import frontUtils

randomENG = utils.findCrowModule("randomENG")

print (frontUtils)

results = {"pass":0,"fail":0}

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


test3D = np.array([[ 0.21573114, -0.92937786,  0.29952775],
                   [ 0.94716548, -0.31085637, -0.07903087],
                   [ 0.6485263,  -0.72106429,  0.24388507],
                   [ 0.3466882,  -0.78716832,  0.51007189],
                   [ 0.15463182, -0.18730138,  0.97005525],
                   [ 0.02937279, -0.91175393,  0.40968525],
                   [-0.24039731,  0.54889384,  0.80057772],
                   [ 0.06213356,  0.28552822, -0.95635404],
                   [-0.20190017,  0.66695686, -0.71722024],
                   [-0.62399932, -0.22858416,  0.74724436]])

mask = frontUtils.nonDominatedFrontier(test3D, returnMask=True)
answerMask = np.array([True, True, True, False, False, True, False, True, True, True])
checkArray('nonDominatedFrontier with mask', mask.tolist(), answerMask.tolist())

indexes = frontUtils.nonDominatedFrontier(test3D, returnMask=False)
answerIndexes = np.array([0, 1, 2, 5, 7, 8, 9])
checkArray('nonDominatedFrontier with indexes', indexes.tolist(), answerIndexes.tolist())

indexesMinMask = frontUtils.nonDominatedFrontier(test3D, returnMask=False, minMask=np.array([True,True,True]))
answerIndexesMinMask = np.array([0, 1, 2, 5, 7, 8, 9])
checkArray('nonDominatedFrontier MinMask with indexes', indexesMinMask.tolist(), answerIndexesMinMask.tolist())


test2D = np.array([[1.00000000e+00, 5.48813504e-01],
 [9.77077053e-01, 7.14821914e-01],
 [9.61380818e-01, 6.01524934e-01],
 [9.47678668e-01, 5.42365339e-01],
 [9.45824227e-01, 4.20176599e-01],
 [8.96915367e-01, 6.37614902e-01],
 [6.36289229e-01, 4.45417231e-17]])

indexes2D = frontUtils.nonDominatedFrontier(test2D, returnMask=False, minMask=np.array([False,True]))
answerIndexes = np.array([0, 3, 4, 6])
checkArray('2D nonDominatedFrontier MinMask with indexes', indexes2D.tolist(), answerIndexes.tolist())

## Testing crowding distances
# test1: 2 objective functions
testCDarray = np.array([[12, 0],
                       [11.5, 0.5],
                       [11, 1],
                       [10.8, 1.2],
                       [10.5, 1.5],
                       [10.3, 1.8],
                       [9.5, 2],
                       [9, 2.5],
                       [7, 3],
                       [5, 4],
                       [2.5, 6],
                       [2, 10],
                       [1.5, 11],
                       [1, 11.5],
                       [0.8, 11.7],
                       [0, 12]])

rankCDSingleFront = frontUtils.rankNonDominatedFrontiers(testCDarray)
indexesCD2D = frontUtils.crowdingDistance(rank=rankCDSingleFront, popSize=len(rankCDSingleFront), fitness=testCDarray)
answerIndexesCD2D = np.array([np.inf,0.16666667,0.11666667,0.08333333,0.09166667,0.125,0.16666667,0.29166667,0.45833333,0.625,0.75,0.5,0.20833333,0.11666667,0.125,np.inf])
checkArray('2D crowding distance', indexesCD2D.tolist(), answerIndexesCD2D.tolist())

# test2: 3 objective functions
rank3D = frontUtils.rankNonDominatedFrontiers(test3D)
indexesCD3D = frontUtils.crowdingDistance(rank=rank3D, popSize=len(rank3D), fitness=test3D)
answerIndexesCD3D = np.array([np.inf, np.inf, 1.06417083, np.inf, np.inf,0.56135102, np.inf, np.inf, np.inf,np.inf])
checkArray('3D crowding distance', indexesCD3D.tolist(), answerIndexesCD3D.tolist())
###########################################
print(results)

sys.exit(results["fail"])

