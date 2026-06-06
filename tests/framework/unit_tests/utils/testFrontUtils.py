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
indexesCD2D = frontUtils.crowdingDistance(rank=rankCDSingleFront, popSize=len(rankCDSingleFront), objectiveValues=testCDarray)
answerIndexesCD2D = np.array([np.inf,0.16666667,0.11666667,0.08333333,0.09166667,0.125,0.16666667,0.29166667,0.45833333,0.625,0.75,0.5,0.20833333,0.11666667,0.125,np.inf])
checkArray('2D crowding distance', indexesCD2D.tolist(), answerIndexesCD2D.tolist())

# test2: 3 objective functions
rank3D = frontUtils.rankNonDominatedFrontiers(test3D)
indexesCD3D = frontUtils.crowdingDistance(rank=rank3D, popSize=len(rank3D), objectiveValues=test3D)
answerIndexesCD3D = np.array([np.inf, np.inf, 1.06417083, np.inf, np.inf,0.56135102, np.inf, np.inf, np.inf,np.inf])
checkArray('3D crowding distance', indexesCD3D.tolist(), answerIndexesCD3D.tolist())
###########################################

###########################################
# Fast non-dominated sort (Deb et al., 2002), constrained variant, O(M*N^2).
# It must produce ranks identical to the recursive peeling ranker it replaces.

def bruteForceConstrainedRanks(directed, violation):
  """
    Independent O(N^3) reference ranker used only to validate the fast sort.
    Peels fronts: a point joins the current front when no still-remaining point
    constrained-dominates it.
    @ In, directed, np.array, (nPoints, nObj) minimization-space objective values
    @ In, violation, np.array, (nPoints,) total positive constraint violation
    @ Out, ranks, list, 1-based front index for each point
  """
  nPoints = directed.shape[0]
  ranks = [0] * nPoints
  remaining = set(range(nPoints))
  rank = 0
  while remaining:
    rank += 1
    front = []
    for cand in sorted(remaining):
      dominated = any(
          frontUtils._dominatesForMinimization(directed[o], directed[cand], violation[o], violation[cand])
          for o in remaining if o != cand)
      if not dominated:
        front.append(cand)
    for idx in front:
      ranks[idx] = rank
      remaining.remove(idx)
  return ranks

# Known unconstrained 2-D case: three clear fronts.
fnsData = np.array([[1.0, 2.0],   # front 1
                    [2.0, 1.0],   # front 1
                    [2.0, 2.0],   # front 2 (dominated by both above)
                    [3.0, 3.0]])  # front 3
fnsViol = np.zeros(4)
checkArray('fast NDS unconstrained ranks',
           frontUtils._fastNonDominatedSortConstrained(fnsData, fnsViol),
           [1, 1, 2, 3])

# Constrained case: the middle point is infeasible (g < 0) so feasible points
# dominate it regardless of objective values.
fnsConData = np.array([[1.0, 1.0],   # feasible
                       [0.0, 0.0],   # infeasible, would dominate all if feasible
                       [2.0, 2.0]])  # feasible
fnsConViol = np.array([0.0, 5.0, 0.0])
checkArray('fast NDS constrained ranks',
           frontUtils._fastNonDominatedSortConstrained(fnsConData, fnsConViol),
           [1, 3, 2])

# Public ranker on the same constrained problem (all objectives minimized).
checkArray('rankNonDominatedFrontiers constrained',
           frontUtils.rankNonDominatedFrontiers(fnsConData,
                                                 constraintVals=np.array([[1.0], [-5.0], [2.0]]),
                                                 minMask=np.array([True, True])),
           [1, 3, 2])

# Randomized equivalence vs the brute-force reference (objectives + constraints).
rng = np.random.RandomState(12345)
for trial in range(25):
  nPts = int(rng.randint(4, 20))
  nObj = int(rng.randint(2, 4))
  rData = rng.rand(nPts, nObj)
  rViol = np.where(rng.rand(nPts) < 0.4, rng.rand(nPts) * 3.0, 0.0)
  checkArray('fast NDS equivalence trial %d' % trial,
             frontUtils._fastNonDominatedSortConstrained(rData, rViol),
             bruteForceConstrainedRanks(rData, rViol))
###########################################

###########################################
# Hypervolume indicator (minimization space)
# Validated against hand-computed exact values.
# 2-D unit square: one point at the origin, reference at (1,1) -> area 1
checkAnswer('HV 2D unit square', frontUtils.hypervolume([[0.0, 0.0]], [1.0, 1.0]), 1.0)
# 2-D three-point front {(1,3),(2,2),(3,1)} with reference (4,4): union of the three
# dominated boxes has area 6 (inclusion-exclusion: 3+4+3-2-2-1+1).
checkAnswer('HV 2D three-point front', frontUtils.hypervolume([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], [4.0, 4.0]), 6.0)
# 2-D two-point front {(1,3),(2,2)} with reference (4,4): 3+4-2 = 5.
checkAnswer('HV 2D two-point front', frontUtils.hypervolume([[1.0, 3.0], [2.0, 2.0]], [4.0, 4.0]), 5.0)
# 3-D unit cube: one point at the origin, reference at (1,1,1) -> volume 1
checkAnswer('HV 3D unit cube', frontUtils.hypervolume([[0.0, 0.0, 0.0]], [1.0, 1.0, 1.0]), 1.0)
# Negative objectives (common after max->min sign conversion): point (-2,-2), reference (-1,-1) -> 1
checkAnswer('HV negative objectives', frontUtils.hypervolume([[-2.0, -2.0]], [-1.0, -1.0]), 1.0)
# A point that does not dominate the reference contributes nothing to the hypervolume.
checkAnswer('HV dominated point ignored', frontUtils.hypervolume([[0.0, 0.0], [5.0, 5.0]], [1.0, 1.0]), 1.0)
###########################################
print(results)

sys.exit(results["fail"])

