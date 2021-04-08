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
  This Module performs Unit Tests for the frontUtils.rankNonDominantFrontiers methods
"""


import os,sys
import numpy as np
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import utils
utils.find_crow(frameworkDir)
from utils import frontUtils

randomENG = utils.findCrowModule("randomENG")

print (frontUtils)

results = {"pass":0,"fail":0}

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

test = np.array([[ 0.21573114, -0.92937786,  0.29952775],
                 [ 0.94716548, -0.31085637, -0.07903087],
                 [ 0.6485263,  -0.72106429,  0.24388507],
                 [ 0.3466882,  -0.78716832,  0.51007189],
                 [ 0.15463182, -0.18730138,  0.97005525],
                 [ 0.02937279, -0.91175393,  0.40968525],
                 [0.35039731,  0.54889384,  0.80057772],
                 [ 0.06213356,  0.28552822, -0.95635404],
                 [-0.20190017,  0.66695686, -0.71722024],
                 [-0.62399932, -0.22858416,  0.74724436]])

rankedFrontiers = frontUtils.rankNonDominatedFrontiers(test)
answerRanking = [1, 1, 1, 2, 2, 1, 3, 1, 1, 1]
checkArray('nonDominatedFrontier with indexes', rankedFrontiers, answerRanking)

print(results)

sys.exit(results["fail"])

