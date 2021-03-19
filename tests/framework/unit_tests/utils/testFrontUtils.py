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
frameworkDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir,'framework'))
sys.path.append(frameworkDir)
from utils import utils
utils.find_crow(frameworkDir)
from utils import frontUtils

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
 [9.16351429e-01, 4.29517734e-01],
 [8.01561839e-01, 8.69414390e-01],
 [7.55562523e-01, 9.32146038e-01],
 [8.90900310e-01, 3.67593058e-01],
 [7.50520934e-01, 7.51391197e-01],
 [8.17338177e-01, 4.96351181e-01],
 [7.86831619e-01, 5.26530023e-01],
 [6.25333898e-01, 8.46377250e-01],
 [9.69178609e-01, 6.40012769e-02],
 [9.59699363e-01, 7.72487767e-02],
 [9.90078478e-01, 1.76166679e-02],
 [5.68376060e-01, 7.12008835e-01],
 [5.75488334e-01, 6.52163917e-01],
 [5.02251555e-01, 7.13559825e-01],
 [4.14678064e-01, 7.84278070e-01],
 [5.01732785e-01, 6.24807325e-01],
 [7.00855974e-01, 3.51391595e-01],
 [4.75249540e-01, 5.77808575e-01],
 [9.17718546e-01, 8.49623570e-02],
 [5.40313150e-01, 4.45181888e-01],
 [8.93878716e-01, 9.63765427e-02],
 [2.80685671e-01, 6.12361216e-01],
 [5.92002553e-01, 3.25367107e-01],
 [6.67684273e-01, 2.48013673e-01],
 [7.83018827e-01, 1.51356673e-01],
 [3.51123954e-01, 4.22371499e-01],
 [6.09926345e-01, 2.36464942e-01],
 [5.04712868e-01, 2.78940516e-01],
 [9.83340976e-01, 8.69100217e-03],
 [4.43529645e-01, 2.67981999e-01],
 [4.40291945e-01, 2.47766153e-01],
 [4.28153540e-01, 2.31514973e-01],
 [1.14322230e-01, 3.25937606e-01],
 [3.52914527e-01, 2.14847179e-01],
 [6.55351333e-01, 1.02289915e-01],
 [5.77261236e-01, 1.10855158e-01],
 [3.19859875e-01, 1.55237545e-01],
 [9.40885137e-01, 1.15126186e-02],
 [3.41780045e-01, 1.06415898e-01],
 [3.34868058e-01, 8.57592673e-02],
 [7.90589593e-01, 2.02015701e-02],
 [8.71338595e-01, 8.26033625e-03],
 [6.84733711e-01, 1.01099763e-02],
 [6.36289229e-01, 4.45417231e-17]])

indexes2D = frontUtils.nonDominatedFrontier(test2D, returnMask=False, minMask=np.array([False,True]))
answerIndexes = np.array([0, 16, 34, 47, 49])
checkArray('2D nonDominatedFrontier MinMask with indexes', indexes2D.tolist(), answerIndexes.tolist())



print(results)

sys.exit(results["fail"])

