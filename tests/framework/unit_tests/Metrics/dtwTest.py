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
  Testing for the DTW distance metrics
  @authors: mandd
"""
import os
import sys
import numpy as np

ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(ravenDir)
frameworkDir = os.path.join(ravenDir, 'framework')

from ravenframework.Metrics.metrics import DTW
from ravenframework import MessageHandler
#from tests.framework.unit_tests.utils.testUtils import checkArray

mh = MessageHandler.MessageHandler()
mh.initialize({'verbosity':'debug'})

#
#
# checkers
#

results = {"pass":0,"fail":0}

def checkAnswerDTW(comment,value,expected,tol=1e-10,relative=False):
  """
    This method is aimed to compare two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float (or string), the value to compare
    @ In, expected, float (or string), the expected value
    @ In, tol, float, optional, the tolerance (valid for floats/ints only)
    @ In, relative, bool, optional, the tolerance needs be checked relative?
    @ Out, None
  """
  isFloat = True
  try:
    val, expect = float(value), float(expected)
  except ValueError:
    val, expect = value, expected
    isFloat = False
  if relative and isFloat:
    denominator = expect if expect != 0. else 1.0
  if isFloat:
    diff = abs(val - expect) if not relative else abs(val - expect)/denominator
  else:
    diff = 0.0 if val == expect else tol + 1.0

  if diff > tol:
    print("checking answer",comment,val,"!=",expect)
    results["fail"] += 1
  else:
    results["pass"] += 1

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

DTWinstance = DTW.DTW()
DTWinstance.order=0
DTWinstance.localDistance = "euclidean"

#
#
# initialize test
#
s1 = np.array([1, 2, 3, 2, 2.13, 1])
s2 = np.array([1, 1, 2, 2, 2.42, 3, 2, 1])

DTWdistance, path = DTWinstance.run(s1, s2, returnPath=True)

## TESTING
# Test DTW distance metric
expectedDTWdistance = 0.5499999999999998
expectedPath = np.array([[0, 0],
 [0, 1],
 [1, 2],
 [1, 3],
 [1, 4],
 [2, 5],
 [3, 6],
 [4, 6],
 [5, 7]])

checkAnswerDTW('DTW analytical test: distance', DTWdistance, expectedDTWdistance)
checkArray('DTW analytical test: path (1)',  path[:,0],   expectedPath[:,0])
checkArray('DTW analytical test: path (2)',  path[:,1],   expectedPath[:,1])

s12d = np.array([[0, 0],
                  [0, 1],
                  [1, 2],
                  [1, 3],
                  [1, 4],
                  [2, 5],
                  [3, 6],
                  [4, 6],
                  [5, 7]])

s22d = 2*np.array([[0, 0],
                    [0, 1],
                    [1, 2],
                    [1, 3],
                    [3, 6],
                    [5, 7]])

DTWdistance2D = DTWinstance.run(s12d.T, s22d.T)
expectedDTWdistance2D = 21.34109414975248
checkAnswerDTW('2D DTW analytical test: distance', DTWdistance2D, expectedDTWdistance2D)
#
# end
#
print('Results:', results)
sys.exit(results['fail'])
