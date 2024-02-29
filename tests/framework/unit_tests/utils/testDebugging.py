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
  This Module performs Unit Tests for the Debugging methods
  It cannot be considered part of the active code but of the regression test system
"""

#For future compatibility with Python 3
import os, sys
import io
from collections import deque
from contextlib import redirect_stdout

import numpy as np

ravenDir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),os.pardir,os.pardir,os.pardir,os.pardir))
sys.path.append(ravenDir)
from ravenframework.utils import Debugging

print(Debugging)

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

def checkAnswer(comment, value, expected, tol=10, updateResults=True):
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

# check size of simple objects
# TODO how much variance on different machines/OS?
checkAnswer('getSize int', Debugging.getSize(42), 28)
checkAnswer('getSize int large', Debugging.getSize(int(1e30)), 40)
checkAnswer('getSize float', Debugging.getSize(42.314), 24)
checkAnswer('getSize np.float32', Debugging.getSize(np.float32(42.314)), 28)
checkAnswer('getSize np.float64', Debugging.getSize(np.float64(42.314)), 32)
checkAnswer('getSize str', Debugging.getSize('hello world!'), 61)
checkAnswer('getSize bool', Debugging.getSize(True), 28)
# list-like; tolerances determined mostly by windows test machine
checkAnswer('getSize tuple', Debugging.getSize(tuple(range(10))), 404)
checkAnswer('getSize list', Debugging.getSize(list(range(10))), 476,tol=80)
checkAnswer('getSize deque', Debugging.getSize(deque(range(10))), 908)
checkAnswer('getSize set', Debugging.getSize(set(range(10))), 1012)
checkAnswer('getSize np array int10', Debugging.getSize(np.arange(10, dtype=int)), 176, tol=40)
checkAnswer('getSize np array int100', Debugging.getSize(np.arange(100, dtype=int)), 896, tol=400)
checkAnswer('getSize np array float', Debugging.getSize(np.arange(10, dtype=float)), 176, tol=20)
checkAnswer('getSize np array bool', Debugging.getSize(np.ones(10, dtype=bool)), 106, tol=20)
checkAnswer('getSize np array object', Debugging.getSize(np.arange(10, dtype=object)), 452, tol=20)
checkAnswer('getSize np array flat', Debugging.getSize(np.arange(24)), 288, tol=100)
checkAnswer('getSize np array shaped', Debugging.getSize(np.arange(24).reshape(2,3,4)), 128, tol=20)
# dict-like
a = dict((i, np.arange(i)) for i in range(10))
checkAnswer('getSize dict nparray', Debugging.getSize(a), 1964, tol=200)

# list of lists
a = []
for i in range(10):
  a.append(range(i))
checkAnswer('getSize list of lists', Debugging.getSize(a), 672)

# numpy object array
a = np.array([None]*10, dtype=object)
for i in range(10):
  a[i] = np.arange(i)
checkAnswer('getSize nparray 2 nested object', Debugging.getSize(a), 1496, tol=200)

b = np.array([None, None, None], dtype=object)
b[0] = a[:3]
b[1] = a[:7]
b[2] = a[:]
checkAnswer('getSize nparray 3 nested object', Debugging.getSize(b), 1728, tol=300)

# unchecked: classes, modules, probably a lot of other things




# checkSizesWalk does the same kind of thing, but prints instead of returning
f = io.StringIO()
with redirect_stdout(f):
  Debugging.checkSizesWalk(b, tol=100)
s = f.getvalue()
expected = """->  (<class 'numpy.ndarray'>): 1.7e+03
  -> [0] (<class 'numpy.ndarray'>): 4.1e+02
    -> [0][1] (<class 'numpy.ndarray'>): 1.0e+02
    -> [0][2] (<class 'numpy.ndarray'>): 1.1e+02
  -> [1] (<class 'numpy.ndarray'>): 9.4e+02
    -> [1][1] (<class 'numpy.ndarray'>): 1.0e+02
    -> [1][2] (<class 'numpy.ndarray'>): 1.1e+02
    -> [1][3] (<class 'numpy.ndarray'>): 1.2e+02
    -> [1][4] (<class 'numpy.ndarray'>): 1.3e+02
    -> [1][5] (<class 'numpy.ndarray'>): 1.4e+02
    -> [1][6] (<class 'numpy.ndarray'>): 1.4e+02
  -> [2] (<class 'numpy.ndarray'>): 1.4e+03
    -> [2][1] (<class 'numpy.ndarray'>): 1.0e+02
    -> [2][2] (<class 'numpy.ndarray'>): 1.1e+02
    -> [2][3] (<class 'numpy.ndarray'>): 1.2e+02
    -> [2][4] (<class 'numpy.ndarray'>): 1.3e+02
    -> [2][5] (<class 'numpy.ndarray'>): 1.4e+02
    -> [2][6] (<class 'numpy.ndarray'>): 1.4e+02
    -> [2][7] (<class 'numpy.ndarray'>): 1.5e+02
    -> [2][8] (<class 'numpy.ndarray'>): 1.6e+02
    -> [2][9] (<class 'numpy.ndarray'>): 1.7e+02
""".split('\n')
# for l, line in enumerate(s.split('\n')):
#  checkTrue(f'checkSizesWalk[{l}]', line.strip(), expected[l].strip())
# TODO Windows is too different to test these output lines.

print(results)

sys.exit(results["fail"])
"""
  <TestInfo>
    <name>framework.Debugging</name>
    <author>talbpaul</author>
    <created>2020-01-18</created>
    <classesTested>utils.Debugging</classesTested>
    <description>
       This test performs Unit Tests for the Debugging methods
       It cannot be considered part of the active code but of the regression test system
    </description>
  </TestInfo>
"""
