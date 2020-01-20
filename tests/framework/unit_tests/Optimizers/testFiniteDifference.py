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
  Testing for FiniteDifference gradient approximation
"""
import os
import sys

import numpy as np

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 3))
print(ravenPath)
sys.path.append(ravenPath)
from utils.utils import find_crow
find_crow(ravenPath)

from Optimizers.gradients import returnInstance

fd = returnInstance('FiniteDifference', 'tester')

# checkers
def checkFloat(comment, value, expected, tol=1e-10, update=True):
  """
    This method compares two floats given a certain tolerance
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, tol, float, optional, the tolerance
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  if np.isnan(value) and np.isnan(expected):
    res = True
  elif np.isnan(value) or np.isnan(expected):
    res = False
  else:
    res = abs(value - expected) <= tol
  if update:
    if not res:
      print("checking float", comment, '|', value, "!=", expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

def checkSame(comment, value, expected, update=True):
  """
    This method compares two identical things
    @ In, comment, string, a comment printed out if it fails
    @ In, value, float, the value to compare
    @ In, expected, float, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results["pass"] += 1
    else:
      print("checking string", comment, '|', value, "!=", expected)
      results["fail"] += 1
  return res

results = {'pass': 0, 'fail': 0}

# initialization
optVars = ['a', 'b', 'c']
proximity = 0.01
fd.initialize(optVars, proximity)

checkSame('Check num vars', fd.N, 3)

# eval points
optPoint = {'a': 0.1,
            'b': 0.2,
            'c': 0.3}
stepSize = 0.5
pts, info = fd.chooseEvaluationPoints(optPoint, stepSize)
correct = [
    {'a': 0.095, 'b': 0.2, 'c': 0.3},
    {'a': 0.1, 'b': 0.205, 'c': 0.3},
    {'a': 0.1, 'b': 0.2, 'c': 0.295}]
cinfo = [
    {'type': 'grad', 'optVar': 'a', 'delta': -0.005},
    {'type': 'grad', 'optVar': 'b', 'delta':  0.005},
    {'type': 'grad', 'optVar': 'c', 'delta': -0.005}]

for p, pt in enumerate(pts):
  for v in ['a', 'b', 'c']:
    checkFloat('Point "{}" var "{}"'.format(p, v), pts[p][v], correct[p][v])
  for i in ['type', 'optVar']:
    checkSame('Point "{}" info "{}"'.format(p, i), info[p][i], cinfo[p][i])
  checkFloat('Point "{}" var "{}"'.format(p, 'delta'), info[p]['delta'], cinfo[p]['delta'])





print('Results:', results)
sys.exit(results['fail'])