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
import os
import sys

import numpy as np
import cloudpickle as pk

# add romLoader to path
here = os.path.abspath(os.path.dirname(__file__))
frameworkPath = os.path.abspath(os.path.join(here, *['..']*4, 'framework'))
sys.path.append(os.path.abspath(os.path.join(frameworkPath, '..', 'scripts')))
import externalROMloader

results = {'pass': 0, 'fail': 0}

def check(comment, value, expected, tol=1e-10, update=True):
  """
    Compare two floats given a certain tolerance
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
      print("checking float",comment,'|',value,"!=",expected)
      results["fail"] += 1
    else:
      results["pass"] += 1
  return res

#
# create
#
targetFile = os.path.join(here, 'StochasticPolyPickleTest', 'ROMpk')
runner = externalROMloader.ravenROMexternal(targetFile, frameworkPath)
#
# serialize
#
x = pk.dumps(runner)
results['pass'] += 1 # success for pickling
pk.loads(x)
results['pass'] += 1 # success for upickling
#
# run
#
inp = {'x1': [3], 'x2': [5]}
res = runner.evaluate(inp)[0]
check('Evaluate x[-1]', res['ans'][-1], 8)


#
# results
#
print(results)
sys.exit(results['fail'])
