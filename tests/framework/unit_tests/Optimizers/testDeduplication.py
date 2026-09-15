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
  Unit tests for RavenSampled submission deduplication: key generation, cache-based
  duplicate detection, and bounded cache eviction.
"""
import os
import sys
import importlib
import numpy as np

ravenPath = os.path.abspath(os.path.join(__file__, *['..'] * 5))
print('... located RAVEN at:', ravenPath)
sys.path.append(ravenPath)
from ravenframework.CustomDrivers import DriverUtils
DriverUtils.doSetup()
from ravenframework.Optimizers.GeneticAlgorithm import GeneticAlgorithm
# imported via importlib (not "import ... as"): the Optimizers package __init__ rebinds the
# attribute "RavenSampled" to the class, which would shadow the submodule otherwise
RavenSampledModule = importlib.import_module('ravenframework.Optimizers.RavenSampled')

results = {'pass': 0, 'fail': 0}

def checkTrue(comment, res, update=True):
  """
    Records whether a boolean condition holds.
    @ In, comment, str, description printed if it fails
    @ In, res, bool, the condition to check
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, the same value passed in
  """
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking bool', comment, '| got False, expected True')
      results['fail'] += 1
  return res

def checkSame(comment, value, expected, update=True):
  """
    Compares two values for equality.
    @ In, comment, str, description printed if it fails
    @ In, value, any, the value to compare
    @ In, expected, any, the expected value
    @ In, update, bool, optional, if False then don't update results counter
    @ Out, res, bool, True if same
  """
  res = value == expected
  if update:
    if res:
      results['pass'] += 1
    else:
      print('checking value', comment, '|', value, '!=', expected)
      results['fail'] += 1
  return res

def makeGA():
  """
    Builds a minimal GeneticAlgorithm instance with just enough state set to exercise
    dedup logic directly, bypassing the full XML handleInput path.
    @ In, None
    @ Out, ga, GeneticAlgorithm, minimal instance
  """
  ga = GeneticAlgorithm()
  ga.toBeSampled = {'x1': 'dummyDist', 'x2': 'dummyDist'}
  ga._deduplication = True
  return ga

#
# _makeSubmissionKey should key on exact values, not truncate to int
#
ga = makeGA()
keyA = ga._makeSubmissionKey({'x1': np.atleast_1d(0.1), 'x2': np.atleast_1d(0.2)})
keyB = ga._makeSubmissionKey({'x1': np.atleast_1d(0.9), 'x2': np.atleast_1d(0.8)})
checkTrue('distinct continuous points sharing an integer part must not collapse to the same key',
          keyA != keyB)
keyA2 = ga._makeSubmissionKey({'x1': np.atleast_1d(0.1), 'x2': np.atleast_1d(0.2)})
checkSame('identical point produces identical key', keyA, keyA2)

#
# _cacheEvaluatedSubmissionPoints + _queueSubmission: exact duplicate is skipped, near-duplicate is not
#
ga = makeGA()
pointA = {'x1': np.atleast_1d(0.1), 'x2': np.atleast_1d(0.2)}
pointB = {'x1': np.atleast_1d(0.15), 'x2': np.atleast_1d(0.2)}
ga._cacheEvaluatedSubmissionPoints({'x1': np.atleast_1d(0.1), 'x2': np.atleast_1d(0.2), 'obj': np.atleast_1d(1.0)})
queuedDuplicate = ga._queueSubmission(pointA, {'traj': 0})
queuedNew = ga._queueSubmission(pointB, {'traj': 0})
checkSame('exact duplicate submission is rejected from the queue', queuedDuplicate, False)
checkSame('non-duplicate submission is accepted into the queue', queuedNew, True)
checkSame('rejected duplicate is recorded for later restoration', len(ga._deduplicatedSubmissions), 1)
checkSame('only the accepted point remains in the submission queue', len(ga._submissionQueue), 1)

#
# _trimDedupCache: cache is capped at _MAX_DEDUP_CACHE_SIZE, oldest entries evicted first
#
ga = makeGA()
cap = RavenSampledModule._MAX_DEDUP_CACHE_SIZE
for i in range(cap + 5):
  ga._cacheEvaluatedSubmissionPoints({'x1': np.atleast_1d(float(i)), 'x2': np.atleast_1d(0.0), 'obj': np.atleast_1d(1.0)})
checkSame('dedup cache never grows past the configured cap', len(ga._evaluatedSubmissionData), cap)
oldestKey = ga._makeSubmissionKey({'x1': np.atleast_1d(0.0), 'x2': np.atleast_1d(0.0)})
newestKey = ga._makeSubmissionKey({'x1': np.atleast_1d(float(cap + 4)), 'x2': np.atleast_1d(0.0)})
checkTrue('oldest cached entry was evicted once the cap was exceeded', oldestKey not in ga._evaluatedSubmissionData)
checkTrue('most recently cached entry is retained', newestKey in ga._evaluatedSubmissionData)

#
# end
#
print('Results:', results)
sys.exit(results['fail'])
